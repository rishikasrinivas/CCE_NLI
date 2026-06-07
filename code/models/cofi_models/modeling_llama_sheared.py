import logging
import math
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutputWithPast
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaPreTrainedModel, LlamaAttention
from transformers.cache_utils import Cache, DynamicCache
from llm2vec.models.bidirectional_llama import LlamaBiModel, ModifiedLlamaDecoderLayer, ModifiedLlamaAttention
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from cofi.utils.cofi_utils import *
from einops import rearrange

logger = logging.getLogger(__name__)


#replaced the rms norm and moved logits to gpu
class CoFiLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device: Optional[str] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.variance_epsilon = eps
    
    def prune_params(self, hidden_z):
        remaining_index = torch.where(~hidden_z.eq(0))[0]
        self.weight = torch.nn.Parameter(self.weight.data.mul(hidden_z.squeeze())[remaining_index])

    def forward(self, hidden_states, hidden_z=None): 
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(hidden_states, dim=-1, index=remaining_index)
        else:
            compressed_input = hidden_states
        variance = compressed_input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        output = self.weight * hidden_states
        if hidden_z is not None:
            #print(f"In RMS norm, after rms norm d hidden states the result is {output.shape} and hiddenz is {hidden_z.shape}")
            output = output.mul(hidden_z)
        return output
def normal_attn_fn(
    query,
    key, 
    value,
    attention_mask=None,
    head_z=None
):
    bsz, n_heads, q_len, head_dim = query.shape
    dim = n_heads * head_dim
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value) # (bsz, n_heads, q_len, head_dim)
    if head_z is not None:
    
        attn_output *= head_z.unsqueeze(-1)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, dim)
    return attn_output
def flash_attn_fn(
    query,
    key,
    value,
    softmax_scale=None,
    attn_bias=None,
    query_padding_mask=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    head_z=None,

):
    try:
        from flash_attn import bert_padding  # type: ignore
        from flash_attn import flash_attn_interface  # type: ignore
    except ImportError as e:
        raise e

    # check_valid_inputs(query, key, value)

    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')

    batch_size, seqlen = query.shape[:2]

    if query_padding_mask is None:
        query_padding_mask = torch.ones((batch_size, seqlen), dtype=torch.bool, device=query.device)
    if key_padding_mask is None:
        key_padding_mask = torch.ones((batch_size, seqlen), dtype=torch.bool, device=key.device)

    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = bert_padding.unpad_input(
        query, query_padding_mask)
    # query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(
        key, key_padding_mask)
    # key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    # value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    dropout_p = dropout_p if training else 0.0
    
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=is_causal,
        return_attn_probs=needs_weights)

    if head_z is not None:
        output_unpad = output_unpad * head_z # 1 * h * 1
    output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    return output, None
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CoFiLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.attn_impl = "eager"

        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        assert self.num_key_value_groups == 1

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.wq = nn.Linear(self.d_model, self.q_size, bias=False)
        self.wk = nn.Linear(self.d_model, self.kv_size, bias=False)
        self.wv = nn.Linear(self.d_model, self.kv_size, bias=False)
        self.out_proj = nn.Linear(self.q_size, self.d_model, bias=False)

        self.softmax_scale = self.head_dim ** -0.5
        self.attn_dropout_p = 0.0
    
    
    def prune_params(self, zs_block):
        head_z = None; head_layer_z = None; hidden_z = None; qk_head_dim_z = None; vo_head_dim_z = None
        if "head_z" in zs_block:
            head_z = zs_block["head_z"].squeeze()
        
        if "head_layer_z" in zs_block:
            head_layer_z = zs_block["head_layer_z"].squeeze()
        
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"].squeeze()
        
        if "qk_head_dim_z" in zs_block:
            qk_head_dim_z = zs_block["qk_head_dim_z"].squeeze() # qk_head_dim is the same as hidden_z
            vo_head_dim_z = zs_block["vo_head_dim_z"].squeeze() # vo_head_dim is the same as hidden_z
            
            
        # update params #
        if head_z is not None:
            head_z_for_update = torch.repeat_interleave(head_z, self.head_dim)
            self.wv.weight.data = self.wv.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
        if head_layer_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
        if hidden_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
        if qk_head_dim_z is not None:
            self.wq.weight.data = self.wq.weight.data.transpose(0, 1).mul(qk_head_dim_z).transpose(0, 1)
            self.wv.weight.data = self.wv.weight.data.transpose(0, 1).mul(vo_head_dim_z).transpose(0, 1)
        #################
        
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    Head hidden: {len(hidden_z)} -> {len(remaining_index)}") 
            half = next(self.wq.parameters()).dtype == torch.float16
            self.wk = prune_linear_layer(self.wk, remaining_index, dim=1)
            self.wq= prune_linear_layer(self.wq, remaining_index, dim=1)
            self.wv = prune_linear_layer(self.wv, remaining_index, dim=1)
            self.out_proj = prune_linear_layer(self.out_proj, remaining_index)
            if half:
                self.wq.half()
                self.wk.half()
                self.wv.half()
                self.out_proj.half()
         
        to_prune_heads = turn_head_z(head_z, head_layer_z)
        len_to_prune_heads = len(to_prune_heads)
        if len_to_prune_heads == 0:
            print(f"    Heads: {self.n_heads} -> {self.n_heads}")
            return

        heads, index = find_pruneable_heads_and_indices(
            to_prune_heads, self.n_heads, self.head_dim, self.pruned_heads
        )
        
        qk_index = index; vo_index = index
        if qk_head_dim_z is not None:
            remaining_qk_index = torch.where(~qk_head_dim_z.eq(0))[0]
            remaining_vo_index = torch.where(~vo_head_dim_z.eq(0))[0]
            import numpy as np
            qk_index = torch.from_numpy(np.intersect1d(index.detach().cpu().numpy(), remaining_qk_index.detach().cpu().numpy())).to(index.device).to(index.dtype)
            vo_index = torch.from_numpy(np.intersect1d(index.detach().cpu().numpy(), remaining_vo_index.detach().cpu().numpy())).to(index.device).to(index.dtype)
            print(f"    QKVO dims: {len(hidden_z)} -> {len(qk_index)}")
        
        # Prune linear layers
        # setting layers to be None if all the heads are pruned
        if len(index) == 0:
            self.wq = None
            self.wk = None
            self.wv = None
            self.out_proj = None
        else:
            half = next(self.wq.parameters()).dtype == torch.float16
            self.wq = prune_linear_layer(self.wq, qk_index)
            self.wk = prune_linear_layer(self.wk, qk_index)
            self.wv = prune_linear_layer(self.wv, vo_index)
            self.out_proj = prune_linear_layer(self.out_proj, vo_index, dim=1)
            if half:
                self.wq.half()
                self.wk.half()
                self.wv.half()
                self.out_proj.half()

        print(f"    Heads: {self.n_heads} -> {self.n_heads - len(heads)}")

        # Update hyper params and store pruned heads
        self.n_heads = self.n_heads - len(heads)
        self.all_head_size = self.head_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
            
            

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_z=None,  # Prune KV heads
        head_layer_z=None,  # Prune entire attention layer
        hidden_z = None,
        qk_head_dim_z=None,
        vo_head_dim_z=None,
        **kwargs,
    ):
        if self.v_proj is None: #only return none if the final proj is pruned out, othewise just apply the mask and see for youself

            return (None, None) if output_attentions else (None, None) #shud ret hidden 

        bsz, q_len, _ = hidden_states.size()
        
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if qk_head_dim_z is not None:
            query_states = query_states.mul(qk_head_dim_z)
            value_states = value_states.mul(vo_head_dim_z)
        
        '''query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)'''
        

        actual_kv_heads = key_states.shape[-1] // self.head_dim      # e.g. 192//64 = 3
        actual_q_heads  = query_states.shape[-1] // self.head_dim    # e.g. 1536//64 = 24
        actual_kv_groups = actual_q_heads // actual_kv_heads
        
        # Reshape for GQA
        query_states = query_states.view(bsz, q_len, actual_q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, actual_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, actual_kv_heads, self.head_dim).transpose(1, 2)
        
        assert actual_q_heads == self.num_heads, f'Calculted Q heads is {actual_q_heads} and self num heads is {self.num_heads}'
        assert actual_kv_heads == self.num_key_value_heads, f'Calculted KV heads is {actual_kv_heads} and self kv heads is {self.num_key_value_heads}'
        assert actual_kv_groups == self.num_key_value_groups, f'Calculted KV groups is {actual_kv_groups} and self num kv groups is {self.num_key_value_groups}'
  
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update cache if needed
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat KV heads for grouped query attention
        '''key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)'''
        
        key_states = self.repeat_kv(key_states, actual_kv_groups)
        value_states = self.repeat_kv(value_states, actual_kv_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        

        if head_z is not None:
            #torch.Size([1, 4, 1])
            # head_z is over KV heads, need to expand to query heads
            head_z = head_z.squeeze()
          
            #head_z_expanded = head_z.repeat_interleave(self.num_key_value_groups)
            head_z_expanded = head_z.repeat_interleave(actual_kv_groups)
            
            #print(f"In attention pruning the kv heads norm the atttnout is  {attn_output.shape} and head is expanded to {head_z_expanded.view(1, -1, 1, 1).shape}")
            
            attn_output = attn_output * head_z_expanded.view(1, -1, 1, 1)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        if head_layer_z is not None:
            #print(f"In attention layer {attn_output.shape} and head layer is {head_layer_z.shape}")
            
            attn_output = attn_output.mul(head_layer_z)
            
            
        if hidden_z is not None:
            #print(f"In attention layer {attn_output.shape} and hidden mask is {hidden_z.shape}")
            
            attn_output = attn_output.mul(hidden_z)
            
       
     
        
        return (attn_output, attn_weights) if output_attentions else (attn_output, None )
    

    def repeat_kv(self, hidden_states, n_rep):
        # [B, num_kv_heads, S, D] -> [B, num_heads, S, D]
        b, h_kv, s, d = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(b, h_kv, n_rep, s, d)
        return hidden_states.reshape(b, h_kv * n_rep, s, d)


   


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def prepare_decoder_attention_mask(input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype).to(inputs_embeds.device)

    return combined_attention_mask
class CoFiModifiedLlamaDecoderLayer(ModifiedLlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
        # Replace attention with CoFi-enabled version
        self.self_attn = CoFiLlamaAttention(config, layer_idx)
        
        # Use CoFi-enabled RMSNorm
        self.input_layernorm = CoFiLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CoFiLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # We'll also need CoFi MLP - let's create it
        self.mlp = CoFiModifiedLlamaMLP(config)
    
    
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None,
        **kwargs,
    ):
        residual = hidden_states

        # -------------------------
        # PRE-NORM ATTENTION INPUT
        # -------------------------
        attn_input = self.input_layernorm(hidden_states, hidden_z)

        attn_out, attn_weights = self.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
            qk_head_dim_z=qk_head_dim_z,
            vo_head_dim_z=vo_head_dim_z,
            **kwargs,
        )

        # -------------------------
        # RESIDUAL ADD (ATTN)
        # -------------------------
        if attn_out is None: #if attnout is none,  the input to the dcoder moves onto the mlp
            hidden_states = residual #if attn was pruned out pass forward just what was passed in as input 
        else: 
            hidden_states = residual + attn_out # otherwise add the attn out (ie ignore the layernomr)


        # =========================
        # MLP BLOCK
        # =========================
        residual = hidden_states

        mlp_input = self.post_attention_layernorm(hidden_states, hidden_z)

        mlp_out = self.mlp(
            mlp_input,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )
        if mlp_out.sum().eq(0).item():
            hidden_states = residual # if mlp pruned ignore the layer norm use the result from the outout the self attn (whatever ended up after self attn whether it was just residual or also self attn)

        else:
            hidden_states = residual + mlp_out


        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)



class CoFiModifiedLlamaMLP(nn.Module):
    def __init__(
        self,
        config,
   
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x, intermediate_z=None, mlp_z=None, hidden_z=None):
        gate = self.gate_proj(x)
        
        up = self.up_proj(x)
        if intermediate_z is not None:
            #print(f"In after upproj is {up.shape} and intermediate_z mask is {intermediate_z.shape}")
            
            up *= intermediate_z
        x = self.act_fn(gate) * up
        #print(f"self.act_fn(gate) * up is doing {self.act_fn(gate).shape} * {up.shape} = {x.shape}")
        
        x = self.down_proj(x)
        if mlp_z is not None:
            #print(f"In after upproj is {x.shape} and mlp mask is {mlp_z.shape}")
            
            x = x * mlp_z
        if hidden_z is not None:
            #print(f"In after upproj is {x.shape} and hidden mask is {hidden_z.shape}")
            
            x = x * hidden_z
            
        return x

class CoFiLlamaBiModel(LlamaBiModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace decoder layers with CoFi-enabled versions
        self.layers = nn.ModuleList(
            [CoFiModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # Replace norm with CoFi-enabled version
        self.norm = CoFiLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        head_z=None,  # List of tensors, one per layer
        head_layer_z=None,  # List of tensors, one per layer
        intermediate_z=None,  # List of tensors, one per layer
        mlp_z=None,  # List of tensors, one per layer
        hidden_z=None,  # Single tensor shared across layers
        qk_head_dim_z=None,
        vo_head_dim_z=None
   
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Input embedding
        if inputs_embeds is None:
            try:
                inputs_embeds = self.embed_tokens(input_ids)
            except:
                print("inp ", input_ids.device)
                assert False
            
        # apply hidden mask to embeddings
        if hidden_z is not None:
            inputs_embeds *= hidden_z
            #print(f"Input embeds shape = {inputs_embeds.shape}")
        
        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # Prepare attention mask (bidirectional, so no causal mask needed)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        # Now call _update_causal_mask (from parent LlamaBiModel)
        attention_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions
        )
        
        hidden_states = inputs_embeds
        
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # Process through layers
        for idx, decoder_layer in enumerate(self.layers):
        
            if output_hidden_states:  
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get layer-specific masks
            layer_head_z = head_z[idx] if head_z is not None else None
            layer_head_layer_z = head_layer_z[idx] if head_layer_z is not None else None
            layer_intermediate_z = intermediate_z[idx] if intermediate_z is not None else None
            layer_vo_head_dim_z = vo_head_dim_z[idx] if vo_head_dim_z is not None else None
            layer_qk_head_dim_z = qk_head_dim_z[idx] if qk_head_dim_z is not None else None
            layer_mlp_z = mlp_z[idx] if mlp_z is not None else None
            #if hidden_z is not None:
               # print(f"Passing through decoder layer {idx}")
            '''if layer_head_z is not None:
                print(f"head is {layer_head_z.shape}")
            if layer_head_layer_z is not None:
                print(f"layer_head_layer_z is {layer_head_layer_z.shape}")
            if layer_intermediate_z is not None:
                print(f"layer_intermediate_z is {layer_intermediate_z.shape}")
            if layer_mlp_z is not None:
                print(f"layer_mlp_z is {layer_mlp_z.shape}")
            if hidden_z is not None:
                print(f"Hidden shape is {hidden_z.shape}")'''
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                head_z=layer_head_z,
                head_layer_z=layer_head_layer_z,
                intermediate_z=layer_intermediate_z,
                mlp_z=layer_mlp_z,
                hidden_z=hidden_z,
                vo_head_dim_z=layer_vo_head_dim_z,
                qk_head_dim_z=layer_qk_head_dim_z
                
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
        
        # Final norm
        
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  
        hidden_states = self.norm(hidden_states, hidden_z)
        

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
        
            self.layers[layer].self_attn.prune_heads(heads)
class CoFiLlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name='llama'
        
        self.config=config
        self.model = CoFiLlamaBiModel(config)
   
        self.tokenizer = AutoTokenizer.from_pretrained('knowledgator/Sheared-LLaMA-1.3B')
        if "pad_token" not in self.tokenizer.special_tokens_map:
            num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if num_new_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.encoder_dim = config.hidden_size
        self.mlp_input_dim = self.encoder_dim * 4
        
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),
        )
        
        
        
        self.do_layer_distill = getattr(config, "do_layer_distill", False)

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(
                config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None
        self.num_labels=3
    
    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`dict[int, list[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.model._prune_heads(heads_to_prune)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs
    ):
        print(pretrained_model_name_or_path)

        # -----------------------
        # Load config
        # -----------------------
        if "config" not in kwargs:
            config = AutoConfig.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B")
            config.do_layer_distill = False
        else:
            config = kwargs["config"]

        # Create model (random MLP + CoFi)
        model = cls(config)
        trained=False

        # -----------------------
        # Load .pth checkpoint
        # -----------------------
        if pretrained_model_name_or_path and ".pth" in str(pretrained_model_name_or_path) and os.path.exists(pretrained_model_name_or_path):
            print(f"Loading pretrained llama entailment ({pretrained_model_name_or_path})")
            weights = torch.load(pretrained_model_name_or_path, map_location=kwargs['device'])["state_dict"]

            # Convert old gamma/beta to weight/bias
            old_keys, new_keys = [], []
            for key in list(weights.keys()):
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                elif "beta" in key:
                    new_key = key.replace("beta", "bias")
                else:
                    continue
                old_keys.append(key)
                new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                weights[new_key] = weights.pop(old_key)

            load_pruned_model(model, weights)
            trained = True
            return model, trained
        elif kwargs['ckpt'] is not None and os.path.exists(kwargs['ckpt']):
            
            weights = torch.load(kwargs['ckpt'], map_location=kwargs['device'])['state_dict']
            model.load_state_dict(weights, strict=False)
            assert trained==False
            model.to(kwargs['device'])
            print("student on ", model.device)
            return model, trained

        # -----------------------
        # Load HF pretrained encoder only
        # -----------------------
        print("Loading HF princeton-nlp/Sheared-LLaMA-1.3B")
        hf_encoder = LlamaBiModel.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B")

        # Filter HF weights to match model (skip classifier / MLP)
  
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in model_state.items()}

        model.load_state_dict(filtered_state)
        from pathlib import Path

        Path(kwargs['ckpt']).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict':model.state_dict()}, kwargs['ckpt'])
    
        return model.to(kwargs['device']), trained




    def indices_to_bert_tokens(self, indices):
        batch_size, seq_len = indices.shape
        words = []
        for i in range(batch_size):
            sentence = []
            for idx in indices[i]:
                if idx.item() in self.vocab['itos'].keys():
                    word = self.vocab['itos'][idx.item()]
                    if word not in ("[PAD]", "<pad>", "PAD"): 
                        sentence.append(word)
                else:
                    break
            words.append(sentence)
      

            return self.tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)


    def forward(
            self,
            pre_input_ids=None,
            pre_attention_mask=None,
            hyp_input_ids=None,
            hyp_attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            past_key_values = None,
            output_attentions=None,
            output_hidden_states=None,
            head_z=None,
            head_layer_z=None,
            intermediate_z=None,
            mlp_z=None,
            hidden_z=None,
            final_mlp_hidden_z=None,
            final_mlp_inp_z=None,
            qk_head_dim_z=None,
        vo_head_dim_z=None
    ):
  
        #pre_input_ids = pre_input_ids.unsqueeze(0)  # Shape becomes [1, 46]
        #pre_attention_mask = pre_attention_mask.unsqueeze(0)
        

        #if final_mlp_hidden_z is not None:
            #print(f"finalmlp  shape is {final_mlp_hidden_z.shape}")
        outputs_pre = self.model (
            pre_input_ids,
            attention_mask=pre_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
         
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
            qk_head_dim_z=qk_head_dim_z,
            vo_head_dim_z=vo_head_dim_z
        ) #! [32, 68, 768]
        
        
        #hyp_input_ids = hyp_input_ids.unsqueeze(0)  # Shape becomes [1, 46]
        #hyp_attention_mask = hyp_attention_mask.unsqueeze(0)
        outputs_hyp = self.model(
            hyp_input_ids,
            attention_mask=hyp_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
  
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
            qk_head_dim_z=qk_head_dim_z,
            vo_head_dim_z=vo_head_dim_z
        )
        '''if mlp_z is not None:
            print("PREMISED OUTPUTS ", outputs_pre[0])
            print("HYPED OUTPUTS ", outputs_hyp[0])'''
        
        

        pre_out=outputs_pre.last_hidden_state.mean(dim=1)  #self.encode_sentence(outputs_pre, hyp_attention_mask)
        
        hyp_out =outputs_hyp.last_hidden_state.mean(dim=1)

        diffs = pre_out - hyp_out
        prods = pre_out * hyp_out
        
        mlp_input = torch.cat([pre_out, hyp_out,diffs,prods],dim=1)
        
        
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        
        mlp_unpacked = list(self.mlp)
        
        if final_mlp_inp_z is not None:
            #print("MULT BY ZS Input (3072->1024)", final_mlp_inp_z.shape)
            mlp_input *= final_mlp_inp_z #apply mask of 3072, to mlp_input(shaoe is.  16x3072 )
            
        pre_final_layer_reps=mlp_input
        mlp_input = mlp_unpacked[0](mlp_input)
        
        mlp_input = mlp_unpacked[1](mlp_input)
        mlp_input = mlp_unpacked[2](mlp_input)
        final_layer_reps=mlp_input
        if final_mlp_hidden_z is not None:
            #print("MULT BY ZS Hidden (1024->3): ", final_mlp_hidden_z.shape)
            
            #print(f"MLP input bfore {mlp_input}")
            #print(f"Applying final mask to {mlp_input.shape} of {final_mlp_hidden_z.shape}")
            mlp_input *= final_mlp_hidden_z
            #print(f"MLP input after {mlp_input}")
            
        logits = mlp_unpacked[3](mlp_input)
        
        
        #---
        if hyp_input_ids is not None:
            batch_size = hyp_out.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.tokenizer.pad_token  is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif pre_input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (pre_input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(pre_input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
       
        pooled_logits = logits
     

        loss = None
        
        
        #not moving logits nor labels to cpu
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                pooled_logits.view(-1, self.num_labels),
                labels.view(-1)
            )

            

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(pre_final_layer_reps, final_layer_reps,pooled_logits),
            
            hidden_states=(outputs_pre.hidden_states,outputs_hyp.hidden_states) ,
            attentions=(outputs_pre.attentions, outputs_hyp.attentions)
        )
    def get_final_reprs(
            self,
            pre_input_ids=None,
            pre_attention_mask=None,
            hyp_input_ids=None,
            hyp_attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            past_key_values = None,
            output_attentions=None,
            output_hidden_states=None,
           
    ):
        outputs_pre = self.model(
            pre_input_ids,
            attention_mask=pre_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
         
        ) #! [32, 68, 768]
        
        #hyp_input_ids = hyp_input_ids.unsqueeze(0)  # Shape becomes [1, 46]
        #hyp_attention_mask = hyp_attention_mask.unsqueeze(0)
        outputs_hyp = self.model(
            hyp_input_ids,
            attention_mask=hyp_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
  
        )
    

        pre_out=outputs_pre.last_hidden_state.mean(dim=1)  #self.encode_sentence(outputs_pre, hyp_attention_mask)
        
        hyp_out =outputs_hyp.last_hidden_state.mean(dim=1)
        
        diffs = pre_out - hyp_out
        prods = pre_out * hyp_out
        
        mlp_input = torch.cat([pre_out, hyp_out,diffs,prods],dim=1)
        
        
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input) 
        
        return rep


