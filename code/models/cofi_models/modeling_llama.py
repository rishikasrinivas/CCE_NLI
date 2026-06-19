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
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from llm2vec.models.bidirectional_llama import LlamaBiModel, ModifiedLlamaDecoderLayer, ModifiedLlamaAttention, ModifiedLlamaFlashAttention2
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from cofi.utils.cofi_utils import *

from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)

def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

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



def flash_attn_encoder_forward(q, k, v, padding_mask, dropout_p=0.0, softmax_scale=None):
    """
    q: [B, S, Hq, D]
    k: [B, S, Hkv, D]
    v: [B, S, Hkv, D]
    padding_mask: [B, S], 1 = real token, 0 = pad
    """

    padding_mask = padding_mask.bool()
    B, S = padding_mask.shape

    # Case 1: no padding at all -> fastest path
    if padding_mask.all():
        return flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
        )

    # Case 2: padding exists -> unpad once
    qkv = torch.cat([q, k, v], dim=2)
    qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(qkv, padding_mask)

    Hq = q.shape[2]
    Hkv = k.shape[2]

    q_unpad = qkv_unpad[:, :Hq]
    k_unpad = qkv_unpad[:, Hq:Hq + Hkv]
    v_unpad = qkv_unpad[:, Hq + Hkv:Hq + 2 * Hkv]

    out_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=False,
    )

    return pad_input(out_unpad, indices, B, S)

class CoFiModifiedLlamaAttention(ModifiedLlamaFlashAttention2):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
            
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pruned_heads = set()
        self.attn_impl = 'eager'
    


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # must be [B, S] for flash
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None,
        **kwargs,
    ):
        if self.v_proj is None:
            return (None, None)

        bsz, q_len, _ = hidden_states.shape
        input_dtype = hidden_states.dtype

        # q has num_attention_heads
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

        # k/v have num_key_value_heads for GQA
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query_states = query_states.to(input_dtype)
        key_states = key_states.to(input_dtype)
        value_states = value_states.to(input_dtype)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        dropout_p = 0.0 if not self.training else self.attention_dropout


        # eager expects [B, H, S, D] and 4D additive mask
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout_p,
            scaling=self.scaling,
            **kwargs,
        )

        # eager returns [B, S, H, D]
        # so head_z can be applied directly
        if head_z is not None:
            head_z = head_z.squeeze()
            
            attn_output = attn_output * head_z.view(1, 1, -1, 1)


        # merge heads
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        # output projection
        if self.attn_impl == "flash": 
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                #print("attn_output.dtype bfre oproj", attn_output.dtype)
                attn_output = self.o_proj(attn_output)
                #print("attn_output.dtype after oproj ", attn_output.dtype)
        else:
            attn_output = self.o_proj(attn_output)
            
              
        if head_layer_z is not None:
            attn_output = attn_output * head_layer_z

        if hidden_z is not None:
            attn_output = attn_output * hidden_z
            
        return (attn_output, attn_weights)
    def prune_heads(self, heads):
        len_heads = len(heads)
        
        if len_heads == 0: 
            return
        
    
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.num_attention_heads,      # NOT num_attention_heads
            self.attention_head_size,      # 64
            self.pruned_heads
        )
    
        # Prune linear layers
        if len(index) == 0:
           
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
        else:
            self.k_proj = prune_linear_layer(self.k_proj, index, dim=0)
            self.v_proj = prune_linear_layer(self.v_proj, index, dim=0)
            self.q_proj = prune_linear_layer(self.q_proj, index, dim=0)
            self.o_proj = prune_linear_layer(self.o_proj, index, dim=1)
           
        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
            len(heads)
        self.self.all_head_size = self.self.attention_head_size * \
            self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)


 
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        # Store original dtype
        orig_dtype = q.dtype

        # Ensure cos/sin match the dtype (they might be float32)
        cos = cos.to(orig_dtype)
        sin = sin.to(orig_dtype)

        # Perform rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        # Explicitly cast back to original dtype (safety)
        q_embed = q_embed.to(orig_dtype)
        k_embed = k_embed.to(orig_dtype)

        return q_embed, k_embed

    def _rotate_half(self, x):
        # Preserve dtype through the operation
        orig_dtype = x.dtype
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        result = torch.cat((-x2, x1), dim=-1)
        return result.to(orig_dtype)  # Force back to original dtype
    
class CoFiModifiedLlamaDecoderLayer(ModifiedLlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
        # Replace attention with CoFi-enabled version
        self.self_attn = CoFiModifiedLlamaAttention(config, layer_idx)
        
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

class CoFiLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
class CoFiLlamaBiModel(LlamaBiModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace decoder layers with CoFi-enabled versions
        self.layers = nn.ModuleList(
            [CoFiModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
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
            
            inputs_embeds = self.embed_tokens(input_ids)
            
            
        # apply hidden mask to embeddings
        if hidden_z is not None:
            inputs_embeds *= hidden_z
            #print(f"sfer hidden Input embeds shape = {inputs_embeds.dtype}")
        
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

  
        padding_mask=attention_mask
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
            #print(" decoder_layer nip dtype = ", hidden_states.dtype)
            
        
            if output_hidden_states:  
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get layer-specific masks
            layer_head_z = head_z[idx] if head_z is not None else None
            layer_head_layer_z = head_layer_z[idx] if head_layer_z is not None else None
            layer_intermediate_z = intermediate_z[idx] if intermediate_z is not None else None
            layer_mlp_z = mlp_z[idx] if mlp_z is not None else None
            layer_qk_head_dim_z = qk_head_dim_z[idx] if qk_head_dim_z is not None else None
            layer_vo_head_dim_z = vo_head_dim_z[idx] if vo_head_dim_z is not None else None
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
            #print("hidden state bef dec ", hidden_states.dtype)
            
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
                qk_head_dim_z=layer_qk_head_dim_z,
                vo_head_dim_z=layer_vo_head_dim_z,
                
                
                
                
            )
            
            hidden_states = layer_outputs[0]
            #print("hidden state after dec ", hidden_states.dtype)
            
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
   
        self.tokenizer = AutoTokenizer.from_pretrained('knowledgator/Sheared-LLaMA-encoder-1.3B')
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
            config = AutoConfig.from_pretrained(kwargs['hf_name'])
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
        elif os.path.exists(kwargs['ckpt']):
            weights = torch.load(kwargs['ckpt'], map_location=kwargs['device'])['state_dict']
            model.load_state_dict(weights, strict=False)
            assert trained==False
            
            return model , trained

        # -----------------------
        # Load HF pretrained encoder only
        # -----------------------
        print("Loading HF ", kwargs['hf_name'])
        
        hf_encoder = LlamaBiModel.from_pretrained(kwargs["hf_name"], attn_implementation='flash_attention_2').to(kwargs['device'])

        hf_state = hf_encoder.state_dict()
        model_state = model.state_dict()

       
        #lth_trained = torch.load("/workspace/CCE_NLI/LLAMA/models/pretrained/llama_MAIN_pretrained_inits.pth", map_location=kwargs['device'])['state_dict']

        
        filtered_state = {f'model.{k}':v for k,v in hf_state.items() if f'model.{k}' in model_state.keys()}
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        
        #print(filtered_state.keys())
        print("loaded HF encoder keys:", len(filtered_state))
        print("missing:", len(missing))
        print("unexpected:", len(unexpected))
        from pathlib import Path

        Path(kwargs['ckpt']).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict':model.state_dict()}, kwargs['ckpt'])
        
        return model, trained




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


    def masked_mean_pool(self, hidden, mask):
        hidden = hidden.last_hidden_state
        mask = mask.unsqueeze(-1).float()

        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
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
        past_key_values=None,
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
        vo_head_dim_z=None,
    ):

        outputs_pre = self.model(
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
            vo_head_dim_z=vo_head_dim_z,
        )

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
            vo_head_dim_z=vo_head_dim_z,
        )

        pre_out = self.masked_mean_pool(outputs_pre, pre_attention_mask).float()
        hyp_out = self.masked_mean_pool(outputs_hyp, hyp_attention_mask).float()

        diffs = pre_out - hyp_out
        prods = pre_out * hyp_out

        mlp_input = torch.cat([pre_out, hyp_out, diffs, prods], dim=1).float()

        if final_mlp_inp_z is not None:
            mlp_input = mlp_input * final_mlp_inp_z.to(
                device=mlp_input.device,
                dtype=mlp_input.dtype,
            )

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        mlp_unpacked = list(self.mlp)

        pre_final_layer_reps = mlp_input

        mlp_input = mlp_unpacked[0](mlp_input)
        mlp_input = mlp_unpacked[1](mlp_input)
        mlp_input = mlp_unpacked[2](mlp_input)

        final_layer_reps = mlp_input

        if final_mlp_hidden_z is not None:
            mlp_input = mlp_input * final_mlp_hidden_z.to(
                device=mlp_input.device,
                dtype=mlp_input.dtype,
            )

        logits = mlp_unpacked[3](mlp_input)
        pooled_logits = logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                pooled_logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(pre_final_layer_reps, final_layer_reps, pooled_logits),
            hidden_states=(outputs_pre.hidden_states, outputs_hyp.hidden_states),
            attentions=(outputs_pre.attentions, outputs_hyp.attentions),
        )
    '''def forward(
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
        if mlp_z is not None:
            print("PREMISED OUTPUTS ", outputs_pre[0])
            print("HYPED OUTPUTS ", outputs_hyp[0])
        
        

        pre_out= self.masked_mean_pool(outputs_pre, pre_attention_mask).float() 
        
        hyp_out = self.masked_mean_pool(outputs_hyp, hyp_attention_mask).float()  

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
        )'''
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
    
