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
from llm2vec.models.bidirectional_llama import LlamaBiModel, ModifiedLlamaDecoderLayer, ModifiedLlamaAttention
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from cofi.utils.cofi_utils import *

logger = logging.getLogger(__name__)

class CoFiLlamaRMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states, hidden_z=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        if hidden_z is not None:
            # Mask out pruned dimensions
            masked = hidden_states * hidden_z
            # Compute variance over active dimensions only
            norm_dim = hidden_z.sum(-1, keepdim=True).clamp(min=1.0)
            variance = (masked ** 2).sum(-1, keepdim=True) / norm_dim
            hidden_states = masked * torch.rsqrt(variance + self.variance_epsilon)
            # Mask weight to prevent pruned dims from contributing
            
            
            return (self.weight * hidden_z) * hidden_states.to(input_dtype)
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


class CoFiModifiedLlamaAttention(ModifiedLlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.pruned_heads = set()

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
        hidden_z=None,  # Prune hidden dimensions
        **kwargs,
    ):
        '''input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        print("HIDDEN STATES SHAPE IN ATTN ", hidden_states.shape)
        print("HIDDEN STATES SHAPE IN ATTN ", hhidden_shape)'''
        # Check if entire attention layer is pruned
        if head_layer_z is not None and head_layer_z == 0:
            return (None, None) if output_attentions else (None, None)
        
        bsz, q_len, _ = hidden_states.size()
        
        # Apply hidden_z to input before projections
        if hidden_z is not None:
            hidden_states = hidden_states * hidden_z
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for GQA
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply head_z to prune KV heads
        if head_z is not None:
            key_states = key_states * head_z.view(1, -1, 1, 1)
            value_states = value_states * head_z.view(1, -1, 1, 1)
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update cache if needed
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat KV heads for grouped query attention
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        # Apply head_layer_z to gate entire attention output
        if head_layer_z is not None:
            attn_output = attn_output * head_layer_z
        
        return (attn_output, attn_weights) if output_attentions else (attn_output, None )
    
    def prune_heads(self, heads):
        len_heads = len(heads)
        
        if len_heads == 0: 
            return
        
    
        heads, index = find_pruneable_heads_and_indices(
            heads,
            4,      # NOT num_attention_heads
            self.attention_head_size,      # 64
            self.pruned_heads
        )

        group_size = self.num_heads // self.num_key_value_heads  # e.g. 32 // 4 = 8

        qo_index_to_prune = []
       
        for head in heads:  # these are KV heads
            q_head_start = head * group_size #2*8 to 3*8 so 16->24 
            start = q_head_start * self.attention_head_size
            q_head_end = (head+1)*group_size
            end = q_head_end * self.attention_head_size
            qo_index_to_prune.extend(range(start, end))
        qo_index_to_keep = [i for i in range(2048) if i not in qo_index_to_prune]
        qo_index_to_keep=torch.tensor(qo_index_to_keep)
        # Prune linear layers
        if len(index) == 0:
           
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
        else:
            self.k_proj = prune_linear_layer(self.k_proj, index, dim=0)
            self.v_proj = prune_linear_layer(self.v_proj, index, dim=0)
            self.q_proj = prune_linear_layer(self.q_proj, qo_index_to_keep, dim=0)
            self.o_proj = prune_linear_layer(self.o_proj, qo_index_to_keep, dim=1)
           
        
        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - \
            len(heads)
        self.all_head_size = self.attention_head_size * \
            self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)


 
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        # Helper method to apply rotary embeddings
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _repeat_kv(self, hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,  # Prune MLP intermediate dimensions
        mlp_z=None,  # Prune entire MLP block
        hidden_z=None,
        **kwargs,
    ):
        residual = hidden_states
        
        # Apply hidden_z before norm
        if hidden_z is not None:
            hidden_states = hidden_states * hidden_z
        
        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states, hidden_z)
        
        # Self-attention
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
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
            **kwargs,
        )
        
        # Residual connection (skip if attention output is None)
        if attn_output is not None:
            hidden_states = residual + attn_output
        else:
            hidden_states = residual
        
        residual = hidden_states
        
        # Apply hidden_z before second norm
        if hidden_z is not None:
            hidden_states = hidden_states * hidden_z
        
        # Post-norm
        hidden_states = self.post_attention_layernorm(hidden_states, hidden_z)
        
        # MLP (skip if entire MLP block is pruned)
        if mlp_z is not None and mlp_z == 0:
            mlp_output = None
        else:
            mlp_output = self.mlp(hidden_states, intermediate_z, mlp_z, hidden_z)
        
        # Residual connection
        if mlp_output is not None:
            hidden_states = residual + mlp_output
        else:
            hidden_states = residual
        
        # Final application of hidden_z
        if hidden_z is not None:
            hidden_states = hidden_states * hidden_z
            
        
        
        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


class CoFiModifiedLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x, intermediate_z=None, mlp_z=None, hidden_z=None):
        # Apply hidden_z to input
        if hidden_z is not None:
            x = x * hidden_z
        
        # Apply intermediate_z to prune neurons (mask input to gate/up projections)
       
        # Forward through MLP
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        
        # Apply intermediate_z again to intermediate activations
        if intermediate_z is not None:
            #assert torch.equal(intermediate_z, torch.ones_like(intermediate_z)), f'intermediate_z s not all 1s'
            hidden = hidden * intermediate_z.view(1, 1, -1)
        
        out = self.down_proj(hidden)
        
        # Apply mlp_z to gate entire block
        if mlp_z is not None:
            out = out * mlp_z
            if mlp_z == 0:
                assert out.sum() == 0, f'out is = {out}'
            
        
        return out


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
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Input embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # Prepare attention mask (bidirectional, so no causal mask needed)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
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
            layer_mlp_z = mlp_z[idx] if mlp_z is not None else None
            
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
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
        
        # Final norm
        hidden_states = self.norm(hidden_states, hidden_z)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
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
   
        self.tokenizer = AutoTokenizer.from_pretrained('knowledgator/Llama-encoder-1.0B')
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
            config = AutoConfig.from_pretrained("knowledgator/Llama-encoder-1.0B")
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
            return model, trained

        # -----------------------
        # Load HF pretrained encoder only
        # -----------------------
        print("Loading HF knowledgator/Llama-encoder-1.0B pretrained encoder")
        hf_encoder = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")

        # Filter HF weights to match model (skip classifier / MLP)
  
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in model_state.items()}

        model.load_state_dict(filtered_state)
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
    ):
  
        #pre_input_ids = pre_input_ids.unsqueeze(0)  # Shape becomes [1, 46]
        #pre_attention_mask = pre_attention_mask.unsqueeze(0)

        
        if hidden_z is not None:  
            hidden_z = torch.ones_like(hidden_z)
            
        if intermediate_z is not None:  
            intermediate_z = torch.ones_like(intermediate_z)
        
        if mlp_z is not None:
            mlp_z[5]=0 # manually pruning 1 random mlp layer
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
            hidden_z=hidden_z
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
            hidden_z=hidden_z
        )
        
   
    
   

        hyp_out = outputs_hyp.last_hidden_state[:,0,:]
        pre_out = outputs_pre.last_hidden_state[:,0,:]
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
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                pooled_logits.view(-1, self.num_labels).float().cpu(),
                labels.view(-1).long().cpu()
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
        
   
    
   

        hyp_out = outputs_hyp.last_hidden_state[:,0,:]
        pre_out = outputs_pre.last_hidden_state[:,0,:]
        diffs = pre_out - hyp_out
        prods = pre_out * hyp_out
        
        mlp_input = torch.cat([pre_out, hyp_out,diffs,prods],dim=1)
        
        
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input) 
        
        return rep
    

