import logging
import math
from typing import Optional, Tuple, Union, Dict, List
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPast,
                                           SequenceClassifierOutputWithPast)
from transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding,
    LlamaForSequenceClassification, LlamaDecoderLayer, LlamaModel, LlamaMLP,LlamaPreTrainedModel)

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig
from llm2vec.models.bidirectional_llama import LlamaBiModel, ModifiedLlamaDecoderLayer , ModifiedLlamaAttention
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import train_utils
from cofi.utils.cofi_utils import *
logger = logging.getLogger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
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
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
   
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

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

            

class CoFiLlamaRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size)

    def forward(self, hidden_states, hidden_z):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if hidden_z is not None:
            masked = hidden_states * hidden_z
            norm_dim = hidden_z.sum(-1, keepdim=True).clamp(min=1.0)
            variance = (masked ** 2).sum(-1, keepdim=True) / norm_dim
            hidden_states = masked * torch.rsqrt(variance + self.variance_epsilon)
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}" 

class CoFiLlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name='llama'
        
        self.config=config
        self.model = CoFiLlamaModel(config)
   
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
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
            print("Loading pretrained llama entailment (.pth)")
            weights = torch.load(pretrained_model_name_or_path)["state_dict"]

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

        # -----------------------
        # Load HF pretrained encoder only
        # -----------------------
        print("Loading HF knowledgator/Llama-encoder-1.0B pretrained encoder")
        hf_encoder = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")

        # Filter HF weights to match model (skip classifier / MLP)
        hf_state = hf_encoder.state_dict()
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in hf_state.items() if k in model_state and v.shape == model_state[k].shape}

        model.load_state_dict(filtered_state, strict=False)
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
            mlp_input *= final_mlp_hidden_z
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
            past_key_values= (outputs_pre.past_key_values,outputs_hyp.past_key_values),
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
    
class CoFiLlamaBiModel(LlamaModel):
    _no_split_modules = ["ModifiedLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
       
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )


class CoFiLlamaModel(CoFiLlamaBiModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        self.rotary_emb = LlamaRotaryEmbedding(config=config) #don't apply mask to positional encodings
        
        self.layers = nn.ModuleList(
            [CoFiLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CoFiLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        use_cache=None,
        cache_position=None,
        head_layer_z=None,
        head_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        **flash_attn_kwargs,
    ):
    
    
      
        
      
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        use_cache = self.config.use_cache
        
        assert output_attentions is not None, "output attention is none"
        assert output_hidden_states is not None, "output_hidden_states is none"
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.cuda())

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

      

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask, #dont want causal mask
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                head_z=head_z[i] if head_z is not None else None,
                mlp_z=mlp_z[i] if mlp_z is not None else None,
                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                hidden_z=hidden_z,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states,hidden_z)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
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


class CoFiLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

 
    def forward(self, x, intermediate_z=None, mlp_z=None, hidden_z=None):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = self.act_fn(gate) * up

        if intermediate_z is not None:
            hidden = hidden * intermediate_z.view(1, 1, -1)

        out = self.down_proj(hidden)

  
        if hidden_z is not None:
            out = out * hidden_z.view(1, 1, -1)


        if mlp_z is not None and mlp_z.sum().eq(0):
            return torch.zeros_like(out)

        return out


    
class CoFiLlamaDecoderLayer(ModifiedLlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
        self.layer_idx= layer_idx
        self.self_attn = CoFiModifiedLlamaAttention(config=config, layer_idx=layer_idx)
        
        
        self.mlp = CoFiLlamaMLP(config)
        self.input_layernorm = CoFiLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CoFiLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None, 
        inference=False
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states,hidden_z)

        self_attention_outputs = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            output_attentions=output_attentions,
            head_z=head_z, #doesnt need hidden_z cuz attn doesnt call a selfmlp
        )
        
        
        attn_out, self_attn_weights = self_attention_outputs
        
        if head_layer_z is not None:
            attn_out = attn_out.mul(head_layer_z)
          
        #print("attn_out sum:", attn_out.sum().item())
        if attn_out.sum().eq(0).item():
     
            hidden_states = residual + attn_out
        else:

            attn_out = residual + attn_out
            residual=attn_out
            hidden_states = self.post_attention_layernorm(attn_out, hidden_z)


            hidden_states = self.mlp(hidden_states, intermediate_z, hidden_z, mlp_z)

            hidden_states = residual + hidden_states
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        hidden_states = outputs
   
        return hidden_states
    


class CoFiModifiedLlamaAttention(ModifiedLlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        
        super().__init__(config, layer_idx)
        
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pruned_heads = set()



    def prune_heads(self, heads):
        len_heads = len(heads)
        if len_heads == 0: 
            return
        

        print(f"Before pruning: num_attention_heads={self.num_attention_heads}, attention_head_size={self.attention_head_size}")
        print(f"Pruning heads: {heads}")
        print(f"q_proj weight shape: {self.q_proj.weight.shape}")
        print(f"k_proj weight shape: {self.k_proj.weight.shape}")
        print(f"v_proj weight shape: {self.v_proj.weight.shape}")
        print(f"o_proj weight shape: {self.o_proj.weight.shape}")

        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )
        print(f"Index: {index[:10]}...{index[-10:]}") 
        print(f"Pruning index: {index}")
      
        # Prune linear layers
        if len(index) == 0:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
        else:
            print("q")
            self.q_proj = prune_linear_layer(self.q_proj, index, dim=0)
            print('lk')
            self.k_proj = prune_linear_layer(self.k_proj, index, dim=1)
            print('v')
            self.v_proj = prune_linear_layer(self.v_proj, index, dim=0)
            print('o')
            self.o_proj = prune_linear_layer(
                self.o_proj, index)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - \
            len(heads)
        self.all_head_size = self.attention_head_size * \
            self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
       
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        #if self.config._attn_implementation != "eager":
            #attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
       
        
        if head_z is not None:
            attn_output = attn_output.transpose(1,2) * head_z.view(1, -1, 1, 1).repeat_interleave(8,dim=1)
            attn_output = attn_output.transpose(1,2)
            
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        attn_output = self.o_proj(attn_output)
     
        
        
        if head_layer_z is not None:
            attn_output = attn_output.mul(head_layer_z)
            
        if attn_output.sum().eq(0).item(): #if the hidden states are all masked out preserve the res connect to lep gradient flow
            attn_output = attn_output + hidden_states
            
        else:
            if hidden_z is not None:
                
                
                attn_output = attn_output.mul(hidden_z)
        
                
       
    
    
        #print(f"Next state will be {attn_output.shape}")
        
        return (attn_output, attn_weights)

