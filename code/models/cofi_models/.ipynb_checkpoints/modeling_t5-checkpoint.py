import logging
import math
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling,
                                           SequenceClassifierOutput)
from transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import (
    BertAttention, BertEmbeddings, BertEncoder, BertForQuestionAnswering,
    BertForSequenceClassification, BertLayer, BertModel, BertOutput,
    BertSelfAttention, BertSelfOutput, QuestionAnsweringModelOutput)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import train_utils

from cofi.utils.cofi_utils import *
logger = logging.getLogger(__name__)

class CoFiLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output

class CoFiT5ForSequenceClassification(T5ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.t5 = CoFiT5Model(config)
        
        self.model_name='t5'
        for param in self.t5.parameters():
            param.requires_grad = True


        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
        
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
            config = AutoConfig.from_pretrained("google-t5/t5-small")
            config.do_layer_distill = False
        else:
            config = kwargs["config"]

        # Create model (random MLP + CoFi)
        model = cls(config)
        trained = False
        # -----------------------
        # Load .pth checkpoint
        # -----------------------
    
        if pretrained_model_name_or_path and ".pth" in str(pretrained_model_name_or_path) and os.path.exists(pretrained_model_name_or_path):
            print(f"Loading pretrained t5 entailment ({pretrained_model_name_or_path})")
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
        elif os.path.exists(kwargs['ckpt']):
            weights = torch.load(kwargs['ckpt'])['state_dict']
            model.load_state_dict(weights, strict=False)
            assert trained==False
            return model, trained

        # -----------------------
        # Load HF pretrained encoder only
        # -----------------------
        print("Loading HF google-t5/t5-small pretrained encoder")
        hf_encoder = T5Model.from_pretrained("google-t5/t5-small")

        # Filter HF weights to match model (skip classifier / MLP)
        hf_state = hf_encoder.state_dict()
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in model_state.items() if k in model_state and v.shape == model_state[k].shape}

        model.load_state_dict(filtered_state, strict=False)
        torch.save({'state_dict': model.state_dict()}, kwargs['ckpt'])
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
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            head_z=None,
            head_layer_z=None,
            intermediate_z=None,
            mlp_z=None,
            hidden_z=None,
            final_mlp_hidden_z=None,
            final_mlp_inp_z=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs_pre = self.t5(
            pre_input_ids,
            attention_mask=pre_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z
        ) #! [32, 68, 768]
        sequence_output = outputs_pre[0]
        eos_mask = pre_input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        torch_compilable_check(
            torch.unique_consecutive(eos_mask.sum(1)).numel() == 1,
            "All examples must have the same number of <eos> tokens.",
        )
        batch_size, _, hidden_size = sequence_output.shape
        pre_sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        pre_out=pre_sentence_representation
        
      
        outputs_hyp = self.t5(
            hyp_input_ids,
            attention_mask=hyp_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z
        )
        sequence_output = outputs_hyp[0]
        eos_mask = hyp_input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        torch_compilable_check(
            torch.unique_consecutive(eos_mask.sum(1)).numel() == 1,
            "All examples must have the same number of <eos> tokens.",
        )
        batch_size, _, hidden_size = sequence_output.shape
        hyp_sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        hyp_out = hyp_sentence_representation
   
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   
        
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
        
  
        
        loss = None
        
       
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            combined_hidden_states = (pre_outputs.hidden_states, hyp_outputs.hidden_states)
            combined_attentions = (pre_outputs.attentions, hyp_outputs.attentions)
            output = (logits,) +  (combined_hidden_states, combined_attentions)
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=(pre_final_layer_reps, final_layer_reps,logits),
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=(outputs_pre.hidden_states, outputs_hyp.hidden_states),
            encoder_attentions=(outputs_pre.attentions, outputs_hyp.attentions),
        )
    
    def mean_pool(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def to(self, device):
        self.bert = self.bert.to(device)
        return super().to(device)
        
    def get_final_reprs(
            self,
            pre_input_ids=None,
            pre_attention_mask=None,
            hyp_input_ids=None,
            hyp_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
           
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   
        outputs_pre = self.t5(
            pre_input_ids,
            attention_mask=pre_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z
        ) #! [32, 68, 768]
        sequence_output = outputs_pre[0]
        eos_mask = pre_input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        torch_compilable_check(
            torch.unique_consecutive(eos_mask.sum(1)).numel() == 1,
            "All examples must have the same number of <eos> tokens.",
        )
        batch_size, _, hidden_size = sequence_output.shape
        pre_sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        pre_out=pre_sentence_representation
        
      
        outputs_hyp = self.t5(
            hyp_input_ids,
            attention_mask=hyp_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z
        )
        sequence_output = outputs_hyp[0]
        eos_mask = hyp_input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        torch_compilable_check(
            torch.unique_consecutive(eos_mask.sum(1)).numel() == 1,
            "All examples must have the same number of <eos> tokens.",
        )
        batch_size, _, hidden_size = sequence_output.shape
        hyp_sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        hyp_out = hyp_sentence_representation
        
   
        diffs = pre_out - hyp_out
        prods = pre_out * hyp_out
        
        mlp_input = torch.cat([pre_out, hyp_out,diffs,prods],dim=1)
        
        
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
      
        rep = self.mlp[:-1](mlp_input) 
        return rep



class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        head_layer_z=None,
        head_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None
        **kwargs,
    ) -> tuple[torch.FloatTensor] | Seq2SeqModelOutput:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                intermediate_z=intermediate_z,
                head_z=head_z,
                mlp_z=mlp_z,
                head_layer_z=head_layer_z,
                hidden_z=hidden_z
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]


        return Seq2SeqModelOutput(
            last_hidden_state=None,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class T5Stack(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config)
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        encoder_extended_attention_mask = None
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_extended_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for layer_module in self.block:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                head_z=head_z[i] if head_z is not None else None,
                mlp_z=mlp_z[i] if mlp_z is not None else None,
                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                hidden_z=hidden_z
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class T5Block(GradientCheckpointingLayer):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: int | None = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx)
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        intermediate_z= None,
        head_z= None,
        mlp_z= None,
        head_layer_z=None,
        hidden_z=None,
        **kwargs,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, mlp_z, hidden_z, intermediate_z)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return (
            outputs + attention_outputs
        )  # hidden-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, past_seen_tokens=0):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        output_attentions=False,
        head_z=head_z,
        head_layer_z=head_layer_z,
        hidden_z=hidden_z,
        
        **kwargs,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.key_value_proj_dim)
        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        # We clone here for StaticCache, as we get the value before updating it, but use it after and it's the same ref
        past_seen_tokens = past_seen_tokens.clone() if isinstance(past_seen_tokens, torch.Tensor) else past_seen_tokens

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_values = past_key_values.cross_attention_cache
            else:
                curr_past_key_values = past_key_values.self_attention_cache
        else:
            curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            kv_shape = (*current_states.shape[:-1], -1, self.key_value_proj_dim)
            key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
            value_states = self.v(current_states).view(kv_shape).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, query_states.shape[1], input_shape[1], key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    input_shape[1], key_length, device=scores.device, past_seen_tokens=past_seen_tokens
                )

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        position_bias_masked = position_bias
        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        if head_z is not None:
            #print("Context layer (atttnput ) shape ", context_layer.shape, "head shape", head_z.shape)
            context_layer *= head_z
        

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o(attn_output)
        
        if head_layer_z is not None:
            #print(f"In attention layer {attn_output.shape} and head layer is {head_layer_z.shape}")
            
            attn_output = attn_output.mul(head_layer_z)
            
            
        if hidden_z is not None:
            #print(f"In attention layer {attn_output.shape} and hidden mask is {hidden_z.shape}")
            
            attn_output = attn_output.mul(hidden_z)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: int | None = None):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z
        )
        
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, head_layer_z=None, hidden_z=None, intermediate_z = None, inference=False):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, intermediate_z)
        if head_layer_z is not None:
            forwarded_states*=head_layer_z
        
        if hidden_z is not None:
            forwarded_states = forwarded_states.mul(hidden_z) 
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states, intermediate_z=None, mlp_z=None, hidden_z=None ):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        if intermediate_z is not None: 
            hidden_states *= intermediate_z
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        if mlp_z is not None:
            hidden_states *= mlp_z
        if hidden_z is not None:
            hidden_states *= hidden_z
        return hidden_states
    
class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states, intermediate_z=None, mlp_z=None, hidden_z=None ):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        if intermediate_z is not None:
            hidden_linear *= intermediate_z
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        if mlp_z is not None:
            hidden_states *= mlp_z
        if hidden_z is not None:
            hidden_states *= hidden_z
        return hidden_states
