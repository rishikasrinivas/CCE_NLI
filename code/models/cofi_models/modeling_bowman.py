import logging
import math
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss, MSELoss
from cofi.utils.cofi_utils import *
logger = logging.getLogger(__name__)
import train_utils
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling,
                                           SequenceClassifierOutput)
import json
from safetensors.torch import save_file
class CoFiBowmanEntailmentClassifier(torch.nn.Module):
    """
    The RNN-based entailment model of Bowman et al 2017
    """

    def __init__(self, encoder, device):
        super().__init__()
        self.model_name = 'bowman'
        self.encoder = encoder.to(device)
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Mimic classifier MLP keep rate of 94%
            nn.Linear(1024, 3),
        )
        self.output_dim = 3
        self.device=device
        self.layer_transformation = nn.Linear(
                self.encoder_dim, self.encoder_dim)
    
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # -------------------------------
        # Always initialize random weights
        # -------------------------------
        print("Initializing Bowman model with random weights")
        model = cls(kwargs['encoder'], device='cuda')
        trained = False
        # Optional: if you have pruning weights, you can still load them
        if pretrained_model_name_or_path and '.pth' in str(pretrained_model_name_or_path) and os.path.exists(pretrained_model_name_or_path):
            print("Loading pruning masks (weights) from .pth")
            weights = torch.load(pretrained_model_name_or_path)['state_dict']

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

    
    def forward(self, s1, s1len, s2, s2len, labels,final_mlp_hidden_z=None):
        
        s1enc = self.encoder(s1, s1len)

        s2enc = self.encoder(s2, s2len)

        
        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1) #1x2048

    
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        
        
        pre_final_layer_2048_outs = mlp_input
        
        mlp_input = self.mlp[0](mlp_input)
        
        
       
            
        mlp_input = self.mlp[1](mlp_input)#relu
        mlp_input = self.mlp[2](mlp_input)#dropout
        final_layer_1024_outs = mlp_input
        if final_mlp_hidden_z is not None:
            mlp_input = mlp_input * final_mlp_hidden_z
            
        logits= self.mlp[3](mlp_input)
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1,3), labels.view(-1))
            
        
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=( pre_final_layer_2048_outs, final_layer_1024_outs, logits),
            hidden_states=(s1enc, s2enc ),
            attentions=None,
        )

    def get_final_reprs(self, s1, s1len, s2, s2len, labels):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)


        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        
                
        rep = self.mlp[:-1](mlp_input) 
        
        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds
    
    def get_encoder(self):
        return self.encoder
    
    def save_pretrained(self, save_directory, safe_serialization=True):
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        # Save model weights
        weights_path = os.path.join(save_directory, 
                                    "model.safetensors" if safe_serialization else "model_best.pth")

        if safe_serialization:
            save_file(self.state_dict(), weights_path)
        else:
            torch.save(self.state_dict(), weights_path)

        print(f"Model saved to {weights_path}")
       
    

class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim, bidirectional=bidirectional
        )
        self.output_dim = self.hidden_dim
        self.model_max_length=hidden_dim


    def forward(self, s, slen):

        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
   
        _, (hidden, cell) = self.rnn(spk)
       
        
        #retunr get all cell states w a param for the cell state # 
        return hidden[-1]
        
        

    def get_states(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        outputs, _ = self.rnn(spk)
        outputs_pad = pad_packed_sequence(outputs)[0]
        return outputs_pad #padded hidden states for each word
    
    def get_last_cell_state(self, s,slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (hidden, cell) = self.rnn(spk)
        return cell[-1]
    def save_pretrained(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "tokenizer_model.bin"))

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "bidirectional": self.bidirectional
        }
        with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config, f)
