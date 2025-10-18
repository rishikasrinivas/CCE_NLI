import logging
import os
import sys
import time
import random
from copy import deepcopy
import collections
import datasets
from datasets import load_from_disk
import numpy as np
import torch
import transformers
from torch.utils.data.dataloader import DataLoader
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)
from transformers.models.bert.modeling_bert import (
    BertAttention, BertEmbeddings, BertEncoder, BertForQuestionAnswering,
    BertForSequenceClassification, BertLayer, BertModel, BertOutput,
    BertSelfAttention, BertSelfOutput, QuestionAnsweringModelOutput)
from args import AdditionalArguments, DataTrainingArguments
from utils.cofi_utils import *
from models.l0_module import L0Module
from models.modeling_bert import CoFiBertForSequenceClassification
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from trainer.trainer import CoFiTrainer 
from utils.utils import *
from models.model_args import ModelArguments
import train_utils
import data.snli as snli
from utils.cofi_utils import load_model
from models.modeling_bowman import CoFiBowmanEntailmentClassifier, TextEncoder
from models.modeling_bert import CoFiBertForSequenceClassification
from models.modeling_llama import CoFiLlamaForSequenceClassification
from safetensors.torch import load_file
max_data = 1000 
model_type='bert'
train,val,dataloaders = train_utils.create_dataloaders(model_type=model_type, max_data=None, debug=False)
label_list = list(set(train.labels))
vocab= {'stoi': train.stoi, 'itos': train.itos}

# Labels
is_regression=False
if is_regression:
    num_labels = 1
else:
    num_labels = len(set(train.labels))

t_name='snli'



if model_type=='bowman':
    tokenizer = TextEncoder(len(vocab['stoi']))
    Model = CoFiBowmanEntailmentClassifier(tokenizer, 'cuda')
    model = Model.from_pretrained(
            pretrained_model_name_or_path=f"CoFi_{model_type}/fine_tuned_teacher_snli_{model_type}/model.safetensors",
            from_tf=False,
            teacher=False,
            config=None,
            train_data=train,
            max_data=max_data,
            output_dir = 'test',
            cache_dir=f'CoFi_{model_type}',
            use_auth_token= None,
            encoder=tokenizer,
        )
elif model_type=='bert':
    Model = CoFiBertForSequenceClassification
    config = AutoConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        finetuning_task=t_name,
        cache_dir=f'CoFi_{model_type}',
        use_auth_token=None
    )
    model = Model.from_pretrained(
            pretrained_model_name_or_path=f"CoFi_{model_type}/fine_tuned_teacher_snli_{model_type}/model.safetensors", #if teacher model alr exists, load that (and that will be at this filepath here) but if teacher model doesnt alr exist another default model will be loaded and trained later (Training checks for same path)
            train_data=train,
            max_data=max_data,
            teacher=True,
            config=config
        )
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir=f'CoFi_{model_type}',
        use_fast=True,
        use_auth_token=None,
    )
    
elif model_type=='llama':
    Model = CoFiLlamaForSequenceClassification
    config = AutoConfig.from_pretrained(
        'knowledgator/Llama-encoder-1.0B',
        num_labels=3,
        finetuning_task=t_name,
        cache_dir=f'CoFi_{model_type}',
        use_auth_token=None
    )
    
    model = Model.from_pretrained(
            pretrained_model_name_or_path=f"CoFi_{model_type}/fine_tuned_teacher_snli_{model_type}/model.safetensors", #if teacher model alr exists, load that (and that will be at this filepath here) but if teacher model doesnt alr exist another default model will be loaded and trained later (Training checks for same path)
            train_data=train,
            max_data=max_data,
            teacher=True,
            config=config
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        'knowledgator/Llama-encoder-1.0B',
        cache_dir=f'CoFi_{model_type}',
        use_fast=True,
        use_auth_token=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token  = tokenizer.eos_token 

total_params = 0
zero_params = 0


'''
for name, param in model.named_parameters():
    if param.requires_grad and 'weight' in name:
        # Assume zs[name] is the mask for this weight
        if name in zs:
            mask = zs[name]  # should be same shape as param
            masked_weight = param.data * mask  # simulate pruning

            total_params += masked_weight.numel()
            zero_params += torch.sum(masked_weight == 0).item()
        else:
            # No mask, assume all weights kept
            total_params += param.numel()
            zero_params += torch.sum(param.data == 0).item()

sparsity = zero_params / total_params
print(f"Global sparsity: {sparsity:.4f}")

total_params = 0
total_zeros = 0
for param in model.parameters():
    if param.requires_grad:
        total_params += param.numel()
        total_zeros += torch.sum(param == 0).item()

actual_sparsity = total_zeros / total_params
print(f"Actual sparsity: {actual_sparsity:.4f}")'''

def preprocess_function(examples):
        result = {}
        
        s1_pad, s1len, s2_pad, s2len, labels = snli.pad_collate([
            (torch.tensor(p), plen, torch.tensor(h), hlen, l)
            for p, plen, h, hlen, l in zip(
                examples["premise"],
                examples["premise_len"],
                examples["hypothesis"],
                examples["hypothesis_len"],
                examples["label"]
            )
        ])
        s1_transposed = s1_pad.transpose(1,0)
        s2_transposed = s2_pad.transpose(1,0)
        

        # Build the result with proper batch dimension
        result = { #changed to s1, s1len, s2, s2len from line 200
            's1':s1_transposed,
            's1len': s1len,
            's2': s2_transposed,
            's2len':s2len,
            'labels':labels
        }
            
        return result





def fill_inputs_with_zs(zs, inputs):
    for key in zs:
        inputs[key] = zs[key]
    return inputs

percent=0.0
if percent==0.0:
    pruned_model=model
else:
    zs=torch.load(f"out_{percent}_None/SNLI/CoFi/SNLI_sparsity{percent}/zs.pt")
    pruned_model = load_model(f'out_{percent}_None/SNLI/CoFi/SNLI_sparsity{percent}/', model, zs, encoder=tokenizer)
pruned_model.eval()

pruned_model.cuda()
all_preds = []
all_targets = []

# CORRECTED: Added conditional logic for batch handling
for batch in dataloaders['val']:
    if torch.cuda.is_available():
        #batch = fill_inputs_with_zs(zs, batch)
        batch = {k: v.to('cuda') for k, v in batch.items()}
        targets = batch['labels']


    batch_size = targets.shape[0]

    with torch.no_grad():
        logits = pruned_model(**batch)

    preds = logits[1][2].argmax(1)
    all_preds.append(preds.cpu().numpy())
    all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds, 0)
all_targets = np.concatenate(all_targets, 0)
acc = (all_preds == all_targets).mean()
print(np.round(acc, 3))