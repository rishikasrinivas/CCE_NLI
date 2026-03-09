import os
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from data.snli import SNLI, pad_collate
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import spacy
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
import settings
import models
import util
import train_utils
import data.snli
from cofi.utils.utils import calculate_parameters

from cofi.utils.cofi_utils import load_model
def run_inference(args, pair):
    print("using weights from ", args.ckpt)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load(args.ckpt, map_location = 'cuda' if settings.CUDA else 'cpu')
    print("Building dataset")
    train,_,dataloaders=train_utils.create_dataloaders(max_data=None, model_type=args.model_type, pruning_method=args.pruning_method)
    # ==== BUILD MODEL ====
    model,_ = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='CoFi')
    
    s1,s2=pair
    s1_pad = pad_sequence(s1, padding_value=1) #takes 10,000x longest sent length and flips to longest*10,000 which is then batched
    s1len = torch.tensor(s1len)

    s2_pad = pad_sequence(s2, padding_value=1)
    s2len = torch.tensor(s2len)
    
    s1_indices, s2_indices = s1_pad.cpu().numpy().T, s2_pad.cpu().numpy().T
    
    tokenizer_name = "bert-base-uncased" if model_type == 'bert' else "knowledgator/Llama-encoder-1.0B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        
    s1_sentences = [" ".join([itos.get(i, "") for i in row if i not in (0, 1)]) for row in s1_indices]
    s2_sentences = [" ".join([itos.get(i, "") for i in row if i not in (0, 1)]) for row in s2_indices]
    s1_tokenized = tokenizer(s1_sentences, return_tensors="pt", padding=True, truncation=True)
    s2_tokenized = tokenizer(s2_sentences, return_tensors="pt", padding=True, truncation=True)
    if args.pruning_method=='CoFi':
        batch_data = [{
            "pre_input_ids": s1_tokenized["input_ids"].cpu(),
            "pre_attention_mask": s1_tokenized["attention_mask"].cpu(),
            "hyp_input_ids": s2_tokenized["input_ids"].cpu(),
            "hyp_attention_mask": s2_tokenized["attention_mask"].cpu(),
        }]
    else:
        batch_data=[(s1_tokenized, s2_tokenized)]
    
    return model(**batch_data)


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        default="test.txt",
        help="Data to eval interactively (pairs of sentences); use - for stdin",
    )
    
    parser.add_argument("--root_dir", default="/workspace/CCE_NLI/BOWMAN/models/lottery_ticket/Run0.25/")
    parser.add_argument("--ckpt", default="BOWMAN/models/lottery_ticket/Run0.25/0_Pruning_Iter/model_best.pth")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "bert", "llama"])
    parser.add_argument("--pruning_method", default="lottery_ticket", choices=["lottery_ticket", "wanda", "CoFi"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_data_path", default="data/snli_1.0/")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pair=['A man in apron and latex glove cleaning the desk tables in a theater type auditorium .','The man works hard cleaning the auditorium .']
    print(run_inference(args, pair))

