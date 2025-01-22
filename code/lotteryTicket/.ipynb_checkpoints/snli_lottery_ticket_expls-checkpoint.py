
"""
Train a bowman et al-style SNLI model
"""

import csv
import tqdm
import os
import torch
import torch.optim as optim
import torch.nn as nn
import spacy
import sys
sys.path.append('code/')
import en_core_web_sm
nlp = en_core_web_sm.load()
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from analyze import initiate_exp_run 
import settings
import models
import util
from data import analysis
import importlib.util
import train_utils
import prune_utils

def verify_pruning(model, prev_total_pruned_amt): # does this:
    num_zeros_in_final_weights=torch.where(model.mlp[0].weight.t()==0,1,0).sum()
    new_zeros=num_zeros_in_final_weights-prev_total_pruned_amt
    assert np.round((new_zeros/(1024*2048)),1) == 0.5


def main(args):
    os.makedirs(args.expls_mask_root_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
        
    train,_,_,dataloaders=train_utils.create_dataloaders(max_data=max_data)
    model = train_utils.load_model(max_data=max_data, model_type=args.model_type, train=train, ckpt=args.ckpt)
    
    
    
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(args.ckpt) #trained bowman/bert 
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    device = 'cuda' if settings.CUDA else 'cpu'
    
    return prune_utils.run_expls(args, model,dataset, dataloaders,device)
    
#running the expls using the already finetuned and precreated masks from before

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--expls_mask_root_dir", default="expls/bert/lottery_ticket/Run1")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/BERT/Run1")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=0.95, type=float)
    
    parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--prune_iters", default=5, type=int)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--ckpt", default=settings.MODEL)
    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pruned_percents, final_accs = main(args)
    print(f"pruned_percents: {pruned_percents}\nfinal_accs: {final_accs}")
