
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
from data.snli import SNLI, pad_collate
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import settings
import models
import util
from data import analysis
import importlib.util
import train_utils
sys.path.append("Analysis/")
import pipelines as pipelines
from prune import Pruner

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


def main(args):
    if args.debug:
        max_data = 1000
    else:
        max_data = None
        
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data)
    model = train_utils.load_model(max_data=max_data, model_type=args.model_type, train=train, ckpt=args.ckpt)
    
    
    
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(args.ckpt)
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    if args.model_type == 'bert':
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERT
    else:
        optimizer = optim.Adam(model.parameters())
        
    criterion = nn.CrossEntropyLoss()
    
    if settings.CUDA:
        device='cuda'
    else:
        device='cpu'
    model.to(device)
    
    pruner = Pruner(model)
    return run_prune(
        model,
        pruner,
        args, 
        base_ckpt,
        dataset,
        optimizer, 
        criterion,
        device, 
        train=train,
        val=val,
        test=test,
        dataloaders=dataloaders
    )
    
#running the expls using the already finetuned and precreated masks from before
def run_prune(model, pruner, args, base_ckpt, dataset, optimizer, criterion, device, train, val, test, dataloaders):
    pruned_percents, final_accs, final_weights =[], [], model.mlp[0].weight.detach().cpu().numpy()
    
    for prune_iter in tqdm(range(0, args.prune_iters)):
        print(f"==== PRUNING ITERATION {prune_iter}/{args.prune_iters+1} ====")
        
        #Apply pruning mask to init weights
        for layer in base_ckpt['state_dict'].keys():
            try:
                base_ckpt['state_dict'][layer] *= model.get_layer(layer).pruning_mask.cpu()
            except:
                continue
                
        model.load_state_dict(base_ckpt['state_dict']) # RELOAD INIT WEIGHTS (W/ APPLIED PRUNING MASK)                 
        
        if args.model_type=='bert':
            optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERT
        else:
            optimizer = optim.Adam(model.parameters())
            
        criterion = nn.CrossEntropyLoss()
        model.cuda()
        
        if prune_iter > 0:
            ft_epochs = int(args.finetune_epochs/2)
            model = pruner.prune() #PRUNE AND SAVE PRUNE MASK
        else:
            ft_epochs = args.finetune_epochs
        
            
        #create dir to store metrics for this pruning iteration
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,"Run1", f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)
            
        #finetune
        model, final_weights, _= train_utils.finetune_pruned_model(model,args.model_type, optimizer,criterion, train, val, dataloaders, ft_epochs, prune_metrics_dir, device) #FINETUNE
        
        #stop pruning after max_thresh
        if final_weights_pruned >= args.max_thresh: break
    return pruned_percents, final_accs

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

   
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/LH/bowman")
    parser.add_argument("--model_dir", default="exp/snli/model_dir")
    parser.add_argument("--store_exp_bkdown", default="exp/snli_1.0_dev-6-sentence-5/")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert"])
    parser.add_argument("--save_every", default=1, type=int)
    
    #parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--prune_iters", default=5000, type=int)
    
    
    parser.add_argument("--max_thresh", default=0.95, type=float)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--test_iters", default=1, type=int)
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--baseline", action='store_true')
    parser.add_argument("--ckpt", default=settings.MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pruned_percents, final_accs = main(args)
    print(f"pruned_percents: {pruned_percents}\nfinal_accs: {final_accs}")
    #wandb_ = wandb_init("CCE_NLI_Pruned_Model_Accs", "Run")
    #for i,acc in enumerate(final_accs):
      #  wandb_.log({"prune_iter": i, "accuracy_test": acc})
