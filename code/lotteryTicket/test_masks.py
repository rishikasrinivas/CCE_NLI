
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
from analyze import initiate_exp_run 
import settings
import models
import util
from data import analysis
import importlib.util
import train_utils
sys.path.append("Analysis/")
import pipelines as pipelines
import wandb_utils
sys.path.append("Analysis/")
import pipelines as pipelines
import math

#running the expls using the already finetuned and precreated masks from before
def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)

    # ==== LOAD DATA ====
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data)
   
    # ==== BUILD MODEL LOAD DATALOADERS ====
    model = train_utils.load_model(max_data=max_data, train=train, ckpt=args.ckpt)
    print("Loading base_ckpt from ", settings.MODEL)
    base_ckpt=torch.load(settings.MODEL) #randomly initialzied model
    final_weights=model.mlp[:-1][0].weight.detach().cpu().numpy()
    
    if settings.CUDA:
        device = 'cuda'
        model = model.cuda()
    else:
        device = 'cpu'

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    
    # ==== INITIAL ACCURACY ====
    init_acc=train_utils.run_eval(model, dataloaders['test'])
    print(f"Accuracy: {init_acc}")


    #pruning
    for prune_iter in tqdm(range(1,args.prune_iters+1)):
        print(f"==== PRUNING ITERATION {prune_iter}/{args.prune_iters+1} ====")
        
        model.load_state_dict(base_ckpt['state_dict']) # RELOAD RANDOM WEIGHTS
        if prune_iter > 1:
            model.prune_mask = prune_mask
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
        else:
            device = 'cpu'
            
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
    
        prune_amt = settings.PRUNE_AMT #1-math.pow(1-settings.PRUNE_AMT,prune_iter)
        print("Prune amt", settings.PRUNE_AMT)
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
        print("Bfore pruning: Final_wegihts prune% is: ", bfore)
    
        model=model.prune(amount=prune_amt,final_weights=final_weights, reverse=args.reverse) #PRUNE
        prune_mask=model.prune_mask
        
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
        torch.save(model.prune_mask , f"code/lotteryTicket/Iter/{bfore}_mask.pth")
        
    
        if bfore > 99: break
    return

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp_dir", default="models/snli/LH")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/LH")
    parser.add_argument("--model_dir", default="exp/snli/model_dir")
    parser.add_argument("--store_exp_bkdown", default="exp/snli_1.0_dev-6-sentence-5/")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal"])
    parser.add_argument("--save_every", default=1, type=int)
    
    parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--prune_iters", default=5, type=int)
    
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
    main(args)
    