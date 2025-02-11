
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
from Pruner import Pruner_
import prune_utils

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
    
    # ==== TRAINING SET UP ====
    if args.model_type == 'bert':
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERT
    else:
        optimizer = optim.Adam(model.parameters())
        
    criterion = nn.CrossEntropyLoss()
    
    if settings.CUDA:
        device='cuda'
        print("On cuda")
    else:
        device='cpu'
        print("On CPU")
    model.to(device)
    
    pruner = Pruner_(model)
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
    #train, prune, apply prune mask to init, train
    for prune_iter in tqdm(range(0, args.prune_iters)):
        
            
        #=====SETTINGS AND TRAIN======
         #otherwise take the inital weights that are pruned off and retrain that 
        if args.model_type=='bert':
            optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERT
        else:
            optimizer = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()
        model.cuda()

        prune_metrics_dir = os.path.join(args.prune_metrics_dir, f"{prune_iter}_Pruning_Iter")
        os.makedirs(prune_metrics_dir,exist_ok=True)

        ft_epochs = int(args.finetune_epochs/2) if prune_iter > 0 else args.finetune_epochs

        model = train_utils.finetune_pruned_model(model,args.model_type, optimizer,criterion, train, val, dataloaders, ft_epochs, prune_metrics_dir, device)

        #record accuracy
        final_acc = train_utils.run_eval(model, dataloaders['val'])
        final_weights_pruned = prune_utils.percent_pruned_weights(model)
        pruned_percents.append(final_weights_pruned)
        final_accs.append(final_acc)
        
        #stop pruning after max_thresh
        if final_weights_pruned >= args.max_thresh: break
            
        model.cuda()
        #====PRUNE=====
        
        model = pruner.prune() #PRUNE AND SAVE PRUNE MASK

        #===== APPLY PRUNING MASK TO INIT WEIGHTS ====== 
        for layer in base_ckpt['state_dict'].keys():
            try:
                base_ckpt['state_dict'][layer] *= model.get_layer(layer).pruning_mask.cpu()
                masks = model.get_layer(layer).pruning_mask.cpu()
                print(torch.where(masks== 0,1,0).sum()/(masks.shape[0]*masks.shape[1]))
            except:
                print("Not able to prune this layer ", layer)
                continue
                
        # Reload random inits with pruned weights (that were prnued after fting) 0'd out
        model.load_state_dict(base_ckpt['state_dict'])                 
        
        
    return pruned_percents, final_accs

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

   
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/bowman")
    parser.add_argument("--root_metrics_dir", default="models/snli")
    parser.add_argument("--model_dir", default="expls/snli/model_dir")
    parser.add_argument("--store_exp_bkdown", default="expls/snli_1.0_dev-6-sentence-5/")
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
    parser.add_argument("--ckpt", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pruned_percents, final_accs = main(args)
    print(f"pruned_percents: {pruned_percents}\nfinal_accs: {final_accs}")
    #wandb_ = wandb_init("CCE_NLI_Pruned_Model_Accs", "Run")
    #for i,acc in enumerate(final_accs):
      #  wandb_.log({"prune_iter": i, "accuracy_test": acc})