
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
#import wandb_utils
sys.path.append("Analysis/")
import pipelines as pipelines

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


def main(args):
    if args.debug:
        max_data = 1000
    else:
        max_data = None
        
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data)
    print(len(train))
    model = train_utils.load_model(max_data=max_data, model_type=args.model_type, train=train, ckpt=None)#args.ckpt)
    base_ckpt=torch.load(f"models/snli/{args.model_type}_random_inits.pth") 
    
    # ==== BUILD VOCAB ====
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
    
    return run_prune(
        model,
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
def run_prune(model, args, base_ckpt, dataset, optimizer, criterion, device, train, val, test, dataloaders):
    pruned_percents=[]
    final_accs=[]
    final_weights = model.mlp[0].weight.detach().cpu().numpy()
    
    for prune_iter in tqdm(range(args.prune_iters)):
        print(f"==== PRUNING ITERATION {prune_iter}/{args.prune_iters+1} ====")
        
        model.load_state_dict(base_ckpt['state_dict']) # RELOAD RANDOM WEIGHTS
        
        if args.model_type=='bert':
             optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERT
        else:
            optimizer = optim.Adam(model.parameters())
            
        criterion = nn.CrossEntropyLoss()
        model.cuda()
        
       
        bfore=0
        print("Bfore pruning: % pruned is: ", bfore)
        if prune_iter > 0:
            ft_epochs = int(args.finetune_epochs/2)
            model=model.prune() #PRUNE# SAVE PRUNE MASK
        else:
            ft_epochs = args.finetune_epochs
        
            
        #create dir to store metrics for this pruning iteration
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,"Run1", f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)
            
        model, final_weights, _= train_utils.finetune_pruned_model(model,args.model_type, optimizer,criterion, train, val, dataloaders, ft_epochs, prune_metrics_dir, device) #FINETUNE
        
        #debug: verifying pruning
        bfore=0
        for layer in model.layers:
            l = model.get_layer(layer.name)
            if layer.name not in settings.PRUNE or settings.PRUNE[layer.name]==0.0:
                continue
            bfore+=torch.where(l.weights.detach() == 0,1,0).sum().item()
            pm = l.pruning_mask.detach()
            print("Pruning Mask for layer ", l,  torch.where(pm == 0,1,0).sum().item() / (pm.shape[0]*pm.shape[1]))
            
        #debug:   
        layer = model.get_layer("mlp.0.weight")
        final_weights_pruned = (torch.where(layer.weights==0,1,0).sum().item()/(layer.weights.shape[0] * layer.weights.shape[1]))
        final_weights_pruned = np.round(final_weights_pruned,4)
        print("After pruning: Final_wegihts prune% is: ", final_weights_pruned)
        
        #calcing percent pruned
        acc=train_utils.run_eval(model, dataloaders['val'])
        print(f"Accuracy: {acc}")
        pruned_percents.append(final_weights_pruned)
        final_accs.append(acc)
        
        
        assert(torch.equal(model.mlp[0].weight.detach().cpu(), torch.tensor(final_weights)))
        
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
