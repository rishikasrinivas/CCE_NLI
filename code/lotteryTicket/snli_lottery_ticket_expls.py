
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

def verify_pruning(model, prev_total_pruned_amt): # does this:
    num_zeros_in_final_weights=torch.where(model.mlp[0].weight.t()==0,1,0).sum()
    new_zeros=num_zeros_in_final_weights-prev_total_pruned_amt
    assert np.round((new_zeros/(1024*2048)),1) == 0.5

def make_folders(prune_iter):
    #masks and explanation storing paths after finetuning
    exp_after_finetuning_flder = f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/"
    if not os.path.exists(exp_after_finetuning_flder):
        os.mkdir(exp_after_finetuning_flder) 

    masks_after_finetuning_flder = f"code/LHMasks/Masks{prune_iter}_Pruning_Iter/"
    if not os.path.exists(masks_after_finetuning_flder):
        os.mkdir(masks_after_finetuning_flder)
    return exp_after_finetuning_flder, masks_after_finetuning_flder


#running the expls using the already finetuned and precreated masks from before
def main(args, pruned_percents=[], final_acc=[]):
    
    settings.PRUNE_METHOD='lottery_ticket'
    settings.PRUNE_AMT=0.99078
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
    
    
    # ==== BUILD VOCAB ====
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    
    # ==== INITIAL ACCURACY ====
    init_acc=train_utils.run_eval(model, dataloaders['test'])
    print(f"Accuracy: {init_acc}")


    #pruning
    
    for prune_iter in tqdm(range(1,args.prune_iters+1)):
        print(f"==== PRUNING ITERATION {prune_iter}/{args.prune_iters+1} ====")
        
        #location to store metrics
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(args.prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)


        model.load_state_dict(base_ckpt['state_dict']) # RELOAD RANDOM WEIGHTS
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
        else:
            device = 'cpu'
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        if prune_iter > 1:
            model.prune_mask = pruning_mask.cuda() #RELOAD PRUNING MASK
        model.cuda()
    
    
        print("Prune amt", settings.PRUNE_AMT)
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
    
    
        model=model.prune(amount=settings.PRUNE_AMT,final_weights=final_weights, reverse=args.reverse) #PRUNE
        pruning_mask = model.prune_mask # SAVE PRUNE MASK
        
        print("Bfore pruning: Final_wegihts prune% is: ", bfore)
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
        print("After pruning: Final_wegihts prune% is: ", bfore)
        
        model, final_weights, _= train_utils.finetune_pruned_model(model, optimizer,criterion, train, val, dataloaders, args.finetune_epochs, prune_metrics_dir, device) #FINETUNE
        final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(1024*2048), 3)
        print("After fting: Final_wegihts prune% is: ",final_weights_pruned )
        
        acc=train_utils.run_eval(model, dataloaders['test'])
        print(f"Accuracy: {acc}")
        exp_after_finetuning_flder, masks_after_finetuning_flder = make_folders(final_weights_pruned)
        
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
        
        clusters = 1 if final_weights_pruned == 100 else 4
    
        if final_weights_pruned > 99:
            print(f"======Running Explanations for {final_weights_pruned}% pruned=======")
            #run after pruning before finetuning
            _,final_layer_weights =initiate_exp_run(
                save_exp_dir = exp_after_finetuning_flder, 
                save_masks_dir= masks_after_finetuning_flder, 
                masks_saved=False, 
                model_=model,
                dataset=dataset,
                clusters=clusters,
            )
    
        assert(torch.equal(model.mlp[0].weight.detach().cpu(), torch.tensor(final_weights)))
        
        acc=train_utils.run_eval(model, dataloaders['test'])
        pruned_percents.append(final_weights_pruned)
        final_accs.append(acc)
        
        print(f"Accuracy: {acc}")
        if final_weights_pruned > 99: break
    return init_acc, pruned_percents, final_accs

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
    parser.add_argument("--ckpt", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_acc, pruned_percents, final_accs = main(args)
    print(f"init_acc: {init_acc}\npruned_percents: {pruned_percents}\nfinal_accs: {final_accs}")
