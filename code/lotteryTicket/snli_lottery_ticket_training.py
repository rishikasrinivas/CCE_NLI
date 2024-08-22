
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

def verify_pruning(model, prev_total_pruned_amt): # does this:
    num_zeros_in_final_weights=torch.where(model.mlp[0].weight.t()==0,1,0).sum()
    new_zeros=num_zeros_in_final_weights-prev_total_pruned_amt
    assert np.round((new_zeros/(1024*2048)),1) == 0.5
    

def save_load_ckpt(path, model):
    ckpt=torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    return model, ckpt


#running the expls using the already finetuned and precreated masks from before
def main(args, pruned_percents, final_acc):
    
    settings.PRUNE_METHOD='lottery_ticket'
    settings.PRUNE_AMT=0.2
    os.makedirs(args.exp_dir, exist_ok=True)

    # ==== LOAD DATA ====
    if args.debug:
        max_data = 1000
    else:
        max_data = 10000
   
    # ==== BUILD MODEL ====

    ckpt=torch.load(settings.MODEL)
    model = train_utils.load_model(max_data=max_data, ckpt=ckpt)

    if settings.CUDA:
        device = 'cuda'
        model = model.cuda()
    else:
        device = 'cpu'

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data, ckpt=ckpt)
    
    
    # ==== BUILD VOCAB ====
    vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)


    # == PREPARE MODEL ====
    prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"default")
    os.makedirs(prune_metrics_dir, exist_ok=True)
    
    
    train_utils.finetune_pruned_model(model, optimizer,criterion, train, val, dataloaders, args.finetune_epochs, prune_metrics_dir, device)
    model, base_ckpt = save_load_ckpt(path=f"{prune_metrics_dir}/model_best.pth", model=model)

    
    final_weights=model.mlp[:-1][0].weight.detach().cpu().numpy()
    
    train_utils.finetune_pruned_model(model, optimizer,criterion, train, val, dataloaders, args.finetune_epochs, prune_metrics_dir, device) #finish training
    # setting up pruning mask and weights

    
    
    init_acc=train_utils.run_eval(model, dataloaders['test'])
    print(f"Accuracy: {init_acc}")



    
    
    
    #pruning
    
    for prune_iter in tqdm(range(1,args.prune_iters+1)):
        print(f"==== PRUNING ITERATION {prune_iter} ====")
        #location to store metrics
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(args.prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)


        model.load_state_dict(base_ckpt['state_dict']) # trained for k iters
        
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
        else:
            device = 'cpu'
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        if prune_iter > 1:
          
            model.prune_mask = pruning_mask
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
            model.prune_mask.cuda()
        else:
            device = 'cpu'
            
    
        print("Prune amt", settings.PRUNE_AMT)
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
    
        print("Bfore pruning: Final_wegihts prune% is: ", bfore)
        model=model.prune(amount=settings.PRUNE_AMT,final_weights=final_weights, reverse=args.reverse)
        pruning_mask = model.prune_mask

        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
        print("After pruning: Final_wegihts prune% is: ", bfore)
        
        model, final_weights, _= train_utils.finetune_pruned_model(model, optimizer,criterion, train, val, dataloaders, args.finetune_epochs, prune_metrics_dir, device)

        final_weights_pruned = np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(1024*2048), 2)
        print("After fting: Final_wegihts prune% is: ",final_weights_pruned )
        
        
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
            model.prune_mask.cuda()

        final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(1024*2048), 2)
        print("After fting: Final_wegihts prune% is: ",final_weights_pruned )
        
        
         if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
        

        assert(torch.equal(model.mlp[0].weight.detach().cpu(), torch.tensor(final_weights)))
        
        acc=train_utils.run_eval(model, dataloaders['test'])
        pruned_percents.append(final_weights_pruned)
        final_accs.append(acc)
        
        print(f"Accuracy: {acc}")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inital_accs_=0
    pruned_percents_=[]
    final_accs_=[]
    if args.log:
        wandb_=wandb_utils.wandb_init(proj_name="CCE_NLI_LT_Testing", exp_name='20prune_iters_lowest_pruned')
    for i in range(args.test_iters):
        pruned_percents=[]
        final_accs=[]
        inital_accs, pruned_percents, final_accs= main(args, pruned_percents, final_accs)
        inital_accs_+=inital_accs
        pruned_percents_.append(pruned_percents)
        final_accs_.append(final_accs) #[[2.22, 1.67], [2.91, 3.38]]
    print(f"RESULTS: {inital_accs_, pruned_percents_, final_accs_}")
    wandb_.log({"prune_iter": 0, "accuracy_test": inital_accs_/args.test_iters})
   
    
    for i in range(args.prune_iters ):
        percents=0
        accs = 0 
        for j in range(args.test_iters):
            percents +=  pruned_percents_[j][i] 
            accs += final_accs_[j][i]
        print("Average percent pruned after finetuning for iteration ", i, ": ", percents/args.test_iters)
        print("Average accs after finetuning for iteration ", i, ": ", accs/args.test_iters)
        
        if args.log:
            wandb_.log({"prune_iter": np.round(percents/args.test_iters,3), "accuracy_test": np.round(accs/args.test_iters,3)})
   
    if args.log:
        wandb_.finish()
