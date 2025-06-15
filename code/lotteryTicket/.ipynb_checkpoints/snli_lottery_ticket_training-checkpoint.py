
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
from Pruner import Pruner_
import prune_utils

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


def main(args):
    if args.debug:
        max_data = 1000
    else:
        max_data = None
        
    if args.untrained_model:
        use_pretrained_weights = False
    else:
        use_pretrained_weights = True
        
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data, debug=args.debug)
    model,ckpt = train_utils.load_model(max_data=max_data, model_type=args.model_type, use_pretrained_weights = use_pretrained_weights, train=train, ckpt=args.ckpt)
    
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(ckpt)
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    # ==== TRAINING SET UP ====
    if args.model_type in ['bert', 'llama']:
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
def get_mask(weights):
    return torch.where(weights==0,0,1) 


#running the expls using the already finetuned and precreated masks from before
def run_prune(model, pruner, args, base_ckpt, dataset, optimizer, criterion, device, train, val, test, dataloaders, start=0):
    print("Entered run_prune")
    pruned_percents, final_accs, final_weights =[], [], model.mlp[0].weight.detach().cpu().numpy()
    prune_metrics_dir_base = os.path.join(args.model_type.upper(), "models", "lottery_ticket", args.filename)
    os.makedirs(prune_metrics_dir_base, exist_ok=True)
    #train, prune, apply prune mask to init, train
    
    if start > 0:
        prune_metrics_dir = os.path.join(prune_metrics_dir_base, f"{start-1}_Pruning_Iter")
        if os.path.exists(prune_metrics_dir):
            print(f"Alr lt'd {prune_metrics_dir}")
            state_dict =  torch.load(os.path.join(prune_metrics_dir, 'model_best.pth'), map_location=torch.device('cpu'))['state_dict']
            for layer in state_dict.keys():
                mask = get_mask(state_dict[layer]).cpu()
                base_ckpt['state_dict'][layer] *= mask
            model.load_state_dict(base_ckpt['state_dict']) 
    for prune_iter in tqdm(range(start, args.prune_iters)):
        
            
        #=====SETTINGS AND TRAIN======
         #otherwise take the inital weights that are pruned off and retrain that 
        if args.model_type in ['bert', 'llama']:
            optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # AdamW optimizer is recommended for BERT
        else:
            optimizer = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()
        model.cuda()

        # if some extent of pruning alr happenin, get and save the mask and dont bother finetuning
        prune_metrics_dir = os.path.join(prune_metrics_dir_base, f"{prune_iter}_Pruning_Iter")
        os.makedirs(prune_metrics_dir,exist_ok=True)
        torch.save(model.state_dict(), os.path.join(prune_metrics_dir, "initreloaded.pth"))


        model = train_utils.finetune_pruned_model(model,args.model_type, optimizer,criterion, train, val, dataloaders, args.finetune_epochs, prune_metrics_dir, device)

        #record accuracy
        final_acc = train_utils.run_eval(model, dataloaders['val'])
        final_weights_pruned = prune_utils.percent_pruned_weights(model)
        print(f"% Pruned: {final_weights_pruned}")
        pruned_percents.append(final_weights_pruned)
        final_accs.append(final_acc)

        #stop pruning after max_thresh
        if final_weights_pruned >= args.max_thresh: break

        #model.cuda()
        #====PRUNE=====

        model = pruner.prune() #PRUNE AND SAVE PRUNE MASK

        #===== APPLY PRUNING MASK TO INIT WEIGHTS ====== 
        not_pruneable_layers = []
        for layer in base_ckpt['state_dict'].keys():
            try:
                base_ckpt['state_dict'][layer] *= model.get_layer(layer).pruning_mask.cpu()
                masks = model.get_layer(layer).pruning_mask.cpu()
            except:
                print(f"entered if for {layer} which shouldnt be pruneable")
                not_pruneable_layers.append(layer)
                continue
        assert all(any(kw in x for kw in ['bias', 'bn', 'embeddings', 'LayerNorm']) for x in not_pruneable_layers)
                
        # Reload random inits with pruned weights (that were prnued after fting) 0'd out
        model.load_state_dict(base_ckpt['state_dict'])  
        model.cpu()
        
        
    return pruned_percents, final_accs

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

   
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/bowman")
    #parser.add_argument("--root_metrics_dir", default="models/snli")
    #parser.add_argument("--model_dir", default="expls/snli/model_dir")
    parser.add_argument("--store_exp_bkdown", default="expls/snli_1.0_dev-6-sentence-5/")
    parser.add_argument("--filename", type=str)
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert", "llama"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--untrained_model", action="store_true", default=False)  # If `--untrained_model` is used, set to True 
    
    #parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=5, type=int)
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