
"""
Train a bowman et al-style SNLI model
"""
import json
import csv
import tqdm
import os
import torch
import torch.optim as optim
import torch.nn as nn
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from analyze import initiate_exp_run 
import settings
import models.nli_models as models
import util
from data import analysis
import importlib.util
import train_utils
import prune_utils
import sys
sys.path.append("CCE_NLI/Analysis/")
#import ..Analysis as Analysis
#import alignment
def main(args):
    if args.debug:
        max_data = 1000
    else:
        max_data = None
  
        
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    if args.untrained_model:
        use_pretrained_weights = False
    else:
        use_pretrained_weights = True
        
        
    path_to_overlap = os.path.join(args.model_type.upper(), "overlap", args.pruning_method, args.filename)
    
      
    train,val,dataloaders=train_utils.create_dataloaders(model_type=args.model_type,pruning_method=args.pruning_method, max_data=max_data, debug=args.debug)
    
    ckpt = os.path.join(args.model_type.upper(), "models", 'lottery_ticket', args.filename, '0_Pruning_Iter/model_best.pth')
    
    model,ckpt = train_utils.load_model( model_type=args.model_type, pruning_method=args.pruning_method, use_pretrained_weights = use_pretrained_weights, train=train, ckpt=ckpt, device=device)
    
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(ckpt, map_location = torch.device(device)) #trained bowman/bert 
        
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    all_fm_masks = prune_utils.run_expls(args, model,dataset, dataloaders,device, args.debug)
    
    #alignment.calculate_alignment(all_fm_masks, path_to_overlap)  
    return all_fm_masks
    
#running the expls using the already finetuned and precreated masks from before

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    #parser.add_argument("--expls_mask_root_dir", default="exp/bert/lottery_ticket/Run1")
    #parser.add_argument("--form_mask_root_dir", default="formula_masks/bowman/Run2FIXEDLTH")
    #parser.add_argument("--activations_root_dir", default="activations/bert/lottery_ticket/Run1")
    #parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/BERT/Run1")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert", 'llama'])
    parser.add_argument("--filename", default="Run_Test")
    
    parser.add_argument("--untrained_model", action="store_true", default=False)  # If `--untrained_model` is used, set to True 
    
    parser.add_argument("--pruning_method", default="lottery_ticket", choices=["lottery_ticket", "wanda", "osscar"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=99, type=float)
    
    parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--prune_iters", default=5, type=int)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--ckpt", default=None)
    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
