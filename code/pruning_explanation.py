
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
import sys
sys.path.append("CCE_NLI/Analysis/")
#import ..Analysis as Analysis
import alignment
def main(args):
    os.makedirs(args.expls_mask_root_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
        
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    model, dataloaders = prune_utils.get_model(args.model_type, args.ckpt, device)
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(args.ckpt, map_location = torch.device(device)) #trained bowman/bert 
        
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    all_fm_masks = prune_utils.run_expls(args, model,dataset, dataloaders,device)
    #with open(f"formula_masks/{args.model_type}/formula_masks.json", "w") as f:
        #json.dump(all_fm_masks, f)
    alignment.calculate_alignment(all_fm_masks, f"overlap/{args.model_type}/{args.pruning_method}")  
    return all_fm_masks
    
#running the expls using the already finetuned and precreated masks from before

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--expls_mask_root_dir", default="exp/bert/lottery_ticket/Run1")
    parser.add_argument("--activations_root_dir", default="activations/bert/lottery_ticket/Run1")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/BERT/Run1")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert"])
    parser.add_argument("--pruning_method", default="bowman", choices=["lottery_ticket", "wanda"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=95, type=float)
    
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
    main(args)
