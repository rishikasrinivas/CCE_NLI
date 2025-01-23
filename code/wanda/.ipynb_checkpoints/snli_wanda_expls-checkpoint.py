import sys
sys.path.append('code/')
import train_utils 
import wanda
import torch
import prune_utils
import argparse
import os
import settings
from data import analysis
import torch
import torch.optim as optim
import torch.nn as nn
import settings
import util
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import wanda_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
  
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of calibration samples.')

    parser.add_argument("--expls_mask_root_dir", default="exp/bert/wanda/Run1")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/wanda/BERT/Run1")
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=95, type=float)
    
    parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--prune_iters", default=1, type=int)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(args.expls_mask_root_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    
    
    for folder in os.listdir(args.prune_metrics_dir):
        if folder ==  '.ipynb_checkpoints' or not folder[0].isdigit(): continue
        file_path = f"{args.prune_metrics_dir}/{folder}/model_best.pth"
        model,dataloaders = wanda_utils.get_model(args.model_type, file_path)
    
    # ==== BUILD VOCAB ====
        base_ckpt=torch.load(file_path) #models/snli/bowman_random_inits.pth
        vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

        with open(settings.DATA, "r") as f:
            lines = f.readlines()

        dataset = analysis.AnalysisDataset(lines, vocab)
        prune_utils.run_expls(args, model,dataset, dataloaders,device='cuda')
        
        
        

main()