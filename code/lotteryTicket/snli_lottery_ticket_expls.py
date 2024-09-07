
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
    exp_after_finetuning_flder = f"Analysis/LHExpls/Run1/Expls{prune_iter}_Pruning_Iter/Min_Acts_500_No_Filters/"
    if not os.path.exists(exp_after_finetuning_flder):
        os.mkdir(exp_after_finetuning_flder) 

    masks_after_finetuning_flder = f"code/LHMasks/Run1/Masks{prune_iter}_Pruning_Iter/"
    if not os.path.exists(masks_after_finetuning_flder):
        os.mkdir(masks_after_finetuning_flder)
    return exp_after_finetuning_flder, masks_after_finetuning_flder


def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data)
    model = train_utils.load_model(max_data=max_data, train=train, ckpt=args.ckpt)
    base_ckpt=torch.load(settings.MODEL) 
    
    # ==== BUILD VOCAB ====
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    device = 'cuda' if settings.CUDA else 'cpu'
    return run_prune(model, dataset, optimizer, criterion,train,val,test,dataloaders,device,max_thresh=95, min_thresh=20, prune_iters = args.prune_iters, prune_metrics_dirs=args.prune_metrics_dir)
    
#running the expls using the already finetuned and precreated masks from before
def run_prune(model, dataset, optimizer, criterion, train,val,test,dataloaders,device, max_thresh, min_thresh, prune_iters, prune_metrics_dirs, pruned_percents=[], final_accs=[]):
    base_ckpt=torch.load(settings.MODEL) 
    for prune_iter in tqdm(range(0,prune_iters+1)):
        print(f"==== PRUNING ITERATION {prune_iter}/{prune_iters+1} ====")
        prune_metrics_dir = os.path.join(prune_metrics_dirs, f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(prune_metrics_dirs,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)
        model.to(device)
        print("Loading from ",f"{prune_metrics_dir}/model_best.pth")
        model.load_state_dict(torch.load(f"{prune_metrics_dir}/model_best.pth")['state_dict']) #loading the already finetuned weights
        final_weights = model.mlp[0].weight.detach().cpu().numpy()
        
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
        print("Bfore pruning: Final_wegihts prune% is: ", bfore)
        
        final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(1024*2048), 3)
        
        acc=train_utils.run_eval(model, dataloaders['test'])
        print(f"Accuracy: {acc}")
        pruned_percents.append(final_weights_pruned)
        final_accs.append(acc)
        
        exp_after_finetuning_flder, masks_after_finetuning_flder = make_folders(final_weights_pruned)
        
        if final_weights_pruned >= max_thresh or final_weights_pruned <= min_thresh:
            print(f"======Running Explanations for {final_weights_pruned}% pruned=======")
            #run after pruning before finetuning
            for num_clusters in [3,4,5,6,7,10,15]:
                os.makedirs(f"{exp_after_finetuning_flder}{num_clusters}Clusters", exist_ok=True)
                os.makedirs(f"{masks_after_finetuning_flder}{num_clusters}Clusters", exist_ok=True)
                
                print(f"======RUNNING EXPLANATIONS WITH {num_clusters} CLUSTERS")
                _,final_layer_weights =initiate_exp_run(
                    save_exp_dir = f"{exp_after_finetuning_flder}{num_clusters}Clusters", 
                    save_masks_dir= f"{masks_after_finetuning_flder}{num_clusters}Clusters", 
                    masks_saved=True, 
                    model_=model,
                    dataset=dataset,
                    clusters=num_clusters,
                )
        
        assert(torch.equal(model.mlp[0].weight.detach().cpu(), torch.tensor(final_weights)))
        
     
        
        
        
        bfore=np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(1024*2048),2)
        print("After pruning: Final_wegihts prune% is: ", bfore)
        
   
        
        if final_weights_pruned > max_thresh: break
    return init_acc, pruned_percents, final_accs

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp_dir", default="models/snli/LH")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/LH/Run1")
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
    init_acc, pruned_percents, final_accs = main(args)
    print(f"init_acc: {init_acc}\npruned_percents: {pruned_percents}\nfinal_accs: {final_accs}")
