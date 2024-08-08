
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
    
    
    

#running the expls using the already finetuned and precreated masks from before
def main(args):
    
    settings.PRUNE_METHOD='lottery_ticket'
    settings.PRUNE_AMT=0.2
    os.makedirs(args.exp_dir, exist_ok=True)

    # ==== LOAD DATA ====
    if args.debug:
        max_data = 1000
    else:
        max_data = 10000
   
    # ==== BUILD MODEL ====
    path_to_ckpt="models/snli/6.pth"
    ckpt = torch.load(path_to_ckpt, map_location="cpu")
    model = train_utils.build_model(vocab_size=len(ckpt["stoi"]), model_type='bowman', embedding_dim=300, hidden_dim=512)
    model.load_state_dict(ckpt["state_dict"])

    if settings.CUDA:
        device = 'cuda'
        model = model.cuda()
    else:
        device = 'cpu'
        
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # ==== BUILD VOCAB ====
    vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)

    # ==== Data ====
    
    #initial val accuracy
    
    train = SNLI(
        "data/snli_1.0",
        "train",
        max_data=max_data,
    )
    
    train_loader = DataLoader(
        train,
        batch_size=100,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
        collate_fn=pad_collate,
    )
    
    val_test = SNLI(
        "data/snli_1.0",
        "dev",
        max_data=max_data,
        vocab=(vocab['stoi'], vocab['itos']),
        unknowns=True
    )
    
    val_test_loader = DataLoader(
        val_test,
        batch_size=100,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=pad_collate,
    )
    
    
    print(f"Accuracy: {train_utils.run_eval(model, val_test_loader)}")
 
    val_train = SNLI(
        "data/snli_1.0",
        "dev",
        max_data=max_data,
        vocab=(vocab['stoi'], vocab['itos']),
        unknowns=False
    )
    
    val_train_loader = DataLoader(
        val_train,
        batch_size=100,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=pad_collate,
    )
    
    # setting up pruning mask and weights
    final_weights=model.mlp[:-1][0].weight.detach().cpu().numpy()
    prune_mask = torch.ones(final_weights.shape)
    
    #pruning
    for prune_iter in tqdm(range(1,args.prune_iters+1)):
        print(f"==== PRUNING ITERATION {prune_iter} ====")
        #location to store metrics
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(args.prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)

        # prune 
        print("Prune amt", settings.PRUNE_AMT)
        model, prune_mask = model.prune(amount=settings.PRUNE_AMT,final_weights=final_weights, mask=prune_mask)
        print("After pruning: Final_wegihts prune% is: ", torch.where(model.mlp[0].weight.detach() == 0,1,0).sum()/(1024*2048))
        
        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
       
        #finetuning
        dataloaders = {
            'train': train_loader,
            'val':val_train_loader,
        }
        
        model, final_weights, ckpt= train_utils.finetune_pruned_model(model,optimizer,criterion, train, val_train, dataloaders, args.finetune_epochs, args.prune_metrics_dir, device)
        assert(torch.equal(model.mlp[0].weight.detach().cpu(), torch.tensor(final_weights)))
        print("After fting: Final_wegihts prune% is: ", torch.where(model.mlp[0].weight.detach() == 0,1,0).sum()/(1024*2048))
        

        #accuracy after finetuning 
        '''val = SNLI(
            "data/snli_1.0/",
            "dev",
            vocab=(ckpt["stoi"], ckpt["itos"]),
            max_data=max_data,
        )
        val_loader = DataLoader(
            val,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        
        )'''
        
        print(f"Accuracy: {train_utils.run_eval(model, val_test_loader)}")


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
    parser.add_argument("--finetune_epochs", default=20, type=int)
    parser.add_argument("--prune_iters", default=5, type=int)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
