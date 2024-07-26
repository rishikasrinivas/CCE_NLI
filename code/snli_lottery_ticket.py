
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
from data.snli import SNLI, pad_collate
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from analyze import initiate_exp_run
from snli_train import run, build_model, serialize 
import settings
import models
import util
from data import analysis
import importlib.util
import sys
import fileio
# Define the path to the module you want to import
analysis_path = os.path.abspath("Analysis/pipelines.py")

# Load the module dynamically
spec = importlib.util.spec_from_file_location("pipelines", analysis_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sys.path.append("Analysis/")
import pipelines as pipelines

def finetune_pruned_model(model,optimizer,criterion, dataloaders, train, val, finetune_epochs, prune_metrics_dir, metrics,device):
    for epoch in range(finetune_epochs):
        train_metrics = run(
            "train", epoch, model, optimizer, criterion, dataloaders, args,device
        )

        val_metrics = run(
            "val", epoch, model, optimizer, criterion, dataloaders, args,device
        )

        for name, val in train_metrics.items():
            metrics[f"train_{name}"].append(val)

        for name, val in val_metrics.items():
            metrics[f"val_{name}"].append(val)

        is_best = val_metrics["acc"] > metrics["best_val_acc"]

        if is_best:
            metrics["best_val_epoch"] = epoch
            metrics["best_val_acc"] = val_metrics["acc"]
            metrics["best_val_loss"] = val_metrics["loss"]
            fileio.log_to_csv(os.path.join(prune_metrics_dir,"pruned_status.csv"), [epoch, val_metrics["acc"], val_metrics["loss"]], ["EPOCH", "ACCURACY", "LOSS"])
        


        util.save_metrics(metrics, prune_metrics_dir)
        util.save_checkpoint(serialize(model, train), is_best, prune_metrics_dir)
        if epoch % args.save_every == 0:

            util.save_checkpoint(
                serialize(model, train), False, prune_metrics_dir, filename=f"LotTick{epoch}.pth"
            )
    path_to_ckpt = os.path.join(prune_metrics_dir, f"LotTick{finetune_epochs-1}.pth")
    return path_to_ckpt, metrics, model

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
        max_data = None
    train = SNLI("data/snli_1.0/", "train", max_data=max_data)
    val = SNLI(
        "data/snli_1.0/", "dev", max_data=max_data, vocab=(train.stoi, train.itos)
    )
   
    dataloaders = {
        "train": DataLoader(
            train,
            batch_size=100,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate,
        ),
        "val": DataLoader(
            val,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        ),
    }

    # ==== BUILD MODEL ====
    resume_from_ckpt = False
    if resume_from_ckpt:
        path_to_ckpt= f"models/snli/prune_metrics/0.5%/model_best.pth"
    else:
        path_to_ckpt=settings.MODEL
        
    ckpt = torch.load(path_to_ckpt, map_location="cpu")
    ckpt_orig=torch.load(settings.MODEL, map_location="cpu")
    clf = models.BowmanEntailmentClassifier
    enc = models.TextEncoder(len(ckpt_orig["stoi"])).cuda()
    model=clf(enc)
    model.load_state_dict(ckpt["state_dict"])

    if settings.CUDA:
        device = 'cuda'
        model = model.cuda()
    else:
        device = 'cpu'
    
    vocab = {"itos": ckpt_orig["itos"], "stoi": ckpt_orig["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)



    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    metrics = defaultdict(list)
    metrics["best_val_epoch"] = 0
    metrics["best_val_acc"] = 0
    metrics["best_val_loss"] = np.inf

    # Save model with 0 training
    

    # ==== TRAIN ====
    final_weights=model.mlp[0].weight.t().detach().cpu().numpy()
    prune_mask = torch.ones(final_weights.shape)
    
    for prune_iter in tqdm(range(1,args.prune_iters+1)):
        
        #identifier to track pruning amount'
        
        #masks and explanation storing paths before finetuning
        masks_before_finetuning_flder = f"code/LHMasks/Masks{prune_iter}_Pruning_Iter/BeforeFT"
        if not os.path.exists(masks_before_finetuning_flder):
            os.makedirs(f"code/LHMasks/", exist_ok=True)
            os.makedirs(f"code/LHMasks/Masks{prune_iter}Pruning_Iter", exist_ok=True)
            os.makedirs(masks_before_finetuning_flder,exist_ok=True)


        expls_before_finetuning_flder = f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/BeforeFT"
        if not os.path.exists(expls_before_finetuning_flder):
            os.makedirs(f"Analysis/LHExpls/", exist_ok=True)
            os.makedirs(f"Analysis/Expls{prune_iter}Pruning_Iter",exist_ok=True)
            os.makedirs(expls_before_finetuning_flder,exist_ok=True) 

        #location to store metrics
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(args.prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)

        #masks and explanation storing paths after finetuning
        exp_after_finetuning_flder = f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/AfterFT"
        if not os.path.exists(exp_after_finetuning_flder):
            os.mkdir(f"Analysis/LHExpls/Expls{prune_iter}Pruning_Iter/")
            os.mkdir(exp_after_finetuning_flder) 

        masks_after_finetuning_flder = f"code/LHMasks/Masks{prune_iter}_Pruning_Iter/AfterFT"
        if not os.path.exists(masks_after_finetuning_flder):
            os.mkdir(masks_after_finetuning_flder)

        print("Prune amt", settings.PRUNE_AMT)
        if prune_iter != 1:
            model, prune_mask, new_weights = model.prune(amount=settings.PRUNE_AMT,final_weights=final_weights, mask=prune_mask)
            assert  torch.equal(model.mlp[:-1][0].weight.t().detach().cpu(), new_weights)
            if settings.CUDA:
                device = 'cuda'
                model = model.cuda()
            #run after pruning before finetuning
            _,final_weights =initiate_exp_run(
                        save_exp_dir = expls_before_finetuning_flder, 
                        save_masks_dir= masks_before_finetuning_flder, 
                        masks_saved=False, 
                        model_=model,
                        dataset=dataset,
                    )
        else:
            model, prune_mask, new_weights = model.prune(amount=settings.PRUNE_AMT,final_weights=final_weights, mask=prune_mask)
        assert  torch.equal(model.mlp[:-1][0].weight.t().detach().cpu(), new_weights)
        path_to_ckpt, metrics, model = finetune_pruned_model(model,optimizer,criterion,dataloaders, train, val, args.finetune_epochs, args.prune_metrics_dir, metrics, device)


        prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{prune_iter}_Pruning_Iter")
        weights=torch.load(f"{args.prune_metrics_dir}/model_best.pth")['state_dict']['mlp.0.weight']
        total_pruned_amt=torch.where(weights==0,1,0).sum()
        fileio.log_to_csv(os.path.join(prune_metrics_dir,"pruned_status.csv"), str(total_pruned_amt / (1024*2048)), f"{prune_iter}: % PRUNED")
        


        if settings.CUDA:
            device = 'cuda'
            model = model.cuda()
        else:
            device = 'cpu'
            
        
        #run after pruning and finetuning
        _,final_weights=initiate_exp_run(
            save_exp_dir = exp_after_finetuning_flder, 
            save_masks_dir= masks_after_finetuning_flder, 
            masks_saved=False, 
            model_=model,
            dataset=dataset,

        ) 
        
        
        prunedAfterRT_expls = {'prunedAfter': [
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/AfterFT/Cluster1IOUS1024N.csv",
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/AfterFT/Cluster2IOUS1024N.csv",
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/AfterFT/Cluster3IOUS1024N.csv",
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/AfterFT/Cluster4IOUS1024N.csv",
        ]}
        
        prunedBeforeRT_expls = {'prunedBefore': [
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/BeforeFT/Cluster1IOUS1024N.csv",
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/BeforeFT/Cluster2IOUS1024N.csv",
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/BeforeFT/Cluster3IOUS1024N.csv",
            f"Analysis/LHExpls/Expls{prune_iter}_Pruning_Iter/BeforeFT/Cluster4IOUS1024N.csv",
        ]}
        initial_expls = {'original': 
                         [f"Analysis/LHExpls/Expls0_Pruning_Iter/AfterFT/Cluster1IOUS1024N.csv",
                          f"Analysis/LHExpls/Expls0_Pruning_Iter/AfterFT/Cluster2IOUS1024N.csv",
                          f"Analysis/LHExpls/Expls0_Pruning_Iter/AfterFT/Cluster3IOUS1024N.csv",
                          f"Analysis/LHExpls/Expls0_Pruning_Iter/AfterFT/Cluster4IOUS1024N.csv",

                         ]
                        }
        
        files=[prunedBeforeRT_expls, prunedAfterRT_expls, initial_expls]
        util.record_stats(args.prune_metrics_dir, prune_iter, files, '_')


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
