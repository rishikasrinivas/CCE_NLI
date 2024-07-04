"""
Train a bowman et al-style SNLI model
"""

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
# Define the path to the module you want to import
analysis_path = os.path.abspath("Analysis/pipelines.py")

# Load the module dynamically
spec = importlib.util.spec_from_file_location("pipelines", analysis_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sys.path.append("Analysis/")
import pipelines as pipelines
def finetune_pruned_model(model,optimizer,criterion, dataloaders, train, val, finetune_epochs, prune_metrics_dir, model_dir, metrics,device):
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


        util.save_metrics(metrics, prune_metrics_dir)
        util.save_checkpoint(serialize(model, train), is_best, prune_metrics_dir)
        if epoch % args.save_every == 0:

            util.save_checkpoint(
                serialize(model, train), False, prune_metrics_dir, filename=f"LotTick{epoch}.pth"
            )
    
    torch.save(model.state_dict(), model_dir)
    path_to_ckpt=f"{prune_metrics_dir}/model_best.pth"
    return path_to_ckpt, metrics
        

def main(args):
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
        path_to_ckpt= "models/snli/prune_metrics/0.5%/model_best.pth"
    else:
        path_to_ckpt=settings.MODEL
        
    ckpt = torch.load(path_to_ckpt, map_location="cpu")
    clf = models.BowmanEntailmentClassifier
    enc = models.TextEncoder(len(ckpt["stoi"]))
    model = clf(enc)
    model.load_state_dict(ckpt["state_dict"])

    if args.cuda:
        device = 'cuda'
        model = model.cuda()
    else:
        device = 'cpu'
    
    vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}

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
    
    masks_saved=False
    
    initial_expls = {'original': 
                     ["Analysis/Explanations/Cluster1IOUSOrig.csv",
                      "Analysis/Explanations/Cluster2IOUSOrig.csv",
                      "Analysis/Explanations/Cluster3IOUSOrig.csv",
                     "Analysis/Explanations/Cluster4IOUSOrig.csv"
                     ]
                    }
    # ==== TRAIN ====
    
    for prune_iter in tqdm(range(1,args.prune_iters+1)):
        #identifier to track pruning amount'
        if prune_iter == 1:
            assert model.check_pruned() == False
            masks_saved=True
        else:
            masks_saved=False
            assert model.check_pruned()

        identifier = 0.005*prune_iter*100

        #masks and explanation storing paths before finetuning
        masks_before_finetuning_flder = f"code/Masks{identifier}%Pruned/BeforeFT"
        if not os.path.exists(masks_before_finetuning_flder):
            os.mkdir(f"code/Masks{identifier}%Pruned")
            os.mkdir(masks_before_finetuning_flder)


        expls_before_finetuning_flder = f"Analysis/Expls{identifier}%Pruned/BeforeFT"
        if not os.path.exists(expls_before_finetuning_flder):
            os.mkdir(f"Analysis/Expls{identifier}%Pruned")
            os.mkdir(expls_before_finetuning_flder) 



        #run after pruning before finetuning
        initiate_exp_run(
            save_exp_dir = expls_before_finetuning_flder, 
            save_masks_dir= masks_before_finetuning_flder, 
            masks_saved=masks_saved, 
            path = path_to_ckpt,
            adjust_final_weights=True,
            amount=0.005,
            model_=model,
            dataset=dataset,
        )


        #ANALYSIS: % of lost concepts
        prunedBeforeRT_expls = {'prunedBefore': [
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster1IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster2IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster3IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster4IOUS1024N.csv",
        ]}


        percent_concepts_lost_to_pruning_local = pipelines.pipe_percent_lost(
            [initial_expls,prunedBeforeRT_expls],
            task = 'local',
            fname = f"Analysis/Expls{identifier}%Pruned/LostTo{identifier}%PruningBeforeFinetune.csv"

        )

        percent_concepts_lost_to_pruning_globally = pipelines.pipe_percent_lost(
            [initial_expls,prunedBeforeRT_expls],
            task = 'global'
        )



        #location to store metrics
        prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{identifier}%")
        if not os.path.exists(prune_metrics_dir):
            os.mkdir(prune_metrics_dir)

        model_dir = os.path.join(args.model_dir,f"{identifier}%")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        #finetuning

        model.prune(amount=0.005) #0.5% prune
        assert model.check_pruned()
        prunedBeforeRT_expls = {'prunedBefore': [
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster1IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster2IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster3IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster4IOUS1024N.csv",
        ]}

        path_to_ckpt, metrics = finetune_pruned_model(model,optimizer,criterion,dataloaders, train,val, args.finetune_epochs, prune_metrics_dir, model_dir, metrics, device)

        if args.cuda:
            device = 'cuda'
            model = model.cuda()
        else:
            device = 'cpu'
        #masks and explanation storing paths after finetuning
        exp_after_finetuning_flder = f"Analysis/Expls{identifier}%Pruned/AfterFT"
        if not os.path.exists(exp_after_finetuning_flder):
            os.mkdir(exp_after_finetuning_flder) 

        masks_after_finetuning_flder = f"code/Masks{identifier}%Pruned/AfterFT"
        if not os.path.exists(masks_after_finetuning_flder):
            os.mkdir(masks_after_finetuning_flder)

        #run after pruning and finetuning
        initiate_exp_run(
            save_exp_dir = exp_after_finetuning_flder, 
            save_masks_dir= masks_after_finetuning_flder, 
            masks_saved=masks_saved, 
            path = path_to_ckpt,
            adjust_final_weights=False,
            model_=model,
            dataset=dataset,

        ) 
        prunedBeforeRT_expls = {'prunedBefore': [
                f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster1IOUS1024N.csv",
                f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster2IOUS1024N.csv",
                f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster3IOUS1024N.csv",
                f"Analysis/Expls{identifier}%Pruned/BeforeFT/Cluster4IOUS1024N.csv",
            ]}

        #ANALYSIS measure local consistency and global consistency
        prunedAfterRT_expls = {'prunedAfter': [
            f"Analysis/Expls{identifier}%Pruned/AfterFT/Cluster1IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/AfterFT/Cluster2IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/AfterFT/Cluster3IOUS1024N.csv",
            f"Analysis/Expls{identifier}%Pruned/AfterFT/Cluster4IOUS1024N.csv",
        ]}

        percent_of_cps_preserved_globally = pipelines.pipe_explanation_similiarity(
            [initial_expls,prunedAfterRT_expls], 
            task='global', 
            get_concepts_func = 'indiv',
        )

        percent_of_cps_preserved_locally = pipelines.pipe_explanation_similiarity(
            [initial_expls,prunedAfterRT_expls],
            task='local', 
            get_concepts_func = 'indiv',
            fname = f"Analysis/Expls{identifier}%Pruned/LocallyPreserved{identifier}%Pruned.csv"
        )

        percent_relearned_through_finetuning = pipelines.pipe_relearned_concepts(
            [initial_expls,prunedBeforeRT_expls,prunedAfterRT_expls], 
            task='global', 
            get_concepts_func = 'indiv'
        )

        percent_relearned_through_finetuning = pipelines.pipe_relearned_concepts(
            [initial_expls,prunedBeforeRT_expls,prunedAfterRT_expls], 
            task='local', 
            get_concepts_func = 'indiv',
            fname = f"Analysis/Expls{identifier}%Pruned/LocallyRelearned{identifier}%Pruned.csv"
        )

        tqdm.write(f"% lost to pruning globally: {percent_concepts_lost_to_pruning_globally}%\n% perserved globally: {percent_of_cps_preserved_globally}%\n% relearned by finetuning: {percent_relearned_through_finetuning}%\n")



def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp_dir", default="models/snli/")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics")
    parser.add_argument("--model_dir", default="models/snli/model_dir")

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
