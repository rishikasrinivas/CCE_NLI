
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
# Define the path to the module you want to import
analysis_path = os.path.abspath("Analysis/pipelines.py")

# Load the module dynamically
spec = importlib.util.spec_from_file_location("pipelines", analysis_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sys.path.append("Analysis/")
import pipelines as pipelines

def log_to_csv(file, data):
    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header

        # Write the data rows
        writer.writerow(data)
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


        util.save_metrics(metrics, prune_metrics_dir)
        util.save_checkpoint(serialize(model, train), is_best, prune_metrics_dir)
        if epoch % args.save_every == 0:

            util.save_checkpoint(
                serialize(model, train), False, prune_metrics_dir, filename=f"LotTick{epoch}.pth"
            )
    path_to_ckpt = os.path.join(prune_metrics_dir, f"LotTick{finetune_epochs-1}.pth")
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
    resume_from_ckpt = True
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
        if prune_iter <= 2:
            assert model.check_pruned() == False
            masks_saved=True
        else:
            masks_saved=False
            assert model.check_pruned()==False #should have removd the keys 
        identifier = 0.005*prune_iter*100
        
        #masks and explanation storing paths before finetuning
        masks_before_finetuning_flder = f"code/Masks{identifier}%Pruned/BeforeFT"
        if not os.path.exists(masks_before_finetuning_flder):
            os.makedirs(f"code/Masks{identifier}%Pruned", exist_ok=True)
            os.makedirs(masks_before_finetuning_flder,exist_ok=True)


        expls_before_finetuning_flder = f"Analysis/Expls{identifier}%Pruned/BeforeFT"
        if not os.path.exists(expls_before_finetuning_flder):
            os.makedirs(f"Analysis/Expls{identifier}%Pruned",exist_ok=True)
            os.makedirs(expls_before_finetuning_flder,exist_ok=True) 

        if prune_iter != 1:

            #run after pruning before finetuning
            if prune_iter <= 2:
                initiate_exp_run(
                    save_exp_dir = expls_before_finetuning_flder, 
                    save_masks_dir= masks_before_finetuning_flder, 
                    masks_saved=masks_saved, 
                    path = path_to_ckpt,
                    adjust_final_weights=True,
                    amount=0.005,
                    model_=model,
                    dataset=dataset,
                    q_ret = 1,
                )
            else:
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
            assert model.check_pruned() == True
       
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
            print(f"percent_concepts_lost_to_pruning_globally: {percent_concepts_lost_to_pruning_globally}")



            #location to store metrics
            prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{identifier}%")
            if not os.path.exists(prune_metrics_dir):
                os.makedirs(args.prune_metrics_dir,exist_ok=True)
                os.makedirs(prune_metrics_dir,exist_ok=True)

            #finetuning

            model.prune(amount=0.005) #0.5% prune
            assert model.check_pruned()


            path_to_ckpt, metrics = finetune_pruned_model(model,optimizer,criterion,dataloaders, train,val, args.finetune_epochs, prune_metrics_dir, metrics, device)


            prune_metrics_dir = os.path.join(args.prune_metrics_dir,f"{identifier}%")
            weights=torch.load(f"{prune_metrics_dir}/model_best.pth")['state_dict']['mlp.0.weight']

            log_to_csv(os.path.join(prune_metrics_dir,"pruned_status.csv"), f"{prune_iter}: % PRUNED : {torch.where(weights==0,1,0).sum() / (1024*2048)}")


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
                masks_saved=False, 
                path = path_to_ckpt,
                adjust_final_weights=False,
                model_=model,
                dataset=dataset,

            ) 


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
            log_to_csv(os.path.join(prune_metrics_dir,"glbal_exp_sim_indiv.csv"), f"Explanation similarity individual concept level globally: {percent_of_cps_preserved_globally}")

            percent_of_comp_cps_preserved_globally = pipelines.pipe_explanation_similiarity(
                [initial_expls,prunedAfterRT_expls], 
                task='global', 
                get_concepts_func = 'group',
            )
            log_to_csv(os.path.join(prune_metrics_dir,"global_exp_sim_compos.csv"), f"Explanation similarity compositional concept level globally:  {percent_of_comp_cps_preserved_globally}")


            percent_of_cps_preserved_locally = pipelines.pipe_explanation_similiarity(
                [initial_expls,prunedAfterRT_expls],
                task='local', 
                get_concepts_func = 'indiv',
                fname = f"Analysis/Expls{identifier}%Pruned/LocallyPreserved{identifier}%Pruned.csv"
            )

            percent_relearned_through_finetuning_g = pipelines.pipe_relearned_concepts(
                [initial_expls,prunedBeforeRT_expls,prunedAfterRT_expls], 
                task='global', 
                get_concepts_func = 'indiv'
            )

            log_to_csv(os.path.join(prune_metrics_dir,"indiv_after_finetune_glob.csv"), f"% of indiv concepts relearned after finetuning globally: {percent_relearned_through_finetuning_g}")
            percent_relearned_through_finetuning_g_group = pipelines.pipe_relearned_concepts(
                [initial_expls,prunedBeforeRT_expls,prunedAfterRT_expls], 
                task='global', 
                get_concepts_func = 'group'
            )
            print("% of compositions relearned after finetuning globally: ", percent_relearned_through_finetuning_g_group)
            log_to_csv(os.path.join(prune_metrics_dir,"comp_relearned_glob.csv"), f"% of compositions relearned after finetuning globally: {percent_relearned_through_finetuning_g_group}")

            percent_relearned_through_finetuning_l = pipelines.pipe_relearned_concepts(
                [initial_expls,prunedBeforeRT_expls,prunedAfterRT_expls], 
                task='local', 
                get_concepts_func = 'indiv',
                fname = f"Analysis/Expls{identifier}%Pruned/LocallyRelearned{identifier}%Pruned.csv"
            )

    


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp_dir", default="models/snli/")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics")
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
