"""
Train a bowman et al-style SNLI model
"""


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

import models
import util


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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    clf = models.BowmanEntailmentClassifier
    model = clf(enc)
    model.load_state_dict(ckpt["state_dict"])

    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    metrics = defaultdict(list)
    metrics["best_val_epoch"] = 0
    metrics["best_val_acc"] = 0
    metrics["best_val_loss"] = np.inf

    # Save model with 0 training
    model_c = model
    # ==== TRAIN ====
    for prune_iter in range(args.prune_iters):
        
        initiate_exp_run(save_dir = f"code/Masks{0.005*(prune_iter)*100}%Pruned/")
        
        #prune .5% model.mlp[:-1][0] = prune.ln_structured(model.mlp[:-1][0], name="weight", amount=0.005, dim=1, n=float('-inf'))
        model.prune(amount=0.005)
        #rerun train copy_weigths 
        model.copy_weights_linear(model_c, model)
        
        train_metrics = run(
            "train", prune_iter, model, optimizer, criterion, dataloaders, args
        )

        val_metrics = run(
            "val", prune_iter, model, optimizer, criterion, dataloaders, args
        )
    
    for epoch in range(args.main_epochs):
        train_metrics = run(
            "train", epoch, model, optimizer, criterion, dataloaders, args
        )
        val_metrics = run(
            "val", epoch, model, optimizer, criterion, dataloaders, args
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

        util.save_metrics(metrics, args.exp_dir)
        util.save_checkpoint(serialize(model, train), is_best, args.exp_dir)
        if epoch % args.save_every == 0:
            util.save_checkpoint(
                serialize(model, train), False, args.exp_dir, filename=f"LotTick{epoch}.pth"
            )


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp_dir", default="models/snli/")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal"])
    parser.add_argument("--save_every", default=1, type=int)
    
    parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--main_epochs", default=50, type=int)
    parser.add_argument("--prune_iters", default=5, type=int)
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
