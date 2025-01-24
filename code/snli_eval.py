"""
Train a bowman et al-style SNLI model
"""


import os
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from data.snli import SNLI, pad_collate
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import spacy
import pandas as pd

import settings
import models
import util
import train_utils
import data.snli


def predict(model, premise, hypothesis, nlp, stoi, args):
    pre, prelen = tokenize(premise, nlp, stoi)
    hyp, hyplen = tokenize(hypothesis, nlp, stoi)

    # unbatch
    pre = pre.unsqueeze(1)
    prelen = torch.tensor([prelen])
    hyp = hyp.unsqueeze(1)
    hyplen = torch.tensor([hyplen])

    if args.cuda:
        pre = pre.cuda()
        prelen = prelen.cuda()
        hyp = hyp.cuda()
        hyplen = hyplen.cuda()

    with torch.no_grad():
        logits = model(pre, prelen, hyp, hyplen)
        #  reprs = model.get_final_reprs(pre, prelen, hyp, hyplen)
        #  print(reprs[0, 39])
    pred = logits.squeeze(0).argmax().item()
    predtxt = data.snli.LABEL_ITOS[pred]
    return predtxt


def tokenize(text, nlp, stoi):
    toks = [t.lower_ for t in nlp(text)]
    ns = [stoi.get(t, stoi["UNK"]) for t in toks]
    return torch.tensor(ns), len(ns)


def from_stdin():
    while True:
        pre_raw = input("Premise: ")
        hyp_raw = input("Hypothesis: ")
        yield pre_raw, hyp_raw


def from_file(fpath):
    with open(fpath, "r") as f:
        lines = list(f)

    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]
    lines = [l for l in lines if not l.startswith("#")]

    if len(lines) % 2 != 0:
        raise RuntimeError("uneven src/hyp")

    for i in range(0, len(lines), 2):
        pre_raw = lines[i]
        hyp_raw = lines[i + 1]
        yield pre_raw, hyp_raw


    
def main(args):
    print("using weights from ", settings.MODEL)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load("models/snli/bowman_trained_no_prune.pth")
    stoi = ckpt["stoi"]
    train,_,_,dataloaders=train_utils.create_dataloaders(max_data=10000)
    # ==== BUILD MODEL ====
    model = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512)
    
    
    

    
    model.eval()

    if settings.CUDA:
        model = model.cuda()
    val_loader = dataloaders['val']
    pruned_percents=[0.0,20.0,36.0,48.8,59.04,67.232,73.786,79.028,83.223,86.578,89.263,91.41,93.128,94.502,95.602]
    weights_dir="models/snli/prune_metrics/wanda/bowman/"
    for i,direct in enumerate(os.listdir(weights_dir)):
        if direct == '.ipynb_checkpoints' or not direct[0].isdigit():
            continue
        print(direct)
        model.load_state_dict(torch.load(weights_dir+direct+"/model_best.pth")['state_dict'])
        all_preds = []
        all_targets = []
        for (s1, s1len, s2, s2len, targets) in val_loader:
            if settings.CUDA:
                s1 = s1.cuda()
                s1len = s1len.cuda()
                s2 = s2.cuda()
                s2len = s2len.cuda()

            with torch.no_grad():
                logits = model(s1, s1len, s2, s2len)

            preds = logits.argmax(1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, 0)
        all_targets = np.concatenate(all_targets, 0)
        acc = (all_preds == all_targets).mean()
        print(f"Val acc: {acc:.3f}")



    # ==== INTERACTIVE ====
    


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        default="test.txt",
        help="Data to eval interactively (pairs of sentences); use - for stdin",
    )
    parser.add_argument("--model", default="models/snli/model_best.pth")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "snli"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_data_path", default="data/snli_1.0/")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
