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
from transformers import AutoConfig, AutoTokenizer
import settings
import models
import util
import train_utils
import data.snli
from cofi.utils.utils import calculate_parameters

from cofi.utils.cofi_utils import load_model
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

def get_percent_pruned(model):
    final_weights = model.mlp[0].weight.detach().cpu().numpy()
    return 1 - ((final_weights.shape[0]*2048) + (final_weights.shape[0] * 3)) /((1024*2048)+(1024*3))
    final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(model.mlp[0].weight.shape[0]*model.mlp[0].weight.shape[1]), 3)
    return final_weights_pruned

def main(args):
    print("using weights from ", args.ckpt)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    ckpt = torch.load(args.ckpt, map_location = 'cuda' if settings.CUDA else 'cpu')
    print("Building dataset")
    train,_,dataloaders=train_utils.create_dataloaders(max_data=None, model_type=args.model_type, pruning_method=args.pruning_method)
    # ==== BUILD MODEL ====
    model,_ = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='CoFi')
   
    val_loader = dataloaders['val']
    accs = {}
    
    def fill_inputs_with_zs(zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
        return inputs

    
    try:
        if args.pruning_method == 'CoFi':
            if args.model_type in ['bert', 'llama']:
                #tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.root_dir, f"0_Pruning_Iter"), trust_remote_code=True)
                tokenizer=None
            else:
                tokenizer = model.encoder


            zs=torch.load(os.path.join(args.root_dir,"zs.pt"))

            pruned_model = load_model(os.path.join(args.root_dir), model, zs,tokenizer, train_data=train, ckpt=os.path.join(args.root_dir, 'model_best.pth'), device='cuda')
            pruned_model.eval()

            if settings.CUDA:
                pruned_model.cuda()
            if args.model_type in ['bert', 'llama']:
                pruned_model_size = calculate_parameters(pruned_model)
                if folder == '0_Pruning_Iter': og=pruned_model_size
                final_weights_pruned = 1 - (pruned_model_size / og) 
            else:
                final_weights_pruned = get_percent_pruned(pruned_model)
            print("sparsity=", final_weights_pruned)

            all_preds = []
            all_targets = []

            # CORRECTED: Added conditional logic for batch handling
            for batch in dataloaders['val']:

                if torch.cuda.is_available():
                    #batch = fill_inputs_with_zs(zs, batch)
                    if settings.CUDA:
                        batch = {k: v.to('cuda') for k, v in batch.items()}
                    targets = batch['labels']


                batch_size = targets.shape[0]

                with torch.no_grad():
                    logits = pruned_model(**batch)

                preds = logits[1][2].argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            all_preds = np.concatenate(all_preds, 0)
            all_targets = np.concatenate(all_targets, 0)
            acc = (all_preds == all_targets).mean()
            print(np.round(acc, 3))
        else:
            torch.cuda.empty_cache()
            if '.ipy' in folder or not folder[0].isdigit(): return
            weights = torch.load(os.path.join(args.root_dir, folder, 'model_best.pth'))['state_dict']
            model.load_state_dict(weights)
            all_preds = []
            all_targets = []
            model.eval()
            print("Loaded some apth ", os.path.join(args.root_dir, 'model_best.pth'))
            if settings.CUDA:
                model = model.cuda()
            if args.model_type=='bowman':
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
            else:
                for s1, s2, targets in val_loader:
                    s1={k:v.cuda() for k,v in s1.items()}
                    s2={k:v.cuda() for k,v in s2.items()}

                    with torch.no_grad():
                        logits = model(s1, s2)

                    preds = logits.argmax(1)

                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())


            all_preds = np.concatenate(all_preds, 0)
            all_targets = np.concatenate(all_targets, 0)

            acc = (all_preds == all_targets).mean()

            print(f" Val acc: {acc:.3f}")
        #accs[folder]=np.round(acc,3)
    except Exception as e:
        print(e)
    #pd.DataFrame({'folder':accs.keys(), 'accs':accs.values()}).to_csv(f"{args.root_dir}/accuracy.csv")


   


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
    
    parser.add_argument("--root_dir", default="/workspace/CCE_NLI/BOWMAN/models/lottery_ticket/Run0.25/")
    parser.add_argument("--ckpt", default="BOWMAN/models/lottery_ticket/Run0.25/0_Pruning_Iter/model_best.pth")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "bert", "llama"])
    parser.add_argument("--pruning_method", default="lottery_ticket", choices=["lottery_ticket", "wanda", "CoFi"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_data_path", default="data/snli_1.0/")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
