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
import settings
import torch.nn.utils.prune as prune
import train_utils
import models
import util



def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)

    # ==== LOAD DATA ====
    if args.debug:
        max_data = 1000
    else:
        max_data = 10000
        
    if args.finetune:
        ckpt = torch.load("models/snli/6.pth")
        train = SNLI("data/snli_1.0/", "train", max_data=max_data)
        val = SNLI(
            "data/snli_1.0/", "dev", max_data=max_data, vocab=(ckpt["stoi"], ckpt["itos"]),
        )
        vocab_size=len(ckpt['stoi'])
    else:
        train = SNLI("data/snli_1.0/", "train", max_data=max_data)
        val = SNLI(
            "data/snli_1.0/", "dev", max_data=max_data, vocab=(train.stoi, train.itos)
        )
        
        vocab_size=len(train.stoi)
        print(f"Vocab size: {vocab_size}")
        
        
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
    model = train_utils.build_model(
        vocab_size,
        args.model_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    if args.finetune:
        args.model= 'models/snli/model_best.pth'
        print("Loading model from ",args.model )
        model.load_state_dict(torch.load(args.model)['state_dict'])
        
    final_weights=model.mlp[0].weight.detach().cpu().numpy()
    if args.iter == 1:
        prune_mask = torch.ones(final_weights.shape)
    else:
        prune_mask = torch.load(f"models/snli/mask/mask{args.iter-1}.pth")
    settings.PRUNE_METHOD='lottery_ticket'
    
    model, mask=model.prune(amount=0.2, final_weights=final_weights, mask=prune_mask)
    torch.save(mask, f"models/snli/mask/mask{args.iter}.pth")
    print("Weights pruned ", torch.where(model.mlp[0].weight.detach().cpu() == 0,1,0).sum()/(1024*2048))
    

    if settings.CUDA:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model, final_weights, _= train_utils.finetune_pruned_model(model,optimizer,criterion, train, val, dataloaders, 10, 'models/snli', device='cuda')

    '''
    metrics = defaultdict(list)
    metrics["best_val_epoch"] = 0
    metrics["best_val_acc"] = 0
    metrics["best_val_loss"] = np.inf
    


    # Save model with 0 training
    #util.save_checkpoint(serialize(model, train), False, args.exp_dir, filename="0.pth")
    print("init % pruned: ",torch.where(model.mlp[0].weight.detach().cpu()==0,1,0).sum()/(1024*2048))
    
    # ==== TRAIN ====
    for epoch in range(args.epochs):
        
        train_metrics = train_utils.run(
            "train", epoch, model, optimizer, criterion, dataloaders['train']
        )
        
        val_metrics = train_utils.run("val", epoch, model, optimizer, criterion, dataloaders['val'])

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
        util.save_checkpoint(train_utils.serialize(model, train), is_best, args.exp_dir)
        if epoch % args.save_every == 0:
            util.save_checkpoint(
                train_utils.serialize(model, train), False, args.exp_dir, filename=f"train{epoch}.pth"
            )
    '''
    print("done training Weights pruned ", torch.where(model.mlp[0].weight.detach().cpu() == 0,1,0).sum()/(1024*2048))


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", default="models/snli/6.pth")
    parser.add_argument("--exp_dir", default="models/snli/")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
