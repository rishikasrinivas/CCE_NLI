import models
import torch
import util
import tqdm as tqdm
from contextlib import nullcontext
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import numpy as np
import fileio
import os
import settings
from tqdm import tqdm
from data.snli import SNLI, pad_collate
from collections import defaultdict
import os
def create_dataloaders(max_data):
    if not ('train_loader.pth' in os.listdir("models/DataLoaders/") and 'val_loader.pth' in os.listdir("models/DataLoaders/") and 'test_loader.pth' in os.listdir("models/DataLoaders/")):
        train = SNLI("data/snli_1.0", "train", max_data=max_data)
        train_loader = DataLoader(
            train,
            batch_size=100,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(train_loader, 'train_loader.pth')
        
        val = SNLI("data/snli_1.0","dev",max_data=max_data,vocab=(train.stoi, train.itos),unknowns=False)
        val_loader = DataLoader(
            val, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        
        )
        torch.save(val_loader, 'val_loader.pth')
        
        test = SNLI("data/snli_1.0", "test", max_data=max_data, vocab=(train.stoi, train.itos), unknowns=True)
        test_loader = DataLoader(
            test,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(test_loader, 'test_loader.pth')
    else:
        train_loader=torch.load("models/DataLoaders/train_loader.pth")
        val_loader=torch.load("models/DataLoaders/val_loader.pth")
        test_loader=torch.load("models/DataLoaders/test_loader.pth")
        
        
    
    dataloaders = {
        'train': train_loader,
        'val':val_loader,
        'test': test_loader
    }
    return train_loader.dataset, val_loader.dataset,test_loader.dataset, dataloaders


def run(split, epoch, model, optimizer, criterion, dataloader, device='cuda'):
    training = split == "train"
    if training:
        ctx = nullcontext
        model.train()
    else:
        ctx = torch.no_grad
        model.eval()

    ranger = tqdm(dataloader, desc=f"{split} epoch {epoch}")
    model.prune_mask=model.prune_mask.cuda()
    print("Pruned mask pruned ", torch.where(model.prune_mask==0,1,0).sum()/(2048*1024))

    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()
    for (s1, s1len, s2, s2len, targets) in ranger:
        if device == 'cuda':
            s1 = s1.cuda()
            s1len = s1len.cuda()
            s2 = s2.cuda()
            s2len = s2len.cuda()
            targets = targets.cuda()

    

        batch_size = targets.shape[0]
        
        with ctx():
            logits = model(s1, s1len, s2, s2len)
            loss = criterion(logits, targets)

        if training:
            optimizer.zero_grad()
            loss.backward()
            model.mlp[0].weight.grad *= model.prune_mask
            
            optimizer.step()
        preds = logits.argmax(1)
        acc = (preds == targets).float().mean()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)

        ranger.set_description(
            f"{split} epoch {epoch} loss {loss_meter.avg:.3f} acc {acc_meter.avg:.3f}"
        )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

def finetune_pruned_model(model,optimizer,criterion, train, val, dataloaders, finetune_epochs, prune_metrics_dir,device):
    metrics = defaultdict(list)
    metrics["best_val_acc"]=0.0
    metrics["best_val_epoch"] = 0
    metrics["best_val_loss"] = np.inf
    metrics[f"train_loss"]=[]
    metrics[f"train_acc"]=[]
    metrics[f"val_loss"]=[]
    metrics[f"val_acc"]=[]
    fw=model.mlp[0].weight.detach().cpu().numpy()
    print("In function finetune final weights pruned ", torch.where(torch.tensor(fw)==0,1,0).sum()/(2048*1024))
    for epoch in range(finetune_epochs):
        
        train_metrics = run(
            "train", epoch, model, optimizer, criterion, dataloaders['train'],device
        )

        val_metrics = run(
            "val", epoch, model, optimizer, criterion, dataloaders['val'] ,device
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
        if epoch % 1 == 0:

            util.save_checkpoint(
                serialize(model, train), False, prune_metrics_dir, filename=f"LotTick{epoch}.pth"
            )
        
    path_to_ckpt = os.path.join(prune_metrics_dir, f"model_best.pth")
    print(f"Loading best weights from {path_to_ckpt}")
    model.load_state_dict(torch.load(path_to_ckpt)['state_dict'])
    fw=model.mlp[0].weight.detach().cpu().numpy()
    print("In function finetune final weights pruned ", torch.where(torch.tensor(fw)==0,1,0).sum()/(2048*1024))
    
    return model, model.mlp[0].weight.detach().cpu().numpy(), torch.load(path_to_ckpt)


def build_model(vocab_size, model_type, embedding_dim=300, hidden_dim=512):
    """
    Build a bowman-style SNLI model
    """
    enc = models.TextEncoder(
        vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim
    )
    if model_type == "minimal":
        model = models.EntailmentClassifier(enc)
    else:
        model = models.BowmanEntailmentClassifier(enc)
    return model

def load_model(max_data, train, ckpt=None, device='cuda'):
    model = build_model(vocab_size=len(train.stoi), model_type='bowman', embedding_dim=300, hidden_dim=512)
    if ckpt:
        if type(ckpt) == str:
            ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt["state_dict"])
    else:
        util.save_checkpoint(
                serialize(model, train), False, settings.PRUNE_METRICS_DIR, filename=f"random_inits.pth"
        )
    
    return model


def serialize(model, dataset):
    if model.check_pruned():
        prune.remove(model.mlp[:-1][0], name="weight")
    return {
        "state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }

def run_eval(model, val_loader):
    model.eval()
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
    return np.round(acc,3)