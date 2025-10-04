import models
import torch
import util
import tqdm as tqdm
from contextlib import nullcontext
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import numpy as np
#!pip install llm2vec #add to donwloads
import os
import settings
from tqdm import tqdm
from data.snli import SNLI, pad_collate
from collections import defaultdict
import os,fileio
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast,GradScaler

def create_dataloaders(max_data, debug=False):
    root_dir=f"DataLoaders/"
    os.makedirs(root_dir, exist_ok=True)
    if debug or not ('train_dataset.pth' in os.listdir(root_dir) and 'val_dataset.pth' in os.listdir(root_dir)):
        print(f"No data saved. Loading dataloader")
        train = SNLI("data/snli_1.0", "train", max_data=max_data)
        train_loader = DataLoader(
            train,
            batch_size=settings.BATCH_SIZE,
            shuffle=True,
            pin_memory=False,
            num_workers=4,
            collate_fn=pad_collate,
        )
        torch.save(train_loader.dataset, f'{root_dir}/train_dataset.pth')
        
        val = SNLI("data/snli_1.0","dev",max_data=max_data,vocab=(train.stoi, train.itos),unknowns=False)
        val_loader = DataLoader(
            val, 
            batch_size=settings.BATCH_SIZE, 
            shuffle=False,
            pin_memory=True, 
            num_workers=4, 
            collate_fn=pad_collate
        
        )
        torch.save(val_loader.dataset, f'{root_dir}/val_dataset.pth')
        
       
    else:
        train_dataset = torch.load(f'{root_dir}/train_dataset.pth')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=settings.BATCH_SIZE, 
            shuffle=True, 
            pin_memory=False,
            num_workers=4,
            collate_fn=pad_collate
        )
        
        val_dataset = torch.load(f'{root_dir}/val_dataset.pth')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=settings.BATCH_SIZE, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=4, 
            collate_fn=pad_collate
        )
      
        
        
    
    dataloaders = {
        'train': train_loader,
        'val':val_loader,
    }
    return train_loader.dataset, val_loader.dataset,dataloaders


#learning rate diffs

def run(split, epoch, model,model_type, optimizer, criterion, dataloader, total_epochs, device='cuda'):
    torch.cuda.empty_cache()
    training = split == "train"
    if training:
        ctx = autocast
        model.train()
    else:
        ctx = torch.no_grad
        model.eval()
        
    scaler = GradScaler()
    ranger = tqdm(dataloader[split], desc=f"{split} epoch {epoch}")
    scheduler=None
    
    if model_type in ['bert', 'llama']:
        total_steps = len(dataloader['train']) *6
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    

    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()

    for (s1, s1len, s2, s2len, targets) in ranger:
        if torch.cuda.is_available():
            s1 = s1.cuda()
            s1len = s1len.cuda()
            s2 = s2.cuda()
            s2len = s2len.cuda()
            targets = targets.cuda()

    

        batch_size = targets.shape[0]
        
        with ctx(): #llama half precis
            logits = model(s1, s1len, s2, s2len)
        
          
            loss = criterion(logits.float(), targets)
 
        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            #loss.backward()
            for layer in model.layers:
                if layer.weights.grad is not None:
                    #assert not torch.equal(torch.ones(layer.pruning_mask.shape), layer.pruning_mask), layer.name
                    layer.weights.grad *= layer.pruning_mask.to(device)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            scaler.step(optimizer)
            scaler.update()
            if scheduler: scheduler.step()
            #optimizer.step()
            
                
        preds = logits.argmax(1)
        acc = (preds == targets).float().mean()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)

        ranger.set_description(
            f"{split} epoch {epoch} loss {loss_meter.avg:.3f} acc {acc_meter.avg:.3f}"
        )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

def finetune_pruned_model(model,model_type, optimizer,criterion, train, val, dataloaders, finetune_epochs, prune_metrics_dir,baseline_acc, device):
    metrics = {"best_val_acc": 0.0, "best_val_epoch": 0, "best_val_loss": np.inf, "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    epoch = 0
    accuracy=0
    while accuracy <= baseline_acc:
        train_metrics = run(
            "train", epoch, model, model_type, optimizer, criterion, dataloaders, finetune_epochs, device
        )

        val_metrics = run(
            "val", epoch, model, model_type, optimizer, criterion, dataloaders, finetune_epochs, device
        )

        for name, val in train_metrics.items():
            metrics[f"train_{name}"].append(val)

        for name, val in val_metrics.items():
            metrics[f"val_{name}"].append(val)

        is_best = val_metrics["acc"] > metrics["best_val_acc"]
        accuracy = metrics["best_val_acc"]

        if is_best:
            metrics["best_val_epoch"] = epoch
            metrics["best_val_acc"] = val_metrics["acc"]
            metrics["best_val_loss"] = val_metrics["loss"]
            fileio.log_to_csv(os.path.join(prune_metrics_dir,"pruned_status.csv"), [epoch, val_metrics["acc"], val_metrics["loss"]], ["EPOCH", "ACCURACY", "LOSS"])
        
       
        util.save_metrics(metrics, prune_metrics_dir)
        util.save_checkpoint(serialize(model, model_type, train), is_best, prune_metrics_dir)
        epoch += 1
        
        
    path_to_ckpt = os.path.join(prune_metrics_dir, f"model_best.pth")
    print(f"Loading best weights from {path_to_ckpt}")
    model.load_state_dict(torch.load(path_to_ckpt)['state_dict'])
    
    return model


def build_model(vocab_size, model_type, vocab, pretrained=True, embedding_dim=300, hidden_dim=512, device='cuda'):
    """
    Build a bowman-style SNLI model
    """
    
    if model_type=='bert':
        model=models.BertEntailmentClassifier(vocab=vocab, pretrained=pretrained, device=device)
    elif model_type == 'bowman':
        enc = models.TextEncoder(
            vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim
        )
        model = models.BowmanEntailmentClassifier(enc, device)
    elif model_type == 'llama':
            if vocab is None:
                raise Exception('Llama model requires passing the datasets vocab field')
            model = models.LLAMAEntailmentClassifier(vocab=vocab,freeze_encoder=True)
    return model

def load_model(max_data, model_type, train, ckpt=None, use_pretrained_weights = True, device='cuda', i=0):
    model = build_model(vocab_size=len(train.stoi), model_type=model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, pretrained=use_pretrained_weights, embedding_dim=300, hidden_dim=512, device=device)
    
        
        
    if ckpt:
        if type(ckpt) == str:
            print(f"Loading from {ckpt} with model {model_type}")
            ckpt_ = torch.load(ckpt, map_location = torch.device(device))
        model.load_state_dict(ckpt_["state_dict"])
    else:
        if use_pretrained_weights:
            pretrained_dir= os.path.join(model_type.upper(), "models", "pretrained")
            os.makedirs(pretrained_dir, exist_ok=True)
            save_to_dir = pretrained_dir
            filename = f"{model_type}_{i}_pretrained_inits.pth"
        else:
            untrained_dir = os.path.join(model_type.upper(), "models", "untrained")
            os.makedirs(untrained_dir, exist_ok=True)
            save_to_dir = untrained_dir
            filename = f"{model_type}_untrained_inits.pth"
            
        print("train itos = ", len(train.stoi))
        
        util.save_checkpoint(
                serialize(model, model_type, train), False, save_to_dir, filename
        )
        ckpt = os.path.join(save_to_dir, filename)
        
            
        
    return model.to(device), ckpt


def serialize(model,model_type, dataset):
    if model_type == ['llama','bert']:
        return {
            "encoder_name": model.encoder_name, 
            "state_dict": model.state_dict(), 
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        }
    return {
        "state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }

def run_eval(model, val_loader):
    model.cuda()
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