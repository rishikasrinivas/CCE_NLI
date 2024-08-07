import models
import torch
import util
import tqdm as tqdm
from contextlib import nullcontext
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import numpy as np
def run(split, epoch, model, optimizer, criterion, dataloaders, args, device='cuda'):
    training = split == "train"
    if training:
        ctx = nullcontext
        model.train()
    else:
        ctx = torch.no_grad
        model.eval()

    ranger = tqdm(dataloaders[split], desc=f"{split} epoch {epoch}")

    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()
    for (s1, s1len, s2, s2len, targets) in ranger:

        if device == 'cuda' or args.cuda:
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
            optimizer.step()

        preds = logits.argmax(1)
        acc = (preds == targets).float().mean()

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)

        ranger.set_description(
            f"{split} epoch {epoch} loss {loss_meter.avg:.3f} acc {acc_meter.avg:.3f}"
        )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


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


def serialize(model, dataset):
    if model.check_pruned():
        prune.remove(model.mlp[:-1][0], name="weight")
    return {
        "state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }

def run_eval(model, val_loader):
    
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
    return acc