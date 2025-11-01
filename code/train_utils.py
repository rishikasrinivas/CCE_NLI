import sys
sys.path.append("code/models/")
import models.cofi_models as cofi_models
import models.nli_models as nli_models
import cofi.utils.cofi_utils as cofi_utils
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
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


def collate_as_dict(batch):
    """
    We don't sort here to take advantage of enforce_sorted=False since we'd
    have to sort separately for both s1 and s2
    """

    s1, s1len, s2, s2len, label = zip(*batch)
    label = torch.tensor(label)

    s1_pad = pad_sequence(s1, padding_value=1) #takes 10,000x longest sent length and flips to longest*10,000 which is then batched
    s1len = torch.tensor(s1len)

    s2_pad = pad_sequence(s2, padding_value=1)
    s2len = torch.tensor(s2len)
    

    return {'s1': s1_pad, 's1len': s1len, 's2':s2_pad, 's2len': s2len, 'labels':label}

def create_dataloaders(max_data, model_type, pruning_method,  debug=False):
    """
    Creates and caches dataloaders.
    - Level 1 Cache: Caches the SNLI dataset objects after reading from text.
    - Level 2 Cache: For transformer models, caches the fully converted and tokenized batches.
    """

    root_dir = "../DataLoaders"
    os.makedirs(root_dir, exist_ok=True)

    # --- PART 1: Load or Create the Base SNLI Datasets ---
    # This is the first level of caching. It avoids re-reading the raw .txt files.
    base_train_path = f'{root_dir}/train_dataset.pth'
    base_val_path = f'{root_dir}/val_dataset.pth'
    
    #EDIT: Updated conditions to download specified amounts of data based on cache
    try:
        print("✅ Loading base SNLI dataset from cache...")
        if debug:
            train_dataset = torch.load(f'{root_dir}/train_dataset_debug{max_data}.pth')
            val_dataset = torch.load(f'{root_dir}/val_dataset_debug{max_data}.pth')
        else:
            train_dataset = torch.load(f'{root_dir}/train_dataset.pth')
            val_dataset = torch.load(f'{root_dir}/val_dataset.pth')
    except:
        print("⚠️ Base SNLI dataset cache not found. Creating from text files...")
        if debug:
            print(f"collecting {max_data}")
            train_dataset = SNLI("data/snli_1.0", "train", max_data=max_data)
            val_dataset = SNLI("data/snli_1.0","dev",max_data=max_data,vocab=(train_dataset.stoi, train_dataset.itos),unknowns=False)
            
            torch.save(train_dataset, f'{root_dir}/train_dataset_debug{max_data}.pth')
            torch.save(val_dataset, f'{root_dir}/val_dataset_debug{max_data}.pth')
           
        else:
            train_dataset = SNLI("data/snli_1.0", "train", max_data=None)
            val_dataset = SNLI("data/snli_1.0","dev",max_data=10000,vocab=(train_dataset.stoi, train_dataset.itos),unknowns=False)
            torch.save(train_dataset, f'{root_dir}/train_dataset.pth')
            torch.save(val_dataset, f'{root_dir}/val_dataset.pth')
    
    # --- PART 2: Branch logic based on model type ---
    if model_type in ['bert', 'llama']:
        # Define paths for the second level of caching (the converted batches)
        if pruning_method=='cofi':
            filepath='cofi'
        else:
            filepath='unstructured'
        train_cache_path = f'{root_dir}/converted_batches_train_transformers_{filepath}.pth' #add _pruningalg
        val_cache_path = f'{root_dir}/converted_batches_val_transformers_{filepath}.pth' #add _pruningalg

        if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
            print(f"✅ Loading pre-converted {model_type} batches from cache...{train_cache_path}")
            converted_train_batches = torch.load(train_cache_path)
            converted_val_batches = torch.load(val_cache_path)
        else:
            print(f"⚠️ Converted batches cache not found. Performing one-time conversion for {model_type}...")
            
            model_name = "bert-base-uncased" if model_type == 'bert' else "knowledgator/Llama-encoder-1.0B"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            itos = train_dataset.itos
            
            # Use a temporary loader to create batches from the base dataset
            temp_train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=pad_collate)
            converted_train_batches = []
            for batch in tqdm(temp_train_loader, desc="Converting train batches"):
                s1_pad, _, s2_pad, _, targets = batch
                s1_indices, s2_indices = s1_pad.cpu().numpy().T, s2_pad.cpu().numpy().T
                s1_sentences = [" ".join([itos.get(idx, "") for idx in row if idx not in (0, 1)]) for row in s1_indices]
                s2_sentences = [" ".join([itos.get(idx, "") for idx in row if idx not in (0, 1)]) for row in s2_indices]
                s1_tokenized = tokenizer(s1_sentences, return_tensors="pt", padding=True, truncation=True)
                s2_tokenized = tokenizer(s2_sentences, return_tensors="pt", padding=True, truncation=True)
                if pruning_method != 'cofi':
                    converted_train_batches.append((s1_tokenized, s2_tokenized, targets)) #if pruning alg is not cofi do this else make the dict
                else:
                    result= {
                        "pre_input_ids": s1_tokenized["input_ids"].cpu(),
                        "pre_attention_mask": s1_tokenized["attention_mask"].cpu(),
                        "hyp_input_ids": s2_tokenized["input_ids"].cpu(),
                        "hyp_attention_mask": s2_tokenized["attention_mask"].cpu(),
                        "labels":torch.tensor([l for l in targets])
                    }
                
                    converted_train_batches.append(result)

            temp_val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, num_workers=4, collate_fn=pad_collate)
            converted_val_batches = []
            for batch in tqdm(temp_val_loader, desc="Converting val batches"):
                s1_pad, _, s2_pad, _, targets = batch
                s1_indices, s2_indices = s1_pad.cpu().numpy().T, s2_pad.cpu().numpy().T
                s1_sentences = [" ".join([itos.get(idx, "") for idx in row if idx not in (0, 1)]) for row in s1_indices]
                s2_sentences = [" ".join([itos.get(idx, "") for idx in row if idx not in (0, 1)]) for row in s2_indices]
                s1_tokenized = tokenizer(s1_sentences, return_tensors="pt", padding=True, truncation=True)
                s2_tokenized = tokenizer(s2_sentences, return_tensors="pt", padding=True, truncation=True)
                if pruning_method != 'cofi':
                    converted_val_batches.append((s1_tokenized, s2_tokenized, targets)) #if pruning alg is not cofi do this else make the dict
                else:
                    result = {
                        "pre_input_ids": s1_tokenized["input_ids"].cpu(),
                        "pre_attention_mask": s1_tokenized["attention_mask"].cpu(),
                        "hyp_input_ids": s2_tokenized["input_ids"].cpu(),
                        "hyp_attention_mask": s2_tokenized["attention_mask"].cpu(),
                        "labels": torch.tensor([l for l in targets])
                    }
                    converted_val_batches.append(result)


            print(f"💾 Saving converted batches to cache for future runs...")
            torch.save(converted_train_batches, train_cache_path)
            torch.save(converted_val_batches, val_cache_path)
        
        # Create final DataLoaders from the list of converted batches
        train_loader = DataLoader(converted_train_batches, shuffle=True, batch_size=1, collate_fn=lambda x: x[0],drop_last=True)
        val_loader = DataLoader(converted_val_batches, shuffle=False, batch_size=1, collate_fn=lambda x: x[0],drop_last=True)

    else: # This path is for the 'bowman' model
        print("✅ Using original dataloaders for bowman model.")
        #collate fn changes if not cofi vs if cofi
        collate_fn =  collate_as_dict if pruning_method=='cofi' else pad_collate 
        train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn,drop_last=True)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    return train_dataset, val_dataset, dataloaders
#learning rate diffs


def run(split, epoch, model, model_type, pruning_method, optimizer, criterion, dataloaders, total_epochs, device='cuda'):
    from contextlib import nullcontext # Make sure this is imported
    torch.cuda.empty_cache()
    training = split == "train"
    
    if training:
        # CORRECTED: Disable autocast for this test
        ctx = autocast
        model.train()
    else:
        ctx = torch.no_grad
        model.eval()
    
    # CORRECTED: Disable the GradScaler for this test
    # scaler = GradScaler(enabled=(training and torch.cuda.is_available()))
    
    ranger = tqdm(dataloaders[split], desc=f"{split} epoch {epoch}")
    scheduler = None
    if model_type in ['bert', 'llama'] and training:
        num_training_steps = len(dataloaders['train']) * total_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()
    for batch in ranger:
        if pruning_method != 'cofi':
            if model_type in ['bert', 'llama']: 
                s1_batch, s2_batch, targets = batch
                if torch.cuda.is_available():
                    s1_batch = {k: v.to(device) for k, v in s1_batch.items()}
                    s2_batch = {k: v.to(device) for k, v in s2_batch.items()}
                    targets = targets.to(device)
                batch_size = targets.shape[0]
                with ctx():
                    logits = model(s1_batch, s2_batch)
                    loss = criterion(logits, targets)
            else:
                s1, s1len, s2, s2len, targets = batch
                if torch.cuda.is_available():
                    s1, s1len = s1.to(device), s1len.to(device)
                    s2, s2len = s2.to(device), s2len.to(device)
                    targets = targets.to(device)
                batch_size = targets.shape[0]
                with ctx():
                    logits = model(s1, s1len, s2, s2len)
                    loss = criterion(logits, targets)
        else:
           
            if torch.cuda.is_available():
                batch = {k: v.to(device) for k, v in batch.items()}
                targets = batch['labels']
            batch_size = targets.shape[0]
            with ctx():
                output = model(**batch)
                loss = output.loss
            logits=output[1][2]
        if training:
            optimizer.zero_grad()
            
            # CORRECTED: Use standard loss.backward()
            loss.backward()
            
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    if hasattr(layer.weights, 'grad') and layer.weights.grad is not None:
                        layer.weights.grad *= layer.pruning_mask.to(device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # CORRECTED: Use standard optimizer.step()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                
        preds = logits.argmax(1)
        acc = (preds == targets).float().mean()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        ranger.set_description(f"{split} epoch {epoch} loss {loss_meter.avg:.3f} acc {acc_meter.avg:.3f}")
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}
def finetune_pruned_model(model, model_type, pruning_method, optimizer, criterion, dataloaders, finetune_epochs, prune_metrics_dir, baseline_acc=-1.0, device='cuda'):
    """
    Finetunes a model for a fixed number of epochs and saves the best-performing one.
    """
    metrics = {"best_val_acc": 0.0, "best_val_epoch": 0, "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    #EDIT: finetune until accuracy surpasses that of original model
    epoch = 0
    acc = 0.0
    while (baseline_acc != -1.0 and acc < baseline_acc) or (baseline_acc == -1.0 and epoch < finetune_epochs):
        train_metrics = run("train", epoch, model, model_type, pruning_method, optimizer, criterion, dataloaders, finetune_epochs, device)
        val_metrics = run("val", epoch, model, model_type, pruning_method, optimizer, criterion, dataloaders, finetune_epochs, device)
        
        metrics["train_loss"].append(train_metrics["loss"])
        metrics["train_acc"].append(train_metrics["acc"])
        metrics["val_loss"].append(val_metrics["loss"])
        metrics["val_acc"].append(val_metrics["acc"])

        is_best = val_metrics["acc"] > metrics["best_val_acc"]
            
        if is_best:
            metrics["best_val_acc"] = val_metrics["acc"]
            metrics["best_val_epoch"] = epoch
            util.save_metrics(metrics, prune_metrics_dir)
            util.save_checkpoint(serialize(model, model_type, dataloaders['train']), is_best=True, exp_dir=prune_metrics_dir)
        acc = metrics["best_val_acc"]
        epoch += 1

    # Load the best performing model
    best_model_path = os.path.join(prune_metrics_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best weights from epoch {metrics['best_val_epoch']} with accuracy {metrics['best_val_acc']:.3f}")
        model.load_state_dict(torch.load(best_model_path))

    return model, metrics["best_val_acc"]


def build_model(model_type, vocab, vocab_size=None, pretrained=True, embedding_dim=300, hidden_dim=512, device='cuda', is_cofi=False):
    """
    Builds the specified model. `vocab_size` is only used for the bowman model.
    """
    if is_cofi:
        if model_type=='bowman':
            tokenizer = TextEncoder(len(vocab['stoi']))
            model = cofi_models.modeling_bowman.CoFiBowmanEntailmentClassifier(tokenizer, 'cuda')

        elif model_type=='bert':
            from cofi_models import modeling_bert
            model = modeling_bert.CoFiBertForSequenceClassification
            
    
    else: 
        if model_type == 'bert':
            # CORRECTED: Removed the 'vocab' argument
            model = nli_models.BertEntailmentClassifier(vocab, pretrained=pretrained, device=device)
        elif model_type == 'llama':
            # CORRECTED: Removed the 'vocab' argument
            model = nli_models.LLAMAEntailmentClassifier( freeze_encoder=True, device=device)
        elif model_type == 'bowman':
            # This path remains the same
            enc = nli_models.TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
            model = nli_models.BowmanEntailmentClassifier(enc, device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    return model

def load_model(model_type, train, ckpt=None, use_pretrained_weights=True, pruning_method='', device='cuda', i=0, zs=None):
    """
    Loads or initializes a model.
    """
    # CORRECTED: Call the updated build_model function
    cofi = pruning_method == 'cofi'
    model = build_model(
        vocab= {'itos':train.itos, 'stoi': train.stoi},
        model_type=model_type,
        vocab_size=len(train.stoi), # For bowman
        pretrained=use_pretrained_weights, # For bert
        device=device,
        is_cofi=cofi
    )
    
    if ckpt: #and not cofi
        if zs:
            print(f"Loading zs")
            print(ckpt)
            model = cofi_utils.load_model_with_zs(ckpt, model, zs=zs)
        else:
            print(f"Loading from checkpoint (no zs): {ckpt}")
            ckpt_ = torch.load(ckpt, map_location=torch.device(device))
            model.load_state_dict(ckpt_["state_dict"])
    elif not ckpt:
        # This logic for saving initial weights is fine
        print("Loading pretrained weights")
        save_dir_type = "pretrained" if use_pretrained_weights else "untrained"
        save_dir = os.path.join(model_type.upper(), "models", save_dir_type)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{model_type}_{i}_{save_dir_type}_inits.pth"

        util.save_checkpoint(
            serialize(model, model_type, train), False, save_dir, filename
        )
        ckpt = os.path.join(save_dir, filename)
        if cofi:
            print(f"Loaded new CoFi instance (pretrained)")
    
    
        
        
    return model.to(device), ckpt


def serialize(model, model_type, dataset):
    # CORRECTED: The condition now correctly checks if model_type is in the list
    if model_type in ['llama', 'bert']:
        return {
            "encoder_name": model.model_name, 
            "state_dict": model.state_dict(),
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            # CORRECTED: stoi and itos are no longer needed for transformer models
        }
    # For bowman, we still need the vocab
    return {
        "state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }

def run_eval(model, val_loader, model_type, pruning_method):
    model.cuda()
    model.eval()
    all_preds = []
    all_targets = []

    # CORRECTED: Added conditional logic for batch handling
    for batch in val_loader:
        if pruning_method == 'cofi':
            if torch.cuda.is_available():

                batch = {k: v.to('cuda') for k, v in batch.items()}
                targets = batch['labels']


            batch_size = targets.shape[0]

            with torch.no_grad():
                logits = model(**batch)

            preds = logits[1][2].argmax(1)
        
        else:
            if model_type in ['bert', 'llama']:
                s1_batch, s2_batch, targets = batch
                if settings.CUDA:
                    s1_batch = {k: v.cuda() for k, v in s1_batch.items()}
                    s2_batch = {k: v.cuda() for k, v in s2_batch.items()}

                with torch.no_grad():
                    logits = model(s1_batch, s2_batch)
            else: # Bowman path
                s1, s1len, s2, s2len, targets = batch
                if settings.CUDA:
                    s1, s1len = s1.cuda(), s1len.cuda()
                    s2, s2len = s2.cuda(), s2len.cuda()

                with torch.no_grad():
                    logits = model(s1, s1len, s2, s2len)
        
            preds = logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_targets = np.concatenate(all_targets, 0)
    acc = (all_preds == all_targets).mean()
    return np.round(acc, 3)

