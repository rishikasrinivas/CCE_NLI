import sys
sys.path.append("code/models/")
import models.cofi_models as cofi_models
import models.nli_models as nli_models
import cofi.utils.cofi_utils as cofi_utils
import torch
import util
import tqdm as tqdm
from contextlib import nullcontext
from torch.utils.data import DataLoader, Subset
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

import json
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

import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import glob


def create_dataloaders(max_data, model_type, pruning_method, debug=False):
    """
    Creates and caches dataloaders safely for large datasets.
    Uses "save-as-you-go" + lazy loading to prevent RAM crashes.
    """
    root_dir = "../DataLoaders"
    os.makedirs(root_dir, exist_ok=True)

    # --- PART 1: Load or create base datasets ---
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
            train_dataset = SNLI("data/snli_1.0", "train", max_data=max_data)
            val_dataset = SNLI("data/snli_1.0", "dev", max_data=None,
                               vocab=(train_dataset.stoi, train_dataset.itos),
                               unknowns=False)
            torch.save(train_dataset, f'{root_dir}/train_dataset_debug{max_data}.pth')
            torch.save(val_dataset, f'{root_dir}/val_dataset_debug{max_data}.pth')
        else:
            train_dataset = SNLI("data/snli_1.0", "train", max_data=None)
            val_dataset = SNLI("data/snli_1.0", "dev", max_data=10000,
                               vocab=(train_dataset.stoi, train_dataset.itos),
                               unknowns=False)
            torch.save(train_dataset, f'{root_dir}/train_dataset.pth')
            torch.save(val_dataset, f'{root_dir}/val_dataset.pth')

    # --- PART 2: Transformer-specific batch conversion ---
    if model_type in ['bert', 'llama']:
        print(f"Pruning method: {pruning_method}")
        filepath = 'cofi' if pruning_method == 'CoFi' else 'unstructured'

        train_batch_dir = os.path.join(root_dir, f"train_batches_{model_type}_{filepath}_{max_data}")
        val_batch_dir   = os.path.join(root_dir, f"val_batches_{model_type}_{filepath}_{max_data}")
        os.makedirs(train_batch_dir, exist_ok=True)
        os.makedirs(val_batch_dir, exist_ok=True)

        tokenizer_name = "bert-base-uncased" if model_type == 'bert' else "knowledgator/Llama-encoder-1.0B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        itos = train_dataset.itos
        

        if os.path.exists(os.path.join(root_dir, f"train_batches_{model_type}_{max_data}_all.pth")) and os.path.exists(os.path.join(root_dir, f"val_batches_{filepath}_{max_data}_all.pth")): 
            print(f"Loading combined tokenizations from {root_dir}")
            combined_tokens_train = torch.load(os.path.join(root_dir, f"train_batches_{model_type}_{max_data}_all.pth"))

            train_loader = DataLoader(combined_tokens_train,
                                      batch_size=1,  # already a batch of size 32
                                      shuffle=True,
                                      collate_fn=lambda x: x[0])
        

            combined_tokens_val = torch.load(os.path.join(root_dir, f"val_batches_{model_type}_{max_data}_all.pth"))

            val_loader = DataLoader(combined_tokens_val,
                                    batch_size=1,
                                    shuffle=False,
                                collate_fn=lambda x: x[0])
        else:

            # --- Convert & save train batches ---
            if os.path.exists(train_batch_dir) and len(os.listdir(train_batch_dir)) > 0:
                print(f"✅ Tokenized train batches already exist in {train_batch_dir}, skipping conversion.")

            else:
                temp_train_loader = DataLoader(train_dataset,
                                               batch_size=settings.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=pad_collate)

                print("⚡ Converting and saving train batches...")
                batch_data=[]
                for idx, batch in enumerate(tqdm(temp_train_loader)):
                    s1_pad, _, s2_pad, _, targets = batch
                    s1_indices, s2_indices = s1_pad.cpu().numpy().T, s2_pad.cpu().numpy().T
                    s1_sentences = [" ".join([itos.get(i, "") for i in row if i not in (0, 1)]) for row in s1_indices]
                    s2_sentences = [" ".join([itos.get(i, "") for i in row if i not in (0, 1)]) for row in s2_indices]
                    s1_tokenized = tokenizer(s1_sentences, return_tensors="pt", padding=True, truncation=True)
                    s2_tokenized = tokenizer(s2_sentences, return_tensors="pt", padding=True, truncation=True)

                    if pruning_method != 'CoFi':
                        batch_data.append((s1_tokenized, s2_tokenized, targets))
                    else:
                        batch_data.append({
                            "pre_input_ids": s1_tokenized["input_ids"].cpu(),
                            "pre_attention_mask": s1_tokenized["attention_mask"].cpu(),
                            "hyp_input_ids": s2_tokenized["input_ids"].cpu(),
                            "hyp_attention_mask": s2_tokenized["attention_mask"].cpu(),
                            "labels": torch.tensor([l for l in targets])
                        })

                    if (idx+1) % (len(temp_train_loader)%6000)  == 0: 
                        print(f"Saving until {idx}, {len(batch_data)}")
                        try: 
                            torch.save(batch_data, os.path.join(train_batch_dir, f"batch_{idx:04d}.pth"))
                            print(f"saved to", os.path.join(train_batch_dir, f"batch_{idx:04d}.pth"))
                        except Exception as e:
                            print(f"Error: {e}")
                        del batch_data, s1_tokenized, s2_tokenized
                        batch_data=[]

                if batch_data:
                    torch.save(batch_data, os.path.join(train_batch_dir, f"batch_{idx:04d}.pth"))
                    del batch_data, s1_tokenized, s2_tokenized
            if os.path.exists(val_batch_dir) and len(os.listdir(val_batch_dir)) > 0:
                print(f"✅ Tokenized val batches already exist in {val_batch_dir}, skipping conversion.")
            # --- Convert & save val batches ---
            else:
                temp_val_loader = DataLoader(val_dataset,
                                             batch_size=settings.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=pad_collate)
                print("⚡ Converting and saving val batches...")
                batch_data = []
                for idx, batch in enumerate(tqdm(temp_val_loader)):
                    s1_pad, _, s2_pad, _, targets = batch
                    s1_indices, s2_indices = s1_pad.cpu().numpy().T, s2_pad.cpu().numpy().T
                    s1_sentences = [" ".join([itos.get(i, "") for i in row if i not in (0, 1)]) for row in s1_indices]
                    s2_sentences = [" ".join([itos.get(i, "") for i in row if i not in (0, 1)]) for row in s2_indices]
                    s1_tokenized = tokenizer(s1_sentences, return_tensors="pt", padding=True, truncation=True)
                    s2_tokenized = tokenizer(s2_sentences, return_tensors="pt", padding=True, truncation=True)

                    if pruning_method != 'CoFi':
                        batch_data.append((s1_tokenized, s2_tokenized, targets))
                    else:
                        batch_data.append( {
                            "pre_input_ids": s1_tokenized["input_ids"].cpu(),
                            "pre_attention_mask": s1_tokenized["attention_mask"].cpu(),
                            "hyp_input_ids": s2_tokenized["input_ids"].cpu(),
                            "hyp_attention_mask": s2_tokenized["attention_mask"].cpu(),
                            "labels": torch.tensor([l for l in targets])
                        })
                    if (idx+1) % (len(temp_val_loader)%6000) == 0:

                        torch.save(batch_data, os.path.join(val_batch_dir, f"batch_{idx:04d}.pth"))
                        del batch_data, s1_tokenized, s2_tokenized
                        batch_data=[]

                if batch_data:
                    print("Saving batchdata")
                    torch.save(batch_data, os.path.join(val_batch_dir, f"batch_{idx:04d}.pth"))
            # --- PART 3: Lazy loading ---
       
    
            print("LOADING DATA")
            train_all_batches = []

            files = sorted(os.listdir(train_batch_dir))

            for f in files:
               
                batches = torch.load(os.path.join(train_batch_dir,f))
                train_all_batches.extend(batches)  # flatten into a single list

            # save as one big file

            torch.save(train_all_batches, os.path.join(root_dir, f"train_batches_{filepath}_{max_data}_all.pth"))
            print(f"✅ Saved {len(train_all_batches)} batches into train_batches_{filepath}_{max_data}_all.pth")

            train_loader = DataLoader(train_all_batches,
                                      batch_size=1,  # already a batch of size 32
                                      shuffle=True,
                                      collate_fn=lambda x: x[0])

            val_all_batches = []

            files = sorted(os.listdir(val_batch_dir))

            for f in files:
                batches = torch.load(os.path.join(val_batch_dir,f))
                val_all_batches.extend(batches)  # flatten into a single list


            torch.save(val_all_batches, os.path.join(root_dir, f"val_batches_{filepath}_{max_data}_all.pth"))
            print(f"✅ Saved {len(val_all_batches)} batches into val_batches_{filepath}_{max_data}_all.pth")


            val_loader = DataLoader(val_all_batches,
                                    batch_size=1,
                                    shuffle=False,
                                collate_fn=lambda x: x[0])

    else:
        # --- Bowman / small models ---
        collate_fn = collate_as_dict if pruning_method=='CoFi' else pad_collate
        train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True,
                                  num_workers=4, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False,
                                num_workers=4, collate_fn=collate_fn, drop_last=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    return train_dataset, val_dataset, dataloaders



def run(split, epoch, model, model_type, pruning_method, optimizer, criterion, dataloaders, total_epochs, device='cuda'):
    from contextlib import nullcontext # Make sure this is imported
    torch.cuda.empty_cache()
    training = split == "train"
    model.to(device)
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
       
        if pruning_method != 'CoFi':
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
    while (baseline_acc != -1.0 and acc < baseline_acc) or (baseline_acc == -1.0 and  epoch < finetune_epochs):
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
            util.save_checkpoint(serialize(model, model_type, dataloaders['train'].dataset), is_best=True, exp_dir=prune_metrics_dir)
        acc = metrics["best_val_acc"]
        epoch += 1

    # Load the best performing model
    best_model_path = os.path.join(prune_metrics_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best weights from epoch {metrics['best_val_epoch']} with accuracy {metrics['best_val_acc']:.3f}")
        model.load_state_dict(torch.load(best_model_path)['state_dict'])

    return model

def build_model(model_type, vocab, vocab_size=None, pretrained=True, embedding_dim=300, hidden_dim=512, device='cuda', is_cofi=False):
    """
    Builds the specified model. `vocab_size` is only used for the bowman model.
    """
    tokenizer=None
    if is_cofi:
        if model_type=='bowman':
            from cofi_models import modeling_bowman
            tokenizer = modeling_bowman.TextEncoder(len(vocab['stoi']))
            model = modeling_bowman.CoFiBowmanEntailmentClassifier(tokenizer, 'cuda')

        elif model_type=='bert':
            from cofi_models import modeling_bert
            model = modeling_bert.CoFiBertForSequenceClassification
        else:
            from cofi_models import modeling_llama
            model = modeling_llama.CoFiLlamaForSequenceClassification
            
    
    else: 
        if model_type == 'bert':
            # CORRECTED: Removed the 'vocab' argument
            model = nli_models.BertEntailmentClassifier(pretrained=pretrained, device=device)
        elif model_type == 'llama':
            # CORRECTED: Removed the 'vocab' argument
            model = nli_models.LLAMAEntailmentClassifier(pretrained=pretrained, device=device)
        elif model_type == 'bowman':
            # This path remains the same
            tokenizer = nli_models.TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
            model = nli_models.BowmanEntailmentClassifier(tokenizer, device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    return model, tokenizer

def load_model(model_type, train, ckpt=None, use_pretrained_weights=True, pruning_method='', device='cuda', i=0, zs=None):
    """
    Loads or initializes a model.
    """
    # CORRECTED: Call the updated build_model function
    cofi = pruning_method == 'CoFi'
    model, tokenizer = build_model(
        vocab= {'itos':train.itos, 'stoi': train.stoi},
        model_type=model_type,
        vocab_size=len(train.stoi), # For bowman
        pretrained=use_pretrained_weights, # For bert
        device=device,
        is_cofi=cofi
    )
    
    if ckpt and os.path.exists(ckpt): #and not cofi
        if zs or cofi:
            print(f"Loading zs")
            model = cofi_utils.load_model_with_zs(ckpt, model, zs=zs, train_data=train, ckpt=ckpt, encoder=tokenizer)
        else:
            print(f"Loading from checkpoint (no zs): {ckpt}")
            ckpt_ = torch.load(ckpt, map_location=torch.device(device))
      
            model.load_state_dict(state_dict=ckpt_["state_dict"], strict=False)
    else:
        # This logic for saving initial weights is fine
        print("Loading pretrained/untrained weights")
        save_dir_type = "pretrained" if use_pretrained_weights else "untrained"
        save_dir = os.path.join(model_type.upper(), "models", save_dir_type)
        os.makedirs(save_dir, exist_ok=True)
        if not ckpt:
            filename = f"{model_type}_MAIN_{save_dir_type}_inits.pth"
        else:
            filename=ckpt.split('/')[-1]
        util.save_checkpoint(
            serialize(model, model_type, train), False, save_dir, filename
        )
        ckpt = os.path.join(save_dir, filename)
        if cofi:
            print(f"Loaded new CoFi instance (pretrained)")
    
    
        
    if 'model_best.pth' not in ckpt:
        ckpt = os.path.join(ckpt, 'model_best.pth')
    return model.to(device), ckpt


def serialize(model, model_type, dataset):
    # CORRECTED: The condition now correctly checks if model_type is in the list

    if model_type in ['llama', 'bert']:
        return {
            "encoder_name": model.model_name, 
            "state_dict": model.state_dict()
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
        if pruning_method == 'CoFi':
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

