"""
Train a bowman et al-style SNLI model
"""

import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW

# --- Local Imports ---
# Ensure your python path is set up correctly for these
sys.path.append('code/')
import settings
import models
import util
import train_utils
import prune_utils
from data import analysis
from Pruner import Pruner_


def main(args):
    max_data = 1000 if args.debug else None
    use_pretrained_weights = not args.untrained_model
    
    # CORRECTED: Added model_type to the create_dataloaders call
    train, val, dataloaders = train_utils.create_dataloaders(
        max_data=max_data, 
        model_type=args.model_type, 
        debug=args.debug
    )
    
    # CORRECTED: Removed the unnecessary max_data argument
    model, ckpt = train_utils.load_model(
        model_type=args.model_type, 
        train=train, 
        use_pretrained_weights=use_pretrained_weights, 
        ckpt=args.ckpt, 
        i=args.i
    )
    
    base_ckpt = torch.load(ckpt, map_location='cpu')

    # CORRECTED: This block is only relevant for the bowman model.
    # It will raise a KeyError for transformer models otherwise.
    if args.model_type == 'bowman':
        vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}
        with open(settings.DATA, "r") as f:
            lines = f.readlines()
        dataset = analysis.AnalysisDataset(lines, vocab)
    else:
        dataset = None # This isn't used by the transformer path in this script

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) if args.model_type in ['bert', 'llama'] else optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if settings.CUDA else 'cpu'
    print(f"Running on {device}")
    
    pruner = Pruner_(model)

    return run_prune(
        model, pruner, args, base_ckpt, dataset, optimizer, 
        criterion, device, train, val, dataloaders, 
        start=args.restart_from_ckpt
    )

def get_mask(weights):
    return torch.where(weights == 0, 0, 1)

def apply_mask(model, base_ckpt):
    not_pruneable_layers = []
    for layer in base_ckpt['state_dict'].keys():
        try:
            # Applying mask to a CPU tensor
            base_ckpt['state_dict'][layer] *= model.get_layer(layer).pruning_mask.cpu()
        except LookupError:
            not_pruneable_layers.append(layer)
            continue
    return base_ckpt

def run_prune(model, pruner, args, base_ckpt, dataset, optimizer, criterion, device, train, val, dataloaders, start):
    print("Entered run_prune")
    pruned_percents, final_accs = [], []
    prune_metrics_dir_base = os.path.join(args.model_type.upper(), "models", "lottery_ticket", args.filename)
    os.makedirs(prune_metrics_dir_base, exist_ok=True)
    
    # Logic for restarting from a checkpoint can remain as you had it
    if start > 0:
        # ... your checkpoint restarting logic ...
        pass

    for prune_iter in range(start, args.prune_iters):
        print(f"\n--- Pruning Iteration {prune_iter} ---")
        
        # Re-initialize the optimizer at the start of each finetuning run
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) if args.model_type in ['bert', 'llama'] else optim.Adam(model.parameters())
        model.to(device)
        print(f"[Check] Model is on {next(model.parameters()).device} for training.")

        prune_metrics_dir = os.path.join(prune_metrics_dir_base, f"{prune_iter}_Pruning_Iter")
        os.makedirs(prune_metrics_dir, exist_ok=True)

        # Finetune the model (it will save the best version)
        model = train_utils.finetune_pruned_model(
            model, args.model_type, optimizer, criterion, dataloaders, 
            args.finetune_epochs, prune_metrics_dir, device
        )

        # Evaluate the best model from the finetuning phase
        # CORRECTED: Added model_type to the run_eval call
        final_acc = train_utils.run_eval(model, dataloaders['val'], args.model_type)
        final_weights_pruned = prune_utils.percent_pruned_weights(model)
        
        print(f"Iteration {prune_iter}: Percent Pruned: {final_weights_pruned:.2f}% | Validation Accuracy: {final_acc:.3f}")
        pruned_percents.append(final_weights_pruned)
        final_accs.append(final_acc)

        if final_weights_pruned >= args.max_thresh:
            print(f"Pruning threshold of {args.max_thresh * 100}% reached. Stopping.")
            break

        # Prune the model further for the next iteration
        model = pruner.prune() 
        base_ckpt = apply_mask(model, base_ckpt)

        # Load the pruned initial weights for the next round of training ("rewinding")
        model.load_state_dict(base_ckpt['state_dict']) 
        
    return pruned_percents, final_accs

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--filename", type=str, required=True, help="Unique filename for this experiment's checkpoints.")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "bert", "llama"])
    parser.add_argument("--ckpt", default=None, help="Path to a checkpoint to start from.")
    parser.add_argument("--untrained_model", action="store_true", default=False)
    
    parser.add_argument("--finetune_epochs", default=6, type=int)
    parser.add_argument("--prune_iters", default=20, type=int)
    parser.add_argument("--restart_from_ckpt", default=0, type=int)
    parser.add_argument("--max_thresh", default=0.95, type=float)
    
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--i", default=0, type=int, help="Initialization index.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Your settings.py should handle CUDA setup
    settings.CUDA = torch.cuda.is_available()
    pruned_percents, final_accs = main(args)
    
    print("\n--- Experiment Summary ---")
    print(f"Sparsity Levels (%): {pruned_percents}")
    print(f"Final Accuracies: {final_accs}")