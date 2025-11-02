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
import logging
# --- Local Imports ---
# Ensure your python path is set up correctly for these
sys.path.append('code/')
import settings
import models.nli_models as models
import util
import train_utils
import prune_utils
from data import analysis
from Pruner import Pruner_

logger = logging.getLogger(__name__)
def main(args):
    
    logging.basicConfig(filename='lottery_ticket.log', level=logging.INFO)
    max_data = 1000 if args.debug else 200
    use_pretrained_weights = not args.untrained_model
    
    logger.info(f"Creating Dataloaders with {max_data}")
    # CORRECTED: Added model_type to the create_dataloaders call
    train, val, dataloaders = train_utils.create_dataloaders(
        max_data=max_data, 
        model_type=args.model_type, 
        debug=args.debug,
        pruning_method='lottery_ticket'
    )
    
    logger.info(f"Loading model from {args.ckpt}")
    print(args.ckpt)
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
    
    logger.info("Instantiating Pruner")
    pruner = Pruner_(model)

    logger.info("Starting Pruning")
    return run_prune(
        model, pruner, args, base_ckpt, dataset, optimizer, 
        criterion, device, train, val, dataloaders, 
        start=args.restart_from_ckpt,
        start_idx =args.start_idx,
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

def run_prune(model, pruner, args, base_ckpt, dataset, optimizer, criterion, device, train, val, dataloaders, start, start_idx):
    print("Entered run_prune")
    pruned_percents, final_accs = [], []
    prune_metrics_dir_base = os.path.join(args.model_type.upper(), "models", "lottery_ticket", args.filename)
    os.makedirs(prune_metrics_dir_base, exist_ok=True)
    
    baseline_acc = -1.0
    if start:
        logger.info(f"Loading baseline model from {start}")
        model.load_state_dict(torch.load(os.path.join(start))['state_dict'])
        baseline_acc = train_utils.run_eval(model, dataloaders['val'], args.model_type, 'lottery_ticket')
        
        if os.path.exists(start):
            print(f"Alr lt'd {start}")
            state_dict =  torch.load(os.path.join(start), map_location=torch.device('cpu'))['state_dict']
            for layer in state_dict.keys():
                mask = get_mask(state_dict[layer])
                base_ckpt['state_dict'][layer] *= mask
                if not any(kw in layer for kw in ['bias', 'bn', 'embeddings', 'LayerNorm']):
                    model.set_mask(layer, mask)
                mask = mask.cpu()
            model.load_state_dict(base_ckpt['state_dict']) 
            
            model = pruner.prune() #PRUNE AND SAVE PRUNE MASK
            base_ckpt = apply_mask(model, base_ckpt)



            # Reload random inits with pruned weights (that were prnued after fting) 0'd out
            model.load_state_dict(base_ckpt['state_dict'])  
            final_weights_pruned = prune_utils.percent_pruned_weights(model)
            logger.info(f"After appling mask % Pruned: {final_weights_pruned}")
            model.cpu()
            
    logger.info(f"Starting pruning from start_idx: {start_idx}")
    for prune_iter in range(start_idx, args.prune_iters):
        logger.info(f"\n--- Pruning Iteration {prune_iter} / {args.prune_iters} ---")
        logger.info(f"Baseline Accuracy : {baseline_acc}")
        
        # Re-initialize the optimizer at the start of each finetuning run
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) if args.model_type in ['bert', 'llama'] else optim.Adam(model.parameters())
        model.to(device)
        print(f"[Check] Model is on {next(model.parameters()).device} for training.")

        prune_metrics_dir = os.path.join(prune_metrics_dir_base, f"{prune_iter}_Pruning_Iter")
        os.makedirs(prune_metrics_dir, exist_ok=True)

        # Finetune the model (it will save the best version)
        #EDIT: Adding baseline_acc as an argument
        if prune_iter > 0:
            model = train_utils.finetune_pruned_model(
                model, args.model_type, 'lottery_ticket', optimizer, criterion, dataloaders, 
                args.finetune_epochs, prune_metrics_dir, baseline_acc, device
            )

        # Evaluate the best model from the finetuning phase
        # CORRECTED: Added model_type to the run_eval call
        final_acc = train_utils.run_eval(model, dataloaders['val'], args.model_type, 'lottery_ticket')
        if prune_iter == 0: baseline_acc = final_acc
        final_weights_pruned = prune_utils.percent_pruned_weights(model)
        
        logger.info(f"Iteration {prune_iter}: Percent Pruned: {final_weights_pruned:.2f}% | Validation Accuracy: {final_acc:.3f}")
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

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

   
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/lottery_ticket/bowman")
    parser.add_argument("--i", default=0)
    #parser.add_argument("--root_metrics_dir", default="models/snli")
    #parser.add_argument("--model_dir", default="expls/snli/model_dir")
    parser.add_argument("--store_exp_bkdown", default="expls/snli_1.0_dev-6-sentence-5/")
    parser.add_argument("--filename", type=str)
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal", "bert", "llama"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--untrained_model", action="store_true", default=False)  # If `--untrained_model` is used, set to True 
    
    #parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=5, type=int)
    parser.add_argument("--prune_iters", default=5000, type=int)
    parser.add_argument("--restart_from_ckpt", default=None, type=str)
    parser.add_argument("--start_idx", default=0, type=int)
    
    
    parser.add_argument("--max_thresh", default=0.95, type=float)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--test_iters", default=1, type=int)
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--baseline", action='store_true')
    parser.add_argument("--ckpt", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pruned_percents, final_accs = main(args)
    print(f"pruned_percents: {pruned_percents}\nfinal_accs: {final_accs}")
    #wandb_ = wandb_init("CCE_NLI_Pruned_Model_Accs", "Run")
    #for i,acc in enumerate(final_accs):
      #  wandb_.log({"prune_iter": i, "accuracy_test": acc})