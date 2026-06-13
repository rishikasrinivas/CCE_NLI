import sys
sys.path.append('code/')
import train_utils 
import wanda
import torch
import prune_utils
import argparse
import os
import settings
from data import analysis
import torch
import torch.optim as optim
import torch.nn as nn
import settings
import util

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--filename', type=str)
    
  
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of calibration samples.')

    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--ckpt", default=None, type=str )
    parser.add_argument("--offset", default=0, type=int )

    parser.add_argument('--use_variant', default=False, action="store_true", help="whether to use the wanda variant described in the appendix")

    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=99, type=float)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--untrained_model", action="store_true", default=False)  # If `--untrained_model` is used, set to True 
    
    args = parser.parse_args()
    

    final_accs = []
    if args.debug:
        max_data = 1000
    else:
        max_data = 10000
    if args.untrained_model:
        use_pretrained_weights = False
    else:
        use_pretrained_weights = True
        
    
    #========== Set up Env ===========
    prune_metrics_dir = os.path.join(args.model_type.upper(), "models", args.prune_method, args.filename)
    os.makedirs(prune_metrics_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"On device: {device}")
    
    #========== Set up model & dataset ===========
    train,val,dataloaders=train_utils.create_dataloaders(max_data=max_data, debug=args.debug, model_type=args.model_type, pruning_method='wanda')
    model,ckpt = train_utils.load_model( model_type=args.model_type, use_pretrained_weights = use_pretrained_weights, train=train, ckpt=args.ckpt,pruning_method='wanda',device=device)
    
    #========== Train model if ckpt dne ===========
    if not ckpt or 'pretrained' in ckpt:
        print(f"====Training {args.model_type} from {ckpt} and storing weights in {prune_metrics_dir}====")
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_utils.finetune_pruned_model(model,args.model_type, optimizer,criterion, dataloaders, 10, prune_metrics_dir,device='cuda')
        
    #========== Prune ===========
    print(f"====Done training. Going to prune from {prune_metrics_dir}/{args.offset} Pruning Iter====")
    for i,sparsity_ratio in enumerate(settings.SPARSITY_RATIOS[args.offset:]):
  
        torch.cuda.empty_cache()
        os.makedirs(f"{prune_metrics_dir}/{i+args.offset+1}_Pruning_Iter", exist_ok=True)
        
        # === Getting original (unpruned) model ===
        
        model.load_state_dict(torch.load(ckpt)['state_dict'])

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        #===== Debugging: Initial Eval =====
        #eval_test = train_utils.run_eval(model, dataloaders['val'])

        #print(f"Before Pruning NLI Eval: {eval_test}")

        #===== Pruning =====
      
        if args.model_type != 'bowman':
            wanda.prune_wanda(args, model, 'enc', dataloaders, sparsity_ratio, device)
        
        wanda.prune_wanda(args, model, 'mlp', dataloaders, sparsity_ratio, device)
        
        #===== Debugging: Pruning Verification =====
        weights_pruned = prune_utils.percent_pruned_weights(model, 'mlp.0.weight')
        print(f"MLP: {format(100*weights_pruned, '.2f')}% weights pruned")
        
        
        
        # ===== Saving model =====
        util.save_checkpoint(
                    train_utils.serialize(model, args.model_type, train), False, prune_metrics_dir,filename = f"{i+args.offset+1}_Pruning_Iter/model_best.pth")

        
        #===== Recording Acc =====
        eval_test = train_utils.run_eval(model, dataloaders['val'], args.model_type, 'wanda', device=device)
        print(f"After Pruning NLI Eval: {eval_test}")
        final_accs.append(eval_test)
      
    print("SPARSITIES: ", settings.SPARSITY_RATIOS )
    print("FINAL ACCS: ", final_accs)

main()