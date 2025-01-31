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
    parser.add_argument('--seg', type=str, choices=["enc", "mlp"])
    
  
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of calibration samples.')

    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--ckpt", default=None, type=str )
    parser.add_argument("--offset", default=0, type=int )

    parser.add_argument('--use_variant', default=False, action="store_true", help="whether to use the wanda variant described in the appendix")
    
    #parser.add_argument("--expls_mask_root_dir", default="exp/bert/wanda/Run1")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/wanda/BERT/Run1")
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=95, type=float)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()
    

    final_accs = []
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    
    ckpt = f"models/snli/prune_metrics/lottery_ticket/bert/{args.offset}_Pruning_Iter/model_best.pth"
    for i,sparsity_ratio in enumerate(settings.SPARSITY_RATIOS[args.offset:]):
        torch.cuda.empty_cache()
        os.makedirs(args.prune_metrics_dir, exist_ok=True)
        os.makedirs(f"{args.prune_metrics_dir}/{i+args.offset+1}_Pruning_Iter", exist_ok=True)
        
        # === Getting model ===
        
        model,dataloaders = prune_utils.get_model(args.model_type, ckpt)
   
    
        # ==== BUILD VOCAB ====
        base_ckpt=torch.load(ckpt) #trained model
        vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

        with open(settings.DATA, "r") as f:
            lines = f.readlines()

        dataset = analysis.AnalysisDataset(lines, vocab)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        #===== Debugging: Initial Eval =====
        eval_test = train_utils.run_eval(model, dataloaders['val'])

        print(f"Before Pruning NLI Eval: {eval_test}")

        #===== Pruning =====
        device = torch.device("cuda:0")
        wanda.prune_wanda(args, model, 'enc', dataloaders, sparsity_ratio, device)
        
        wanda.prune_wanda(args, model, 'mlp', dataloaders, sparsity_ratio, device)
        
        #===== Debugging: Pruning Verification =====
        weights_pruned = prune_utils.percent_pruned_weights(model, 'mlp.0.weight')
        print(f"IN MLP: {format(100*weights_pruned, '.2f')}% weights pruned")
        
        
        
        # ===== Saving model =====
        util.save_checkpoint(
                    train_utils.serialize(model, args.model_type, dataloaders['train'].dataset), False, args.prune_metrics_dir,filename = f"{i+args.offset+1}_Pruning_Iter/model_best.pth")

        
        #===== Recording Acc =====
        eval_test = train_utils.run_eval(model, dataloaders['val'])
        print(f"After Pruning NLI Eval: {eval_test}")
        final_accs.append(eval_test)
      
    print("SPARSITIES: ", settings.SPARSITY_RATIOS )
    print("FINAL ACCS: ", final_accs)

main()