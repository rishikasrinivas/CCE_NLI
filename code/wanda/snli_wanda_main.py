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
import wanda_utils

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

    parser.add_argument('--use_variant', default=False, action="store_true", help="whether to use the wanda variant described in the appendix")
    
    #parser.add_argument("--expls_mask_root_dir", default="exp/bert/wanda/Run1")
    parser.add_argument("--prune_metrics_dir", default="models/snli/prune_metrics/wanda/BERT/Run1")
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--max_thresh", default=95, type=float)
    
    parser.add_argument("--prune_epochs", default=10, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument("--prune_iters", default=1, type=int)
    
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()
    
    #prune_n,prune_m=0,0
    final_accs = []
    #os.makedirs(args.expls_mask_root_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    
    prune_iter=1
    for sparsity_ratio in settings.SPARSITY_RATIOS:
        os.makedirs(args.prune_metrics_dir, exist_ok=True)
        os.makedirs(f"{args.prune_metrics_dir}/{sparsity_ratio:.2f}", exist_ok=True)
        model,dataloaders = wanda_utils.get_model(args.model_type, args.ckpt)
    
    # ==== BUILD VOCAB ====
        base_ckpt=torch.load(args.ckpt) #models/snli/bowman_random_inits.pth
        vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

        with open(settings.DATA, "r") as f:
            lines = f.readlines()

        dataset = analysis.AnalysisDataset(lines, vocab)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        eval_test = train_utils.run_eval(model, dataloaders['val'])

        print(f"Before Pruning NLI Eval: {eval_test}")

        
        device = torch.device("cuda:0")
        wanda.prune_wanda(args, model, 'enc', dataloaders, sparsity_ratio, device)
        weights_pruned = prune_utils.percent_pruned_weights(model, 'encoder.rnn.weight_ih_l0')
        print("IN RNN: ", weights_pruned, " weights pruned")
        wanda.prune_wanda(args, model, 'mlp', dataloaders, sparsity_ratio, device)
        weights_pruned = prune_utils.percent_pruned_weights(model, 'mlp.0.weight')
        print("IN MLP: ", weights_pruned, " weights pruned")
        
        
        
        


        
        
        util.save_checkpoint(
                    train_utils.serialize(model, args.model_type, dataloaders['train'].dataset), False, args.prune_metrics_dir,filename = f"{prune_iter}_Pruning_Iter/model_best.pth")
        
        
        #prune_utils.run_expls(args, model, dataset, optimizer, criterion, dataloaders, device)
        
        eval_test = train_utils.run_eval(model, dataloaders['val'])

        print(f"After Pruning NLI Eval: {eval_test}")
        final_accs.append(eval_test)
        prune_iter+=1
    print("SPARSITIES: ", settings.SPARSITY_RATIOS )
    print("FINAL ACCS: ", final_accs)

main()