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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_model(model_name, ckpt):
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=10000)

    model = train_utils.load_model(10000, model_name, train, ckpt=ckpt, device='cuda')

    torch.save(model.state_dict(), "Results/bert_not_prunedby_wanda.pth")


    return model, dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--seg', type=str, choices=["enc", "mlp"])
    
  
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--ckpt", default=None, type=str )

    parser.add_argument('--use_variant', default=False, action="store_true", help="whether to use the wanda variant described in the appendix")
    
    parser.add_argument("--expls_mask_root_dir", default="exp/bert/wanda/Run1")
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
    
    prune_n,prune_m=0,0
    
    os.makedirs(args.expls_mask_root_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    
    model,dataloaders = get_model(args.model_type, args.ckpt)
    
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(args.ckpt) #models/snli/bowman_random_inits.pth
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0")
    
    wanda.prune_wanda(args, model, dataloaders, device, prune_n=prune_n, prune_m=prune_m)
    
    prune_utils.run_expls(args, model, dataset, optimizer, criterion, dataloaders, device)
    eval_test = train_utils.run_eval(model, dataloaders['val'], device)
    
    print(f"NLI Eval: {eval_test}")

main()