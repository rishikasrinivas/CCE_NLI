import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from analyze import initiate_exp_run 
from tqdm import tqdm
import train_utils
import settings

def make_folders(root_dir, prune_iter):
    #masks and explanation storing paths after finetuning
    exp_after_finetuning_flder = f"{root_dir}/Expls/{prune_iter}_Pruning_Iter/"
    if not os.path.exists(exp_after_finetuning_flder):
        os.makedirs(exp_after_finetuning_flder,exist_ok=True) 

    masks_after_finetuning_flder = f"{root_dir}/Masks/{prune_iter}_Pruning_Iter/"
    if not os.path.exists(masks_after_finetuning_flder):
        os.makedirs(masks_after_finetuning_flder, exist_ok=True)
    return exp_after_finetuning_flder, masks_after_finetuning_flder

def percent_pruned_weights(model, layer_name=None):
    if layer_name:
        layer = model.get_layer(layer_name)
        final_weights_pruned = torch.where(layer.weights.detach() == 0,1,0).sum().item() /(layer.weights.shape[0] * layer.weights.shape[1])
        return final_weights_pruned
 
    for layer in model.layers:
        if 'bias' in str(layer.name) or 'bn' in str(layer.name) or layer.name == 'encoder.emb.weight':
            continue
        layer = model.get_layer(layer.name)
        final_weights_pruned = torch.where(layer.weights.detach() == 0,1,0).sum().item() /(layer.weights.shape[0] * layer.weights.shape[1])
        break
    return final_weights_pruned

def get_percent_pruned(model):
    final_weights = model.mlp[0].weight.detach().cpu().numpy()
    final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(model.mlp[0].weight.shape[0]*model.mlp[0].weight.shape[1]), 3)
    return final_weights_pruned

def run_expls(
    args,
    model, 
    dataset,
    dataloaders,
    device
    ):
    
    '''
        args: Methods to prune and train model, then save explanations
        returns: Pruning Accuracies, percents, and cached formula masks (for analysis) 
        
        Runs through each ckpt in the directory and applies CCE 
    '''
    all_fm_masks = []
    
    # Gets the ckpt and numeric pruning iter
    for prune_iter in range(len(os.listdir(args.prune_metrics_dir))):
            
        prune_metrics_dir  = f"{prune_iter}_Pruning_Iter"
        filepath = os.path.join(args.prune_metrics_dir, prune_metrics_dir,"model_best.pth" )
        
        #ignores invalid flders/files
        if prune_metrics_dir not in os.listdir(args.prune_metrics_dir): continue
        model.to(device)
        
        #=== Loading weights ===
        print(f"Loading from {filepath}")
        model.load_state_dict(torch.load(filepath)['state_dict']) #loading the already finetuned weights
        
        # === Recording Accs and Pruned Percents
        final_weights_pruned = get_percent_pruned(model)
    
        # === Runs explanations ===
        if final_weights_pruned < args.max_thresh: #or :
            exp_after_finetuning_flder, masks_after_finetuning_flder = make_folders(args.expls_mask_root_dir, final_weights_pruned)
            print(f"======Running Explanations for {final_weights_pruned}% pruned=======")
            
            _, formulaMasks =initiate_exp_run(
                save_exp_dir = exp_after_finetuning_flder, 
                save_masks_dir= masks_after_finetuning_flder, 
                masks_saved=False, 
                model_=model,
                dataset=dataset
            )
            
            all_fm_masks.append(formulaMasks)
    
         
        else:
            break
    return all_fm_masks


def main():
    if args.debug:
        max_data = 1000
    else:
        max_data = None
        
    train,_,_,dataloaders=train_utils.create_dataloaders(max_data=max_data)
    model = train_utils.load_model(max_data=max_data, model_type=args.model_type, train=train, ckpt=args.ckpt)
    
    # ==== BUILD VOCAB ====
    base_ckpt=torch.load(args.ckpt) #trained bowman/bert 
    vocab = {"itos": base_ckpt["itos"], "stoi": base_ckpt["stoi"]}

    with open(settings.DATA, "r") as f:
        lines = f.readlines()
    
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    device = 'cuda' if settings.CUDA else 'cpu'

    print(f"======RUNNING EXPLANATIONS WITH {settings.NUM_CLUSTERS} CLUSTERS")
    _,final_layer_weights =initiate_exp_run(
        save_exp_dir = f"exp/random/expls/bowman", 
        save_masks_dir= f"exp/random/masks/bowman", 
        masks_saved=False, 
        model_=model,
        dataset=dataset
    )
    