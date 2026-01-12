import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from analyze import initiate_exp_run 
from tqdm import tqdm
import train_utils
import settings
import json
from cofi.utils.utils import calculate_parameters
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
        if any(x in str(layer.name) for x in ['bias', 'bn', 'emb', 'LayerNorm', 'rnn']):
            continue
        layer = model.get_layer(layer.name)
        final_weights_pruned = torch.where(layer.weights.detach() == 0,1,0).sum().item() /(layer.weights.shape[0] * layer.weights.shape[1])
        break
    return final_weights_pruned

def get_percent_pruned(model):
    final_weights = model.mlp[0].weight.detach().cpu().numpy()
    final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(model.mlp[0].weight.shape[0]*model.mlp[0].weight.shape[1]), 3)
    return final_weights_pruned


def get_sparsity(model, zs):
    print(model.mlp[0].weight.shape[0])
    return 1 -  (model.mlp[0].weight.shape[0]/1024)
    
    return None
def run_expls(
    args,
    model, 
    dataset,
    dataloaders,
    device,
    train,
    debug,
    logger
    ):
    
    '''
        args: Methods to prune and train model, then save explanations
        returns: Pruning Accuracies, percents, and cached formula masks (for analysis) 
        
        Runs through each ckpt in the directory and applies CCE 
    '''
    all_fm_masks = []
    path_to_weights = os.path.join(args.model_type.upper(), "models", args.pruning_method, args.filename)
    path_to_explanations = os.path.join(args.model_type.upper(), "exp", args.pruning_method, args.filename, 'Expls')
    path_to_activation_masks = os.path.join(args.model_type.upper(), "exp", args.pruning_method, args.filename, 'Masks')
    path_to_activations = os.path.join(args.model_type.upper(), "activations", args.pruning_method, args.filename)
    path_to_formula_masks = os.path.join(args.model_type.upper(), "formula_masks", args.pruning_method, args.filename)
    logger.info(f"Loading weights from {path_to_weights}")
    os.makedirs(path_to_explanations, exist_ok=True)
    os.makedirs(path_to_activation_masks, exist_ok=True)
    os.makedirs(path_to_activations, exist_ok=True)
    os.makedirs(path_to_formula_masks, exist_ok=True)
    
    # Gets the ckpt and numeric pruning iter
    for prune_iter in range(0, len(os.listdir(path_to_weights)) +1):
        prune_metrics_dir  = f"{prune_iter}_Pruning_Iter"
        
        if prune_metrics_dir not in os.listdir(path_to_weights): 
            logger.info(f"{prune_metrics_dir} is not a valid directory. Skipping")
            continue
        
        filepath = os.path.join(path_to_weights, prune_metrics_dir,"model_best.pth" )
        
        #ignores invalid flders/files
        if prune_metrics_dir not in os.listdir(path_to_weights): continue
        model.to(device)
        
        #=== Loading weights ===
        print(f"Loading from {filepath}")
        
        #TODO: need to reload model load_model with zs from cofi utils and load zs (as demoed in calc_pruning)
        #model.load_state_dict(torch.load(filepath, map_location=torch.device(device))['state_dict'], strict=False) #loading the already finetuned weights
        zs=None
        if args.pruning_method == 'CoFi' and prune_iter>0:
            
            zs_path= os.path.join(path_to_weights, f'{prune_iter}_Pruning_Iter/zs.pt')
            zs = torch.load(zs_path)
        
        model,ckpt = train_utils.load_model(model_type=args.model_type, pruning_method=args.pruning_method, train=train, ckpt=os.path.join(path_to_weights, f'{prune_iter}_Pruning_Iter', 'model_best.pth'), device=device, zs=zs)
        if prune_iter==0:
            original_model_size=calculate_parameters(model)

            
        
        # === Recording Accs and Pruned Percents
        if args.pruning_method == 'CoFi':
            pruned_model_size = calculate_parameters(model)
            final_weights_pruned = 1 - (pruned_model_size / original_model_size)  
            
        else:
            final_weights_pruned = get_percent_pruned(model)
           
        print("Explaining: ", final_weights_pruned)
        # === Runs explanations ===
        if final_weights_pruned < args.max_thresh: #or :
            
            logger.info(f"======Running Explanations for {final_weights_pruned}% pruned=======")
            
            formulaMasks =initiate_exp_run(
                save_exp_dir = os.path.join(path_to_explanations, f"{final_weights_pruned}%Pruned"), 
                save_masks_dir= os.path.join(path_to_activation_masks, f"{final_weights_pruned}%Pruned"), 
                activations_dir=os.path.join(path_to_activations, prune_metrics_dir),
                device=device,
                train=train,
                masks_saved=False, 
                model_=model,
                model_type=args.model_type,
                dataset=dataset,
                debug=debug,
                is_cofi= args.pruning_method=='CoFi',
            )
            logger.info("Recorded explanations")
            
            all_fm_masks.append(formulaMasks)
            
            os.makedirs(path_to_formula_masks, exist_ok=True)
            with open(os.path.join(path_to_formula_masks,f"formula_masks_{final_weights_pruned}.json"), "w") as f:
                json.dump(formulaMasks, f)
            logger.info(f"Saved formula masks to {os.path.join(path_to_formula_masks,f'formula_masks_{final_weights_pruned}.json')} ")
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
    