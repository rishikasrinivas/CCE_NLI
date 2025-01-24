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


def run_expls(
    args,
    model, 
    dataset,
    dataloaders,
    device
    ):
    
    '''
        args: methods to prune and train model, then save explanations
        returns: path of pruning and accuracies 
        
        Applies lottery ticket hypothesis to prune and finetune a model reloading the initialized weights between pruning iterations 
    '''
    pruned_percents, final_accs = [], []
    
    
    for prune_iter, prune_metrics_dir in enumerate(os.listdir(args.prune_metrics_dir)):
        if not prune_metrics_dir[0].isdigit(): continue
        model.to(device)
        
        print(f"Loading from {prune_metrics_dir}/model_best.pth")
        model.load_state_dict(torch.load(f"{prune_metrics_dir}/model_best.pth")['state_dict']) #loading the already finetuned weights
        final_weights = model.mlp[0].weight.detach().cpu().numpy()
        
        final_acc=train_utils.run_eval(model, dataloaders['val'])
        print(f"Accuracy: {final_acc}")
        
        final_weights_pruned= np.round(100*torch.where(torch.tensor(final_weights) == 0,1,0).sum().item()/(model.mlp[0].weight.shape[0]*model.mlp[0].weight.shape[1]), 3)
        pruned_percents.append(final_weights_pruned)
        final_accs.append(final_acc)
        
        
        if final_weights_pruned < args.max_thresh: #or :
            exp_after_finetuning_flder, masks_after_finetuning_flder = make_folders(args.expls_mask_root_dir, final_weights_pruned)
            print(f"======Running Explanations for {final_weights_pruned}% pruned=======")
            #run after pruning before finetuning
            os.makedirs(f"{exp_after_finetuning_flder}3Clusters", exist_ok=True)
            os.makedirs(f"{masks_after_finetuning_flder}3Clusters", exist_ok=True)

            print(f"======RUNNING EXPLANATIONS WITH {settings.NUM_CLUSTERS} CLUSTERS")
            _,final_layer_weights =initiate_exp_run(
                save_exp_dir = f"{exp_after_finetuning_flder}3Clusters", 
                save_masks_dir= f"{masks_after_finetuning_flder}3Clusters", 
                masks_saved=False, 
                model_=model,
                dataset=dataset
            )
        
        
        after =np.round(torch.where(model.mlp[0].weight.detach() == 0,1,0).sum().item()*100/(model.mlp[0].weight.shape[0]*model.mlp[0].weight.shape[1]),2)
        print("After pruning: Final_wegihts prune% is: ", after)
        
   
        
        if final_weights_pruned >= args.max_thresh: break
    return  pruned_percents, final_accs


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
    