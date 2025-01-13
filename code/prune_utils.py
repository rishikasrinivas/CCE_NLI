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

def percent_pruned_weights(model):
    final_weights_pruned = 0
    for layer in model.layers:
        if 'bias' in str(layer.name) or 'bn' in str(layer.name):
            continue
        layer = model.get_layer(layer.name)
        final_weights_pruned += torch.where(layer.weights.detach() == 0,1,0).sum().item() / model.get_total_num_weights())
    return final_weights_pruned


def run_expls(
    args,
    model, 
    dataset, 
    optimizer, 
    criterion, 
    dataloaders,
    device, 
    ):
    
    '''
        args: methods to prune and train model, then save explanations
        returns: path of pruning and accuracies 
        
        Applies lottery ticket hypothesis to prune and finetune a model reloading the initialized weights between pruning iterations 
    '''
    pruned_percents, final_accs = [], []
    
    
    for prune_iter in tqdm(range(0,args.prune_iters)):
        print(f"==== PRUNING ITERATION {prune_iter}/{args.prune_iters+1} ====")
        
        prune_metrics_dir = os.path.join(args.prune_metrics_dir, f"{prune_iter}_Pruning_Iter")
        if not os.path.exists(prune_metrics_dir):
            os.makedirs(prune_metrics_dir,exist_ok=True)
            os.makedirs(prune_metrics_dir,exist_ok=True)
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
