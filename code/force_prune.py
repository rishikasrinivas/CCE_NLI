import train_utils
import settings
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    train,val,test,dataloaders=train_utils.create_dataloaders(max_data=max_data)
    model = train_utils.load_model(max_data=max_data, train=train)
    base_ckpt=torch.load(settings.MODEL) 
    dataset = analysis.AnalysisDataset(lines, vocab)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    if settings.CUDA:
        device='cuda'
    else:
        device='cpu'
    return force_prune(model, optimizer, criterion,device, ft_epoch, prune_metrics_dirs, train,val,test,dataloaders):

def force_prune(model, optimizer, criterion,device, ft_epoch, prune_metrics_dirs, train,val,test,dataloaders):
    #weights = model.mlp[0].weight 
    #get max for each neuron
    #for each neuron only save the max weight 
    #create the pruning mask so it's 0 everywhere except at the mask
        #model.layers[mlp.0.weight].pruning_mask=mask
    #call train_utils.finetune 