import sys
sys.path.append("./code")
import models
import cofi.utils.cofi_utils as cofi_utils
import torch
import snli_eval
import train_utils
import pandas as pd
import csv
from selective_pruning_utils import *
import random
def get_model(args,ckpt, train,zs=None):

    # ==== BUILD MODEL ====
    model,tok = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='CoFi')
    model, _=train_utils.load_model(model_type=args.model_type, train=train, ckpt=ckpt, use_pretrained_weights=True, pruning_method=args.pruning_method, device='cpu', i=0, zs=zs)
    return model,tok

def prune_neurons(model, ckpt, neurons_to_prune, prunedckpt=None):
    try:
        ckpt = torch.load(ckpt, map_location='cpu')['state_dict']
    except:
        print(f" ckpt not a file")
        
    #mimic wanda
    
#     for neuron, topruneckpt, refckpt in zip(range(1024), ckpt['mlp.0.weight'], ref['mlp.0.weight']):
#         for i,weight in enumerate(refckpt):
#             if weight==0:
#                 ref['mlp.0.weight'][neuron][i]=0
    #prune random weights
#     for neuron in range(1024):
#         for i in range(int(1024*neurons_to_prune)):
#             ckpt['mlp.0.weight'][neuron][i]=0
    #prune nrurins
    
    #prunedckpt should be fully wanda pruned but unprune the mlp
    prunedckpt['mlp.0.weight']=ckpt['mlp.0.weight']
    prunedckpt['mlp.3.weight']=ckpt['mlp.3.weight']
    #then manually prune mlp
    for neuron in neurons_to_prune:
        prunedckpt['mlp.0.weight'][neuron] = torch.zeros_like(prunedckpt['mlp.0.weight'][neuron])
        
        print(f"Pruned neuron {neuron}")
    
    
    print(torch.where(prunedckpt['mlp.0.weight']==0,1,0).sum())
    model.load_state_dict(prunedckpt)
    return model


    

def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        default="test.txt",
        help="Data to eval interactively (pairs of sentences); use - for stdin",
    )

    parser.add_argument("--model_type", default='bert', choices=["bowman", "bert", "llama"])
    parser.add_argument("--pruning_method", default="lottery_ticket", choices=["lottery_ticket", "wanda", "CoFi"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = parse_args()
    train,_,dataloaders=train_utils.create_dataloaders(max_data=None, model_type=args.model_type, pruning_method=args.pruning_method)
    val_loader = dataloaders['val']
    
   
    
    
    root_dir='/workspace/CCE_NLI/BERT/exp/lottery_ticket/Run0.25_5/Expls/0.0%Pruned'
   
                                              
    #test2 groups
    neuron_groups_formulas=get_all_cps_for_pi(root_dir)
    sparsity= {1:25.0, 2:43.75, 3:57.812, 4: 68.359, 5:76.27}
    #ckpt = f'LLAMA/models/lottery_ticket/Run0.25_5/0_Pruning_Iter/model_best.pth'
    #model,_ = get_model(args, ckpt, train)
    print("Removing BASELINE Concepts lost to WANDA")
    for i in [1,2]:
        ckpt = f'/workspace/CCE_NLI/BERT/models/lottery_ticket/Run0.25_5/0_Pruning_Iter/model_best.pth'
        pruned_ckpt = f'/workspace/CCE_NLI/BERT/models/lottery_ticket/Run0.25_5/1_Pruning_Iter/model_best.pth'
        model,_ = get_model(args, ckpt, train)
        dense_val_acc = train_utils.run_eval(model, val_loader, args.model_type, args.pruning_method)
        print(f"Initial dense acc at {sparsity[i]} = ", dense_val_acc)
        #for unit in neurons_to_prune:
        
#         print(f"Pruning out concepts that are lost to wanda from dense&pretrained {sparsity[i]}% ")
          #neurons_to_prune = get_neurons_for_cps(cps_to_prune[i]['removed_found'], neuron_groups_formulas)
#         print(f"Pruning {len(neurons_to_prune)} neurons that encompass all forgotten concepts")
        
#         ckpt = f'/workspace/CCE_NLI/LLAMA/models/wanda/Run0.25_pruneonlyenc/{i}_Pruning_Iter/model_best.pth'
#         specifically_pruned_model = prune_neurons(model, ckpt, neurons_to_prune=neurons_to_prune)
#         isPruned = [torch.sum(specifically_pruned_model.state_dict()['mlp.0.weight'][neuron])==0 for neuron in neurons_to_prune]
#         for j in isPruned:
#             assert j, 'some neuron not pruned'
#         torch.save(specifically_pruned_model.state_dict(), f'code/SelectivePruning/{i}_weights.pth')
#         specifically_pruned_model.eval()
#         specifically_pruned_model_val_acc = train_utils.run_eval(specifically_pruned_model, val_loader, args.model_type, args.pruning_method)
#         print(f"Validation acc: after pruning {len(neurons_to_prune)} = {specifically_pruned_model_val_acc}")

           
        neurons_to_prune = get_neurons_for_cps([], neuron_groups_formulas)
        ckpt = f'/workspace/CCE_NLI/BERT/models/lottery_ticket/Run0.25_5/0_Pruning_Iter/model_best.pth'
        specifically_pruned_model = prune_neurons(model, ckpt, neurons_to_prune=neurons_to_prune)
        specifically_pruned_model.eval()
        specifically_pruned_model_val_acc = train_utils.run_eval(specifically_pruned_model, val_loader, args.model_type, args.pruning_method)
        print(f"Validation acc: after pruning {len(neurons_to_prune)} = {specifically_pruned_model_val_acc}")


        #num_removed_found=len(cps_to_prune[i]['removed_found'])
        #concepts_to_prune = abs_to_rem
        #random_neurons_to_prune, concepts_pruned_out, _ = get_k_neurons(abs_to_rem, 0, neuron_groups_formulas, k=1024)
        random_neurons_to_prune  = [random.randint(0, 1024) for _ in range(256)]
        randomly_pruned_model = prune_neurons(model, ckpt,random_neurons_to_prune,pruned_ckpt)
        isPruned = [torch.sum(randomly_pruned_model.state_dict()['mlp.0.weight'][neuron])==0 for neuron in random_neurons_to_prune]
        for j in isPruned:
            assert j, 'some neuron not pruned'
        randomly_pruned_model.eval()
        randomly_pruned_model_val_acc = train_utils.run_eval(randomly_pruned_model, val_loader, args.model_type, args.pruning_method)
        print(f"Validation acc: after pruning {len(random_neurons_to_prune)} = {randomly_pruned_model_val_acc}")
        