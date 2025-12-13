
import sys
sys.path.append('/workspace/CCE_NLI/code')

import models
import torch
import snli_eval
import train_utils
import pandas as pd
import csv
import settings
from collections import defaultdict
import numpy as np 
import os
from analyze import pad_collate, pairs
from data import analysis
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import pandas as pd
from pathlib import Path
def get_model(args, train):

    # ==== BUILD MODEL ====
    model = train_utils.build_model(vocab_size=len(train.stoi), model_type=args.model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, is_cofi=args.pruning_method=='cofi')

    return model
def parse_tensor_string(tensor_str):
    """
    Convert string like '[tensor(0.0001), tensor(1.0171)]' to [0.0001, 1.0171]
    """
    # Extract all numbers from tensor() calls
    numbers = re.findall(r'tensor\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)', tensor_str)
    return [float(num) for num in numbers]

def load_activation_range_neuron_data(folder_path):
    """
    Load all Cluster*.IOUS1024N.csv files and build the cluster dictionary.
    
    Returns:
        dict: Structure like {'c1': {unit: [start, end], ...}, 'c2': {...}, ...}
    """
    cluster_dict = {}
    

    for cluster in range(1,4):
        cluster_name = f'c{cluster}'
        csv_file=os.path.join(folder_path,f'Cluster{cluster}IOUS1024N.csv')
        # Read the CSV
        df = pd.read_csv(os.path.join(folder_path,csv_file))

        # Initialize cluster dict
        cluster_dict[cluster] = {}

        # Process each row
        for idx, row in df.iterrows():

            unit = row['unit']
            activation_str = row['activation_value_for_samples']

            # Parse the activation values
            activation_range = parse_tensor_string(activation_str)

            if len(activation_range) == 2:
                cluster_dict[cluster][unit] = activation_range
            else:
                print(f"Warning: Unexpected activation format in {csv_file}, unit {unit}")
    print(cluster_dict[3])
    return cluster_dict

def map_samples_to_neurons(cluster_dict, sample_id, sample_activations):
    """
    Map samples to neurons within each cluster.
    
    Args:
        cluster_dict: The dictionary from load_cluster_data()
        all_sample_activations: dict like {sample_id: {neuron_id: activation_value, ...}, ...}
    
    Returns:
        dict: Structure like {
            'c1': {neuron1, neuron2,...},
            'c2': {neuron3: [samples], neuron4: [samples], ...},
            'c3': {...}
        }
        Example: {
            'c1': {
                'neuron_1',
                'neuron_2'
            },
            'c2': {
                'neuron_3',
            }
        }
    """
    # Initialize the result structure
    result = defaultdict(list)
    
 
    # For each neuron activation in the sample
    print(max(sample_activations))
    for neuron, activation_value in enumerate(sample_activations):
        # Check each cluster to see if this neuron belongs and is active
       
        for cluster_name, units in cluster_dict.items():
            
            if neuron not in units: continue
           
            if units[neuron][0] <= activation_value <= units[neuron][1]: 
                result[cluster_name].append(neuron)
                break
                

    return result
def refactor_structure(data):
    """
    Refactor from {sample: {cluster: [neurons]}} 
    to {cluster: {neuron: [samples]}}
    """

    result = {}

    for sample, clusters in data.items():
        
      
        for cluster, neurons in clusters.items():
            # Initialize cluster dict if not exists
            if cluster not in result:
                result[cluster] = {}
            
            # Add sample to each neuron's list
            for neuron in neurons:
                if neuron not in result[cluster]:
                    result[cluster][neuron] = set()
                result[cluster][neuron].add(sample)
    #print(f"Cluster3 neurons: {result[cluster].keys()}")
    return result


def run_eval(model, val_loader, model_type, pruning_method, neuron_range_mapping={}):
    model.cuda()
    model.eval()
    all_preds = []
    all_targets = []
    all_final_layer=[]
    # CORRECTED: Added conditional logic for batch handling
    for batch in val_loader:
        if pruning_method == 'cofi':
            if torch.cuda.is_available():

                batch = {k: v.to('cuda') for k, v in batch.items()}
                targets = batch['labels']


            batch_size = targets.shape[0]

            with torch.no_grad():
                final_layer_logits = model.get_final_reprs(**batch)
                
                logits = model(**batch)

            preds = logits[1][2].argmax(1)
        
        else:
            if model_type in ['bert', 'llama']:
                s1_batch, s2_batch, targets = batch
                if settings.CUDA:
                    s1_batch = {k: v.cuda() for k, v in s1_batch.items()}
                    s2_batch = {k: v.cuda() for k, v in s2_batch.items()}

                with torch.no_grad():
                    logits = model(s1_batch, s2_batch)
                    final_layer_logits = model.get_final_reprs(s1_batch, s2_batch) #8,1024
                    np.save('my_array.npy', final_layer_logits.cpu().numpy())
                    exit(1)
              
            else: # Bowman path
                s1, s1len, s2, s2len, targets = batch
                if settings.CUDA:
                    s1, s1len = s1.cuda(), s1len.cuda()
                    s2, s2len = s2.cuda(), s2len.cuda()

                with torch.no_grad():
                    final_layer_logits = model.get_final_reprs(s1, s1len, s2, s2len)
                    
                    logits = model(s1, s1len, s2, s2len)
            
        
            preds = logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_final_layer.append(final_layer_logits.cpu().numpy())
        
        

    all_preds = np.concatenate(all_preds, 0)
    all_targets = np.concatenate(all_targets, 0)
    all_final_layer = np.concatenate(all_final_layer, 0)
    acc = (all_preds == all_targets).mean()
    
    neurons_actvated={}
    
    for sample, sample_activations in enumerate(all_final_layer):
        neurons_actvated[sample] = map_samples_to_neurons(neuron_range_mapping, sample, sample_activations)
        print(sample,sample_activations,  neurons_actvated[sample], neuron_range_mapping[2][19])
        exit(1)
       
    
    responses = {}

    for sample_num, (pred, target) in enumerate(zip(all_preds, all_targets)):

        coorect = 1 if pred == target else 0
        responses[sample_num] = coorect
  
        
 
    return responses, refactor_structure(neurons_actvated)

def load_formula_mask(path='CCE_NLI/BERT/formula_masks/lottery_ticket/Run0.25_3/formula_masks_0.0.json'):
    import json
    formula_mask_dict=defaultdict(lambda: defaultdict(list))
    # Open the JSON file
    with open(path, "r") as file:
        data = json.load(file)
    for cluster, mask in data.items():
        for neuron, samples in mask.items():
            for num,bin_val in enumerate(samples):
                if bin_val==1:
                    formula_mask_dict[cluster][neuron].append(num)

    return formula_mask_dict



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
   
    parser.add_argument("--ckpt", default="BERT/models/lottery_ticket/Run0.25_3/0_Pruning_Iter/model_best.pth")
    parser.add_argument("--model_type", default="bert", choices=["bowman", "bert", "llama"])
    parser.add_argument("--pruning_method", default="lottery_ticket", choices=["lottery_ticket", "wanda", "cofi"])
    parser.add_argument("--base_exp_dir", default="/workspace/CCE_NLI/BERT/exp/lottery_ticket/Run0.25_3/Expls/0.0%Pruned/", choices=["lottery_ticket", "wanda", "cofi"],)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train,_,dataloaders=train_utils.create_dataloaders(max_data=None, model_type=args.model_type, pruning_method=args.pruning_method)
    val_loader = dataloaders['val']
    model = get_model(args, train)
    
    print("Getting range mapping")
    neuron_range_mapping = load_activation_range_neuron_data(args.base_exp_dir)
    print("Getting preds and activations")
    response, neurons_actvated =run_eval(model, val_loader, args.model_type, args.pruning_method, neuron_range_mapping)
    
    
    ious=defaultdict(dict)
    print("Getting IOUS")
    for cluster in range(1,4):
        path=os.path.join(args.base_exp_dir, f'Cluster{cluster}IOUS1024N.csv')
        df = pd.read_csv(path)
        for neuron, iou in zip(df.unit, df.best_iou):
            ious[f'c{cluster}'][neuron] = iou
    
    print("Getting Neuron Accs")
    print(neurons_actvated.keys())
    
    for c in range(1,4):
        neuron_accuracies = {}
        for neuron, samples in neurons_actvated[c].items():
            neuron_acc=0

            for sampl in samples:
                if response[sampl]==1:
                    neuron_acc += 1

            neuron_accuracies[neuron] = {'Accuracy': neuron_acc / len(samples), 'IOU': ious[f'c{c}'][neuron]}
        #print(neuron_accuracies)
        pd.DataFrame(neuron_accuracies).transpose().rename(columns={'Unnamed: 0': 'unit', '0': 'Accuracy', '1': 'IOU'}).to_csv(f"neuron_interpret_Cluster{c}.csv")
      
            
        
    
    





