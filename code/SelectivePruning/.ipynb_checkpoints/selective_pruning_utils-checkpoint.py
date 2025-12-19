import os
import pandas as pd
import re
from collections import defaultdict
import json
import numpy as np
import torch
from pathlib import Path
def get_indiv_concepts(formula) -> set:
    concepts = set()
    concps = re.findall(r'(?<!\bNOT\s)(?:\b(?:hyp|pre|oth):[^\s)]+)', formula)
    for c in concps:
        try:
            end_idx = c.index(')')
        except:
            end_idx = len(c)
        concepts.add(c[:end_idx])
 
    return concepts
def get_k_neurons(concepts, starting_concept_idx, mapping, k):
    """
    Get k neurons by incrementally adding concepts starting from starting_concept_idx.
    
    Args:
        concepts: List of concepts to consider
        starting_concept_idx: Index in concepts list to start from
        mapping: Dict mapping {neuron: set of concepts}
        k: Target number of neurons to collect
    
    Returns:
        tuple: (selected_neurons, concepts_used, last_cp_idx)
    """
    selected_neurons = set()
    concepts_used = []
    last_cp_idx=starting_concept_idx
    # Iterate through concepts starting from starting_concept_idx
    for i in range(starting_concept_idx, len(concepts)):
        if len(selected_neurons) >= k:
            last_cp_idx = i
            break
            
        current_concept = concepts[i]
        concepts_used.append(current_concept)
        
        # Find all neurons that explain this concept
        for neuron, neuron_concepts in mapping.items():
            if current_concept in neuron_concepts:
                selected_neurons.add(neuron)
                
                # Stop early if we've reached k neurons
                if len(selected_neurons) >= k:
                    last_cp_idx = i
                    break
    
    return list(selected_neurons)[:k], concepts_used, last_cp_idx



    
def load_csv_data(filepath):
    """Load CSV and extract unit-concept mappings."""
    df = pd.read_csv(filepath)
    unit_concepts = defaultdict(set)
    
    for _, row in df.iterrows():
        unit = row['unit']
        formula = row['best_name']
        concepts = get_indiv_concepts(formula)
        unit_concepts[unit].update(concepts)
    
    return unit_concepts

def build_binary_mask(neuron_mask, conceptset) -> torch.Tensor:
    num_neurons = len(neuron_mask)
    num_concepts = len(conceptset)

    # Step 1: Initialize tensor
    tensor = torch.zeros((num_neurons, num_concepts), dtype=torch.float32)

    # Step 2: Fill in ones
    for i, concepts in enumerate(neuron_mask.values()):
        for j, concept in enumerate(conceptset):
            if concept in concepts:
                tensor[i, j] = 1.0

    # Step 3: Compute row sums
    row_sums = tensor.sum(dim=1, keepdim=True)

    # Step 4: Normalize safely
    dist_tensor = torch.zeros_like(tensor)
    row_mask = (row_sums != 0).squeeze(1)  # True for rows with sum > 0
    dist_tensor[row_mask] = tensor[row_mask] #/ row_sums[row_mask]

    return dist_tensor

def get_topk_concepts(mask,k, concept_list):
    """Test if concepts are uniformly distributed across neurons."""
    col_sums = mask.sum(dim=0).numpy()
    sorted_idx = np.argsort(col_sums)[::-1]
    top=[]
    freqs={}
    for i in range(len(sorted_idx))[:k]:
        idx = sorted_idx[i]
        top.append(concept_list[idx])
        freqs[concept_list[idx]]=col_sums[idx]
    return top, freqs

def get_foundationals(root_dir):
    root_path = Path(root_dir)

    # Find all matching CSV files
    fldr_pattern = '*%Pruned'
    fldr_files = list(root_path.rglob(fldr_pattern))
    cross_iter_concept_dict=defaultdict(list)
    for fldr_file in fldr_files:
        concepts = []
        for csvs in os.listdir(os.path.join(root_dir, fldr_file)):

            if 'IOUS1024N' not in csvs: continue

            csv_file = os.path.join(root_dir, fldr_file, csvs)
            df = pd.read_csv(csv_file)
            for unit, formula in zip(df.unit, df.best_name):
                concepts.extend(get_indiv_concepts(formula))
        cross_iter_concept_dict[fldr_file]=set(concepts)
    preserved_concepts = set.intersection(*cross_iter_concept_dict.values())
   
    return list(preserved_concepts)

def get_non_foundationals(foundational, pi):
    unit_to_cp_dict = get_all_cps_for_pi(pi)
    allcps = set()
    for unit,cps in unit_to_cp_dict.items():
        allcps.update(cps)
    return list(allcps - set(foundational))

def get_neurons_for_cps(concepts, mapping):
    neurons = []
    for neuron, cps in mapping.items():
        for c in cps:
            if c in concepts:
                neurons.append(neuron)
                break
    return neurons

def get_all_cps_for_pi(folder):
    root_path = Path(folder)

    # Find all matching CSV files
    csv_pattern = 'Cluster*IOUS1024N.csv'
    csv_files = list(root_path.rglob(csv_pattern))

    
    concept_dict=defaultdict(set)
    for csv_file in csv_files:
        concepts = []
        csv_file = os.path.join(folder, csv_file)
        df = pd.read_csv(csv_file)
        for unit, formula in zip(df.unit, df.best_name):
            concept_dict[unit].update(set(get_indiv_concepts(formula)))
    return concept_dict