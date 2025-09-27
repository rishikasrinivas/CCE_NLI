import sys
sys.path.append('code/')
import metrics, random, torch,os
import pandas as pd
import numpy as np
import json
#{clus1: {neuon: mask}}
def calculate_alignment_with_original(all_fm_masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    initial_masks = all_fm_masks[0] #formula masks for without prunin
    if isinstance(initial_masks,list):
        initial_masks=initial_masks[0]
    for i, mask_dict in enumerate(all_fm_masks):
        if isinstance(mask_dict,list):
            mask_dict=mask_dict[0]
        for cluster, pair in mask_dict.items():
            neurons,alignments=[],[]

            for neuron, mask in pair.items():
                neurons.append(neuron)
                #print(initial_masks[cluster].keys(), neuron)
                if neuron in initial_masks[cluster].keys():
                    #print(f"Comparing mask {mask} with {initial_masks[cluster][neuron]}")
                    alignments.append(metrics.iou(torch.tensor(mask), torch.tensor(initial_masks[cluster][neuron])))
                else:
                    alignments.append(0)
            data = {
                'neuron': neurons,
                'iou': alignments
            }
            df = pd.DataFrame(data)
            os.makedirs(f"{save_dir}/Cluster{cluster}/", exist_ok=True)
            df.to_csv(f"{save_dir}/Cluster{cluster}/{i}Iter_{cluster}Cluster_Alignment.csv")

def calculate_alignment_with_random(all_fm_masks,random, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    random_masks = random[0]
    if isinstance(random_masks,list):
        random_masks=random_masks[0]
    for i, mask_dict in enumerate(all_fm_masks):
        if isinstance(mask_dict,list):
            mask_dict=mask_dict[0]
        for cluster, pair in mask_dict.items():
            neurons,alignments=[],[]

            for neuron, mask in pair.items():
                neurons.append(neuron)
                #print(random_masks[cluster].keys(), neuron)
                if neuron in random_masks[cluster].keys():
                    #print(f"Comparing mask {mask} with {initial_masks[cluster][neuron]}")
                    alignments.append(metrics.iou(torch.tensor(mask), torch.tensor(random_masks[cluster][neuron])))
                else:
                    alignments.append(0)
            data = {
                'neuron': neurons,
                'iou': alignments
            }
            df = pd.DataFrame(data)
            os.makedirs(f"{save_dir}/Cluster{cluster}/", exist_ok=True)
            df.to_csv(f"{save_dir}/Cluster{cluster}/{i+1}Iter_{cluster}Cluster_Alignment.csv")
        
#========testing alignment code=====
fm_masks=[]
flder="/workspace/CCE_NLI/BERT/formula_masks/wanda/Run0.25"
import json,os
for file in sorted(os.listdir(flder)):
    if '.ipy' in file: continue
    with open(os.path.join(flder, file), 'r') as f:
        data = json.load(f)
    fm_masks.append(data)
    
'''    
random_masks = [] 
flder = "/workspace/CCE_NLI/LLAMA/formula_masks/PretrainedWeights"
for file in sorted(os.listdir(flder)):
    if '.ipy' in file: continue
    with open(os.path.join(flder, file), 'r') as f:
        data = json.load(f)
    random_masks.append(data)'''
print(fm_masks[0].keys())
#calculate_alignment_with_random(fm_masks, random_masks, "/workspace/CCE_NLI/BERT/overlap/lottery_ticket/overlap_w_pretrained/Run2Full")
calculate_alignment_with_original(fm_masks, "/workspace/CCE_NLI/BERT/overlap/wanda/Run0.25")
    
    
        