import sys
sys.path.append('code/')
import metrics, random, torch,os
import pandas as pd
#{clus1: {neuon: mask}}
def calculate_alignment(all_fm_masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    initial_masks = all_fm_masks[0][0] #formula masks for without pruning

    for i, fm_mask in enumerate(all_fm_masks[1:]):
        for mask_dict in fm_mask:
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
                df.to_csv(f"{save_dir}/Cluster{cluster}/{i+1}Iter_{cluster}Cluster_Alignment.csv")

        
#========testing alignment code=====
fm_masks=[]
import json,os
for file in sorted(os.listdir("BOWMAN/formula_masks/bowman/lottery_ticket/Run2FULL")):
    if '.ipy' in file: continue
    with open (f"BOWMAN/formula_masks/bowman/lottery_ticket/Run2FULL/{file}", 'r') as f:
        d=json.load(f)
    fm_masks.append(d)
calculate_alignment(fm_masks, "BOWMAN/overlap/bowman/lottery_ticket/Run2Full")

    
    
    
        