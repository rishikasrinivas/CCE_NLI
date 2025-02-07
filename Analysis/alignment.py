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
                df.to_csv(f"{save_dir}/{i+1}Iter_{cluster}Cluster_Alignment.csv")

        
#========testing alignment code=====
for 
'''fm_masks = []
for l in range(2):
    mask_dict = {}
    for i in range(3):
        mask_dict[i+1] = {j: [random.randint(0,1) for _ in range(5)] for j in range(3)}
    fm_masks.append([mask_dict, {}])
print(fm_masks[0])
print(fm_masks[1])
calculate_alignment(fm_masks, "Testing6")'''


    
    
    
        