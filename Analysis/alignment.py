import sys
sys.path.append('code/')
import metrics, random, torch,os
import pandas as pd
def calculate_alignment(all_fm_masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    neurons,alignments=[],[]
    initial_masks = all_fm_masks[0] #formula masks for without pruning
    for i, fm_mask in enumerate(all_fm_masks[1:]):
        for neuron,mask in fm_mask.items():
            neurons.append(neuron)
            if neuron in initial_masks.keys():
                alignments.append(metrics.iou(torch.tensor(mask), torch.tensor(initial_masks[neuron])))
            else:
                alignments.append(0)
        data = {
            'neuron': neurons,
            'iou': alignments
        }
        df = pd.DataFrame(data)
        df.to_csv(f"{save_dir}/{i+1}_Pruning_Iter_Alignment.csv")

        
#========testng alignment code=====
neurons = range(0,10)

fm_masks=[]
for i in range(3):
    fm_mask = {}
    for neuron in neurons:
        fm_mask[neuron] = [random.randint(0, 1) for _ in range(5)]
    fm_masks.append(fm_mask)
print(fm_masks)

calculate_alignment(fm_masks, "Test")


    
    
    
        