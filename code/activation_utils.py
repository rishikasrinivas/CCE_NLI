#added
import numpy as np
import torch
import sklearn.cluster as scikit_cluster
count=0
def build_act_mask(states, activ_ranges, cluster_num):

    #here you're looking at the activations (its called on the states dk why says feats here) and if theyre greater than 0 it goes into the mask as true else false 

    if cluster_num > len(activ_ranges):
        cluster_num = len(activ_ranges)
    #change this to do a binary map within a range
    lower_thresh_in_range=activ_ranges[cluster_num-1][0]
    upper_thresh_in_range = activ_ranges[cluster_num-1][1]
    act_masks = torch.where((states > lower_thresh_in_range) & (states < upper_thresh_in_range), True, False)

    #print("act_masks_per_range: ", act_masks)

    return act_masks.reshape(1, 1024) #returns 1x1024 binary map saying which neurons activates (true if neuron a col does else false)

def compute_activ_ranges(activations, clusters, num_clusters):
    #all activations will be >0 here
    activation_ranges=[]
    for label in range(num_clusters):
        min_in_range = torch.min(activations[clusters==label])
        max_in_range = torch.max(activations[clusters==label])
        activation_ranges.append([min_in_range.item(),max_in_range.item()])
    return sorted(activation_ranges, key=lambda x:x[0])

def create_clusters(activations, num_clusters):
    if torch.nonzero(activations).size(0) < num_clusters:
        return  [(0, torch.tensor(float("inf")))]
    clusters = scikit_cluster.KMeans(n_clusters= num_clusters, random_state=0).fit(activations)
    cluster_lst = clusters.labels_
    
    activation_ranges=compute_activ_ranges(activations, cluster_lst, num_clusters)
 
    return activation_ranges

    
def build_masks(activations, num_clusters, cluster_num):
    act_masks=[]
    count = 0
    for i, activ_for_sample in enumerate(activations):
        #if torch.all(activ_for_sample >= 0):
            #activ_for_sample=activ_for_sample[activ_for_sample>0]

        activ_for_sample = activ_for_sample.reshape(-1,1)
        activation_ranges = create_clusters(activ_for_sample,num_clusters)
        mask=build_act_mask(activ_for_sample.squeeze(),activation_ranges, cluster_num)
        act_masks.append(mask)
    print(torch.stack(act_masks), torch.stack(act_masks).shape, )

    return torch.stack(act_masks).permute(1,0,2).squeeze().numpy()
    
        
    