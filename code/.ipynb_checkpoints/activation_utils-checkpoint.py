#added
import numpy as np
import torch
import sklearn.cluster as scikit_cluster
def build_act_mask(states, activ_ranges):
    print(activ_ranges)
    #here you're looking at the activations (its called on the states dk why says feats here) and if theyre greater than 0 it goes into the mask as true else false 
    #change this to do a binary map within a range
    lower_thresh_in_range=activ_ranges[0]
    upper_thresh_in_range = activ_ranges[1]
    act_masks = torch.where((states > lower_thresh_in_range) & (states < upper_thresh_in_range, True, False))
        
    #print("act_masks_per_range: ", act_masks)
    return act_masks #returns binary map saying which neurons activates (true if neuron a col does else false)

def compute_activ_ranges(activations, clusters, num_clusters):
    #all activations will be >0 here
    activation_ranges=[]
    for label in range(num_clusters):
        min_in_range = torch.min(activations[clusters==label])
        max_in_range = torch.max(activations[clusters==label])
        activation_ranges.append([min_in_range.item(),max_in_range.item()])
    return activation_ranges

def create_clusters(activations, num_clusters):
    if torch.all(activations >= 0):
        activations=activations[activations>0]
    print(activations[0])
    activations = activations.reshape(-1,1)
    print(activations, activations.shape)
    if num_clusters == 1:
        return  [(threshold, torch.tensor(float("inf")))]
    
    clusters = scikit_cluster.KMeans(n_clusters= num_clusters, random_state=0).fit(activations)
    cluster_lst = clusters.labels_
    activation_ranges=compute_activ_ranges(activations, cluster_lst, num_clusters)
    return activation_ranges