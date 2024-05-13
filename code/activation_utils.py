#added
import numpy as np
import torch
import sklearn.cluster as scikit_cluster
count=0
def build_act_mask(states, activ_ranges, cluster_num):
    shape=states.shape
    states=states.flatten()
    if cluster_num > len(activ_ranges):
        cluster_num = len(activ_ranges)
    #change this to do a binary map within a range
    lower_thresh_in_range=activ_ranges[cluster_num-1][0]
    upper_thresh_in_range = activ_ranges[cluster_num-1][1] #same
    print("[", lower_thresh_in_range, ", ", upper_thresh_in_range, "]")
    
    act_masks = torch.where((states >= lower_thresh_in_range) & (states <= upper_thresh_in_range) & (states > 0), True, False)

    #print("act_masks_per_range: ", act_masks)
    act_masks=act_masks.reshape(shape)
    return act_masks.numpy() #returns 1024 x1binary map saying which neurons activates (true if neuron a col does else false)

def active_neurons(activations): #activs should be 10,000x1024
    active_neurons = []
    activations = activations.transpose()
    for i,activs in enumerate(activations):
        if len(np.where(activs==1)[0])  > 0:
            active_neurons.append(i+1)
    
    return active_neurons

def compute_activ_ranges(activations, clusters, num_clusters):
    #all activations will be >0 here
    activation_ranges=[]
    for label in range(num_clusters):
        inds = np.where(clusters == label)

        min_in_range = activations[inds].min()
        max_in_range = activations[inds].max()
        print([min_in_range.item(),max_in_range.item()])
        activation_ranges.append([min_in_range.item(),max_in_range.item()])
    return sorted(activation_ranges, key=lambda x:x[0])

def create_clusters(activations, num_clusters):
    if activations.requires_grad:
        activations=activations.detach()
    if torch.nonzero(activations).size(0) < num_clusters:
        return  [(0, torch.tensor(float("inf")))]

    activations  = activations.flatten().reshape(-1, 1)
    print(activations.shape)
    print("goinh to cluster")
    clusters = scikit_cluster.KMeans(n_clusters= num_clusters, random_state=0).fit(activations)
    cluster_lst = clusters.labels_
    print("fin to cluster")
    activation_ranges = compute_activ_ranges(activations, cluster_lst, num_clusters)
 
    return activation_ranges

    

    
    
'''
def store_activ_ranges(activations, num_clusters, cluster_num):
    active_ranges=[]
    
    for i, activ_for_sample in enumerate(activations):

        activ_for_sample = activ_for_sample.reshape(-1,1)
        activation_ranges = create_clusters(activ_for_sample,num_clusters)
        active_ranges.append(activation_ranges)

    return torch.stack(active_ranges).numpy()

in main
    for each cluster_num
        mask=build_act_mask(activations,activation_ranges, cluster_num) lower thresh, act_masks would be in loop
        searh...
'''
    
        
    