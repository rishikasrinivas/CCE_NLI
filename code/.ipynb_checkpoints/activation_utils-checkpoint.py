#added
import numpy as np
import torch
import sklearn.cluster as scikit_cluster
import os
import settings
count=0
def build_act_mask(states, activ_ranges, cluster_num):
    shape=states.shape
    states=states.flatten()

    if len(activ_ranges) == 0:
        return torch.zeros(10000,dtype=torch.bool)
    #change this to do a binary map within a range
    lower_thresh_in_range=activ_ranges[cluster_num-1][0]
    upper_thresh_in_range = activ_ranges[cluster_num-1][1] #same
    
    act_masks = torch.where((states >= lower_thresh_in_range) & (states <= upper_thresh_in_range) & (states > 0), True, False)

    act_masks=act_masks.reshape(shape)
    return act_masks #returns 10000 x1binary map saying which samples activates (true if sample a col does else false)



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
        activation_ranges.append([min_in_range.item(),max_in_range.item()])
    
    return sorted(activation_ranges, key=lambda x:x[0])

def create_clusters(activations, num_clusters):
    if activations.requires_grad:
        activations=activations.detach()
    # ensure activations is the right shape 
    if activations.shape[0] == 10000 and activations.shape[1] == 1024:
        activations=activations.t()
    assert activations.shape[0] == 1024 and activations.shape[1] == 10000
    
    #clustering
    activation_ranges=[]
    dead_neurons=[]
    for i,neurons_acts in enumerate(activations):
        nonzero_activs_num= torch.nonzero(neurons_acts).size(0)
        
        if nonzero_activs_num < settings.MIN_ACTS:
            dead_neurons.append(i)
            activation_ranges.append([])
            continue
            
        neurons_acts = neurons_acts[neurons_acts>0]
        neurons_acts  = neurons_acts.reshape(1,-1).t()
        
        clusters = scikit_cluster.KMeans(n_clusters= num_clusters, random_state=0).fit(neurons_acts)
        cluster_lst = clusters.labels_
        
        activation_range = compute_activ_ranges(neurons_acts, cluster_lst, num_clusters)
        activation_ranges.append(activation_range)
 
       
    return activation_ranges, dead_neurons

def get_avgs(all_act_rags):
    data_array = np.array(all_act_rags)

    # Calculate the average along the first axis (across all lists for each index)
    averages = np.mean(data_array, axis=0)

    # Convert the result back to a list if needed
    averages_list = averages.tolist()
    return averages_list
        
        
    return start/lars, end/lars
def build_masks(activations, activation_ranges, num_clusters, save_dir):
    activations=torch.Tensor(activations)
    
            
  
    for cluster_num in range(1,num_clusters+1):
        act_masks=[]
        os.makedirs(f"{save_dir}/Cluster{cluster_num}/", exist_ok=True)
        for i, activ_for_neuron in enumerate(activations):
            #if torch.all(activ_for_sample >= 0):
                #activ_for_sample=activ_for_sample[activ_for_sample>0]
            mask=build_act_mask(activ_for_neuron.squeeze(),activation_ranges[i], cluster_num)
            act_masks.append(mask)

        masks = torch.stack(act_masks)
        act_tens=torch.save(masks, f"{save_dir}/Cluster{cluster_num}masks.pt")
    return masks.numpy()
        
    