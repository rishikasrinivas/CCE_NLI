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
    
    act_masks = torch.where((states >= lower_thresh_in_range) & (states <= upper_thresh_in_range) & (states > 0), True, False)

    #print("act_masks_per_range: ", act_masks)
    act_masks=act_masks.reshape(shape)
    return act_masks #returns 1024 x1binary map saying which neurons activates (true if neuron a col does else false)



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
    nonzero_activs_num= torch.nonzero(activations).size(0)
    activations  = activations.flatten().reshape(-1, 1)
    
    if nonzero_activs_num < num_clusters:
        num_clusters = nonzero_activs_num
        
    clusters = scikit_cluster.KMeans(n_clusters= num_clusters, random_state=0).fit(activations)
    cluster_lst = clusters.labels_
    activation_ranges = compute_activ_ranges(activations, cluster_lst, num_clusters)
    
    return activation_ranges

def get_avgs(all_act_rags):
    data_array = np.array(all_act_rags)

    # Calculate the average along the first axis (across all lists for each index)
    averages = np.mean(data_array, axis=0)

    # Convert the result back to a list if needed
    averages_list = averages.tolist()
    return averages_list
        
        
    return start/lars, end/lars
def build_masks(activations, num_clusters, cluster_num):
    act_masks=[]
    all_act_rags=[]
    activations=torch.Tensor(activations)
    for i, activ_for_sample in enumerate(activations):
        #if torch.all(activ_for_sample >= 0):
            #activ_for_sample=activ_for_sample[activ_for_sample>0]

        activ_for_sample = activ_for_sample.reshape(-1,1)
        activation_ranges = create_clusters(activ_for_sample,num_clusters)
        all_act_rags.append(activation_ranges)
        mask=build_act_mask(activ_for_sample.squeeze(),activation_ranges, cluster_num)
        torch.save(mask, f"code/Masks/Cluster{cluster_num}/SentPair{i}sMask.pt")
        act_masks.append(mask)
    print(torch.stack(act_masks).shape )
    print(get_avgs(all_act_rags))
    act_tens=torch.save(torch.stack(act_masks), f"code/Masks/Cluster{cluster_num}masks.pt")
    return torch.stack(act_masks).numpy()
        
    