from concept_analysis import concept_similarity, Union, calculate_similarity_across_explanations, count_ANDOR, sum_andor
from concept_getters import get_lost_concepts, get_avg_iou, get_indiv_concepts, get_all_grouped_cps, get_new_concepts
import csv
import pandas as pd


def save_to_csv(dictionary, fname):
    ml=0
    for k,v in dictionary.items():
        if len(v)>ml:
            ml=len(v)
    for k,v in dictionary.items():
        dictionary[k]=list(dictionary[k])
        while len(dictionary[k])!=ml:
            dictionary[k].append('')
    pd.DataFrame(dictionary).to_csv(fname)
    
def intersection(lst1, lst2):
    intersect=0
    for i,j in zip(lst1,lst2):
        if i==j:
            intersect += 1
    return intersect

def union(lst1,lst2):
    union=0
    for i,j in zip(lst1,lst2):
        if i==1 or j==1:
            union += 1
    return union

 
    
def record_avg_ious(pruned: list, noprune: list):
    pruned_avg_ious=[]
    nopruned_avg_ious=[]
    
    for clustera, clusterb in zip(pruned, noprune):
        pruned_avg_ious.append( get_avg_iou(clustera.best_iou) )
        nopruned_avg_ious.append( get_avg_iou(clusterb.best_iou) )
        
def record_neuron_level_similarity(pruned_clusters,nopruned_clusters):
    i=0
    for p,np in zip(pruned_clusters,nopruned_clusters):
        i+=1
        calculate_similarity_across_explanations(p, np).to_csv(f"Analysis/similarity_cluster{i}.csv")
    
def record_cluster_level_sim(pruned, noprune, grouped_cps):
    i = 0
    d = {}
    for p,np in zip(pruned, noprune):
        i += 1
        pruned_row = []
        nopruned_row =[]
        # creating a csv writer object
        if grouped_cps:
            pruned_cps = get_all_grouped_cps (p)
            nopruned_cps = get_all_grouped_cps (np)
        else:
            pruned_cps = get_indiv_concepts(p)
            nopruned_cps = get_indiv_concepts(np)
    
        # writing the fields
        all_cps_in_cluster = pruned_cps.union(nopruned_cps)
        for cp in all_cps_in_cluster:
            if cp in pruned_cps:
                pruned_row.append(1)
            else:
                pruned_row.append(0)

            if cp in nopruned_cps:
                nopruned_row.append(1)
            else:
                nopruned_row.append(0)
        d[f'Cluster{i}'] = intersection(pruned_row,nopruned_row)/union(pruned_row,nopruned_row)
    return d


def record_lost_concepts(nonpruned_dict: dict, pruned_dict : dict, fname=None) -> dict:
    lost_cps = {}
    i = 0
    
    for p,np in zip(pruned_dict.values(), nonpruned_dict.values()):
        i += 1
        lost_from_orig = get_lost_concepts(np, p)
        lost_cps[f'Cluster{i}'] = lost_from_orig
    if fname != None:
        save_to_csv(lost_cps, fname)
    return lost_cps


def record_across_concepts(orig_dict, retrained_pruned_dict, noretrain_prune_dict, task, fname=None):
    concepts_lost_after_pruning_wo_rt = record_lost_concepts(orig_dict,noretrain_prune_dict)
    recordings = {}
    #find concepts lost after pruning in retrrained
    for cluster in retrained_pruned_dict.keys():
        cps_in_retrained = retrained_pruned_dict[cluster]
        #retrained cps set and lost cps set find the union
        lost_cps =concepts_lost_after_pruning_wo_rt[cluster]
        if task == 'relearned':
            task_concepts = cps_in_retrained.intersection(lost_cps)
        else:
            task_concepts = lost_cps.difference(cps_in_retrained)
        recordings[cluster] = task_concepts
    if fname != None:
        save_to_csv(recordings, fname)
    return recordings

        
def record_new_concepts(orig_dict, pruned_dict, fname=None):
    new_cps={}
    i=0
    for p_keys,np_keys in zip(pruned_dict.keys(), orig_dict.keys()):
        i+=1
        p = pruned_dict[p_keys]
        np= orig_dict[np_keys]
        new_cps[f'Cluster{i}']=get_new_concepts(np, p)
    if fname != None:
        save_to_csv(new_cps,fname)
    return new_cps
    

def record_retained_concepts(pdicts, npdicts):
    retained = {}
    i = 0
    for p, np in zip(pdicts.keys(), npdicts.keys()):
        i += 1
        p = pdicts[p]
        np = npdicts[np]
        retained[f'Cluster{i}'] = p.intersection(np)
    return retained

def record_common_concepts(*dicts, task, fname=None):
    if task == 'across_clusters':
        i = 0
        for k,v in dicts[0].items():
            i += 1
            if i == 1:
                common = v
            else:
                common = common.intersection(v)
        common = {"Concepts common to all clusters": common}
    elif task == 'across_prune_runs': #accros pruning
        i = 0
        common = {}
        for d1, d2, d3 in zip(dicts[0].values(), dicts[1].values(), dicts[2].values()):
            i+=1
            common[f"Cluster{i}"] = (d1.intersection(d2)).intersection(d3)
    else:
        raise TypeError("Invalid task argument")
        return
    
    if fname != None:
        save_to_csv(common,fname)
    return common


