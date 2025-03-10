import concept_analysis 
from concept_getters import get_lost_concepts, get_avg_iou, get_indiv_concepts, get_all_grouped_cps, get_new_concepts
import csv
import pandas as pd
import utils
    
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
        d[f'Cluster{i}'] = utils.intersection(pruned_row,nopruned_row)/utils.union(pruned_row,nopruned_row)
    return d


def record_lost_concepts(nonpruned_dict: dict, pruned_dict : dict, fname=None, as_percent = False) -> dict:
    lost_cps = {}
    i = 0
    
    for p,np in zip(pruned_dict.values(), nonpruned_dict.values()):
        i += 1
        lost_from_orig = get_lost_concepts(np, p)
        if as_percent:
            lost_cps[f'Cluster{i}'] = len(lost_from_orig) / len(np)
        else:
            lost_cps[f'Cluster{i}'] = lost_from_orig 
            
    if fname != None:
        utils.save_to_csv(lost_cps, fname)
    return lost_cps



        
def record_new_concepts(orig_dict, pruned_dict, fname=None):
    new_cps={}
    i=0
    for p_keys,np_keys in zip(pruned_dict.keys(), orig_dict.keys()):
        i+=1
        p = pruned_dict[p_keys]
        np= orig_dict[np_keys]
        new_cps[f'Cluster{i}']=get_new_concepts(np, p)
    if fname != None:
        utils.save_to_csv(new_cps,fname)
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

#only func that handles glob and loc
def record_common_concepts(concepts : list, fname=None):
    common = {}
    cluster = 0
    for pruned, original in zip(concepts[0], concepts[1]):
        cluster += 1
        common[f'Cluster{cluster}'] = concept_analysis.calculate_similarity_across_explanations(pruned, original).to_dict()
        
    if fname != None:
        dfs = []
        for cluster in common.keys():
        # Create a DataFrame from the dictionary
            df1 = pd.DataFrame(common[cluster])

            # Add a column for the cluster
            df1['cluster'] = cluster
            dfs.append(df1)

        # Concatenate the DataFrames
        df = pd.concat(dfs)

        # Pivot the DataFrame to get the desired format
        pivot_df = df.pivot_table(index='unit', columns='cluster', values='sim', aggfunc='sum')

        pivot_df.to_csv(fname)
    return common




