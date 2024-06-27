from cleaning import prep, store_best_exp
from concept_analysis import concept_similarity, Union, calculate_similarity_across_explanations, count_ANDOR, sum_andor
from concept_getters import get_indiv_concepts, get_grouped_concepts, get_grouped_concepts_per_unit, get_all_grouped_cps, get_lost_concepts, get_preserved_concepts, find_neurons_explaining, get_common_concepts_explained_neurons, get_avg_iou, get_common_neurons, get_new_concepts
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
    lst3 = [value for value in lst1 if (value == 1 and value in lst2)]
    return lst3
 
    
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
        d[f'Cluster{i}'] = len(intersection(pruned_row,nopruned_row))/len(nopruned_row)
    return d


def record_lost_concepts(nonpruned_dfs: list, pruned_dfs : list, concept_retrieve_func, fname=None) -> dict:
    lost_cps = {}
    i = 0
    for p,np in zip(pruned_dfs, nonpruned_dfs):
        i += 1
        pruned_cps = concept_retrieve_func (p)
        
        nopruned_cps = concept_retrieve_func (np)
        lost_from_orig = get_lost_concepts(nopruned_cps, pruned_cps)
        lost_cps[f'Cluster{i}'] = lost_from_orig
    if fname != None:
        save_to_csv(lost_cps, fname)
    return lost_cps

def record_relearned_concepts(orig_dfs, retrained_pruned_dfs, noretrain_prune_dfs, concept_retrieve_func, fname=None):
    concepts_lost_after_pruning_wo_rt = record_lost_concepts(orig_dfs,noretrain_prune_dfs, concept_retrieve_func)
    relearned_per_clus ={}
    #find concepts lost after pruning in retrrained
    for i,retrained_df in enumerate(retrained_pruned_dfs):
        cps_in_retrained = concept_retrieve_func(retrained_df)
        #retrained cps set and lost cps set find the union
        lost_cps =concepts_lost_after_pruning_wo_rt[f'Cluster{i+1}']
        relearned_concepts = cps_in_retrained.intersection(lost_cps)
        relearned_per_clus[f'Cluster{i+1}'] = relearned_concepts
    if fname != None:
        save_to_csv(relearned_per_clus, fname)
    return relearned_per_clus

def record_lost_completely(orig_dfs, retrained_pruned_dfs, noretrain_prune_dfs, concept_retrieve_func,fname = None):
    concepts_lost_after_pruning_wo_rt = record_lost_concepts(orig_dfs,noretrain_prune_dfs, concept_retrieve_func)
    completely_lost = {}
    for i,retrained_df in enumerate(retrained_pruned_dfs):
        cps_in_retrained = concept_retrieve_func(retrained_df)
        lost_cps =concepts_lost_after_pruning_wo_rt[f'Cluster{i+1}']
        lost_completely = lost_cps.difference(cps_in_retrained)
        
        completely_lost[f'Cluster{i+1}'] = lost_completely
    if fname != None:
        save_to_csv(completely_lost, fname)
    return completely_lost
        
def record_new_concepts(orig_dfs, pruned_dfs, concept_retrieve_func, fname=None):
    new_cps={}
    i=0
    for p,np in zip(pruned_dfs,orig_dfs ):
        i+=1
        p = concept_retrieve_func(p)
        np=concept_retrieve_func(np)
        new_cps[f'Cluster{i}']=get_new_concepts(np, p)
    if fname != None:
        save_to_csv(new_cps,fname)
    return new_cps
    
# find %of concepts lost from initislly learned when 1st pruned  = num concepts lost/union(pruned no rt, orig)
def percent_concepts_lost(lost, pruned_dfs, nonpruned_dfs, concept_retrieve_func):
    i=0
    percent_lost_per_clus = {}
    for p,np in zip(pruned_dfs, nonpruned_dfs):
        i += 1
        pruned_cps = concept_retrieve_func (p)
        nopruned_cps = concept_retrieve_func (np)
        
        all_cps_in_cluster = pruned_cps.union(nopruned_cps)
        percent_lost =len(lost[f"Cluster{i}"]) / len(all_cps_in_cluster)
        percent_lost_per_clus[f"Cluster{i}"] = percent_lost
    return percent_lost_per_clus

# %of concepts relearned of the lost
def percent_overlap(relearned, lost):
    i=0
    percent_relearned_per_clus = {}
    for rl_clus, l_clus in zip(relearned.values(), lost.values()):
        i+=1
        percent_rl = len(rl_clus)/len(l_clus)
        percent_relearned_per_clus[f"Cluster{i}"]=percent_rl
    return percent_relearned_per_clus

# % of new concepts
def percent_of_new_cps(new_cps, pruned_cps):
    assert type(pruned_cps) == dict
    percent_of_new_concps={}
    i=0
    for new, ref in zip(new_cps.values(), pruned_cps.values()):
        i += 1
        assert len(set(new).intersection(set(ref))) == len(new)
        percent_of_new_concps[f"Cluster{i}"]=len(new)/len(ref)
    
    return percent_of_new_concps

def record_retained_concepts(pdfs, npdfs, concept_retrieve_func):
    retained = {}
    i = 0
    for p, np in zip(pdfs, npdfs):
        i += 1
        p = concept_retrieve_func(p)
        np = concept_retrieve_func(np)
        retained[f'Cluster{i}'] = p.intersection(np)
    return retained

def record_common_concepts(dfs1, dfs2, dfs3, concept_ret, fname=None):
    common = {}
    i=0
    for df1, df2, df3 in zip(dfs1, dfs2, dfs3):
        i+=1
        df1 = concept_ret(df1)
        df2 = concept_ret(df2)
        df3 = concept_ret(df3)
        
        common[f"Cluster{i}"] = (df1.intersection(df2)).intersection(df3)
        
    if fname != None:
        save_to_csv(common,fname)
    return common

def main():
    concept_retrieve_func=get_all_grouped_cps #get_indiv_concepts#
    clus_1_5 = prep("Analysis/Explanations/Cluster1IOUS5%.csv")
    clus_2_5 = prep("Analysis/Explanations/Cluster2IOUS5%.csv")
    clus_3_5 = prep("Analysis/Explanations/Cluster3IOUS5%.csv")
    clus_4_5 = prep("Analysis/Explanations/Cluster4IOUS5%.csv")
    
    clus_1_np = prep("Analysis/Explanations/Cluster1IOUSOrig.csv")
    clus_2_np = prep("Analysis/Explanations/Cluster2IOUSOrig.csv")
    clus_3_np = prep("Analysis/Explanations/Cluster3IOUSOrig.csv")
    clus_4_np = prep("Analysis/Explanations/Cluster4IOUSOrig.csv")
    
    clus_1_pwr = prep("Analysis/Explanations/Cluster1IOUSPruneWoRetrain.csv")
    clus_2_pwr = prep("Analysis/Explanations/Cluster2IOUSPruneWoRetrain.csv")
    clus_3_pwr =prep("Analysis/Explanations/Cluster3IOUSPruneWoRetrain.csv")
    clus_4_pwr = prep("Analysis/Explanations/Cluster4IOUSPruneWoRetrain.csv")
    
    
    pruned_clusters = [clus_1_5,clus_2_5,clus_3_5,clus_4_5]
    nopruned_clusters = [clus_1_np,clus_2_np,clus_3_np,clus_4_np]
    pwor_clusters = [clus_1_pwr,clus_2_pwr,clus_3_pwr,clus_4_pwr ] #pruned w/o retraining
    #record_avg_ious(pruned_clusters,nopruned_clusters)
    
    #record_similarity(pruned_clusters,nopruned_clusters)
    
    iou_indiv_cps = record_cluster_level_sim(pruned_clusters, nopruned_clusters, grouped_cps=False)
    iou_grouped_cps = record_cluster_level_sim(pruned_clusters, nopruned_clusters, grouped_cps=True)
    print(f"IOU for individual concepts: {iou_indiv_cps}\nIOU of groups of concepts: {iou_grouped_cps}")
    
    
    concepts_lost_after_pruning_beforeRT = record_lost_concepts(nopruned_clusters, pwor_clusters, concept_retrieve_func)
    concepts_retained_after_pruning_beforeRT = record_retained_concepts(pwor_clusters, nopruned_clusters, concept_retrieve_func)
    
    concepts_preserved_throughout = record_common_concepts(pruned_clusters, nopruned_clusters, pwor_clusters, concept_retrieve_func)
    
    for i,v in enumerate(concepts_preserved_throughout.values()):
        print(f"Cluster{i+1}: % of preserved from original explanations: {len(v)/len(concept_retrieve_func(nopruned_clusters[i]))}")
                                                                  
    relearned = record_relearned_concepts(nopruned_clusters, pruned_clusters, pwor_clusters, concept_retrieve_func)

    
    lost_completely = record_lost_completely(nopruned_clusters, pruned_clusters, pwor_clusters, concept_retrieve_func)
    
    percent_relearned_from_training = percent_overlap(relearned, concepts_lost_after_pruning_beforeRT)
    print("percent relearned from training: ", percent_relearned_from_training)
    
    percent_lost_cmplt = percent_overlap(lost_completely, concepts_lost_after_pruning_beforeRT)
    print("percent not relearned from training: ", percent_lost_cmplt)
    
    
    new_concepts_in_retrained = record_new_concepts(nopruned_clusters, pruned_clusters,concept_retrieve_func)
    
    new_concepts_wo_retraining = record_new_concepts(nopruned_clusters, pwor_clusters,concept_retrieve_func)
    
    
    
    
    
    
main()

# when u prune wo retran you retain some concepts and lost some. of the ones you retain, do the pruned expls have these?
# with the new ones, hiw many are new
# how many old the new onces propogate to the fully pruned


    
        
