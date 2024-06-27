from cleaning import prep, store_best_exp
from concept_analysis import concept_similarity, Union, calculate_similarity_across_explanations, count_ANDOR, sum_andor
from concept_getters import get_indiv_concepts, get_grouped_concepts, get_grouped_concepts_per_unit, get_all_grouped_cps, get_lost_concepts, get_preserved_concepts, find_neurons_explaining, get_common_concepts_explained_neurons, get_avg_iou, get_common_neurons
import csv


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


def record_lost_concepts(pruned_dfs : list, nonpruned_dfs: list) -> dict:
    d = {}
    num_lost = {}
    i = 0
    for p,np in zip(pruned_dfs, nonpruned_dfs):
        i += 1
        pruned_cps = get_indiv_concepts (p)
        nopruned_cps = get_indiv_concepts (np)
        lost_from_orig = get_lost_concepts(nopruned_cps, pruned_cps)
        d[f'Cluster{i}'] = lost_from_orig
        num_lost[f'Cluster{i}'] = len(lost_from_orig)
    print(f"Number of lost concepts: {num_lost}")
    return d

def record_relearned_concepts(orig_dfs, retrained_pruned_dfs, noretrain_prune_dfs):
    concepts_lost_after_pruning_wo_rt = record_lost_concepts(orig_dfs,noretrain_prune_dfs)
    relearned_per_clus ={}
    num_of_relearned_concepts = {}
    #find concepts lost after pruning in retrrained
    for i,retrained_df in enumerate(retrained_pruned_dfs):
        grouped_cps_in_retrained = get_indiv_concepts(retrained_df)
        #retrained cps set and lost cps set find the union
        lost_cps =concepts_lost_after_pruning_wo_rt[f'Cluster{i+1}']
        relearned_concepts = grouped_cps_in_retrained.intersection(lost_cps)
        relearned_per_clus[f'Cluster{i+1}'] = relearned_concepts
        
        num_of_relearned_concepts[f'Cluster{i+1}'] = len(relearned_concepts)
    print(f"Number of relearned concepts: {num_of_relearned_concepts}")
    return relearned_per_clus

def record_lost_complete(orig_dfs, retrained_pruned_dfs, noretrain_prune_dfs):
    concepts_lost_after_pruning_wo_rt = record_lost_concepts(orig_dfs,noretrain_prune_dfs)
    completely_lost = {}
    num_completely_lost = {}
    for i,retrained_df in enumerate(retrained_pruned_dfs):
        grouped_cps_in_retrained = get_indiv_concepts(retrained_df)
        print(len(grouped_cps_in_retrained))
        lost_cps =concepts_lost_after_pruning_wo_rt[f'Cluster{i+1}']
        lost_completely = lost_cps.difference(grouped_cps_in_retrained)
        
        print(f"Preserved in Cluster{i+1}: {len(grouped_cps_in_retrained.difference(lost_cps))}")
        completely_lost[f'Cluster{i+1}'] = lost_completely
        num_completely_lost[f'Cluster{i+1}'] = len(lost_completely)
    print(f"Number of completely lost concepts: {num_completely_lost}")
    return completely_lost
        
        
    

def main():
    clus_1_5 = prep("Cluster1IOUS5%.csv")
    clus_2_5 = prep("Cluster2IOUS5%.csv")
    clus_3_5 = prep("Cluster3IOUS5%.csv")
    clus_4_5 = prep("Cluster4IOUS5%.csv")
    
    clus_1_np = prep("Cluster1IOUSOrig.csv")
    clus_2_np = prep("Cluster2IOUSOrig.csv")
    clus_3_np = prep("Cluster3IOUSOrig.csv")
    clus_4_np = prep("Cluster4IOUSOrig.csv")
    
    clus_1_pwr = prep("Cluster1IOUSPruneWoRetrain.csv")
    clus_2_pwr = prep("Cluster2IOUSPruneWoRetrain.csv")
    clus_3_pwr =prep("Cluster3IOUSPruneWoRetrain.csv")
    clus_4_pwr = prep("Cluster4IOUSPruneWoRetrain.csv")
    
    
    pruned_clusters = [clus_1_5,clus_2_5,clus_3_5,clus_4_5]
    nopruned_clusters = [clus_1_np,clus_2_np,clus_3_np,clus_4_np]
    pwr_clusters = [clus_1_pwr,clus_2_pwr,clus_3_pwr,clus_4_pwr ]
    #record_avg_ious(pruned_clusters,nopruned_clusters)
    
    #record_similarity(pruned_clusters,nopruned_clusters)
    
    iou_indiv_cps = record_cluster_level_sim(pruned_clusters, nopruned_clusters, grouped_cps=False)
    iou_grouped_cps = record_cluster_level_sim(pruned_clusters, nopruned_clusters, grouped_cps=True)
    print(f"IOU for individual concepts: {iou_indiv_cps}\nIOU of groups of concepts: {iou_grouped_cps}")
    
    
    #print("Num of lost concepts: ", record_lost_concepts(pwr_clusters, nopruned_clusters))
    
    relearned = record_relearned_concepts(nopruned_clusters, pruned_clusters, pwr_clusters)

    
    lost = record_lost_complete(nopruned_clusters, pruned_clusters, pwr_clusters)
    
main()