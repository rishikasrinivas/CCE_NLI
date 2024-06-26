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
    
def record_cluster_level_sim(pruned, noprune):
    i = 0
    d = {}
    for p,np in zip(pruned, noprune):
        i += 1
        pruned_row = []
        nopruned_row =[]
        # creating a csv writer object

        pruned_cps =get_indiv_concepts(p)
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
        
    
def main():
    clus_1_5 = prep("Cluster1IOUS5%.csv")
    clus_2_5 = prep("Cluster2IOUS5%.csv")
    clus_3_5 = prep("Cluster3IOUS5%.csv")
    clus_4_5 = prep("Cluster4IOUS5%.csv")
    
    clus_1_np = prep("Cluster1IOUSOrig.csv")
    clus_2_np = prep("Cluster2IOUSOrig.csv")
    clus_3_np = prep("Cluster3IOUSOrig.csv")
    clus_4_np = prep("Cluster4IOUSOrig.csv")
    
    pruned_clusters = [clus_1_5,clus_2_5,clus_3_5,clus_4_5]
    nopruned_clusters = [clus_1_np,clus_2_np,clus_3_np,clus_4_np]
    #record_avg_ious(pruned_clusters,nopruned_clusters)
    
    #record_similarity(pruned_clusters,nopruned_clusters)
    
    d = record_cluster_level_sim(pruned_clusters, nopruned_clusters)
    print(d)
    
main()