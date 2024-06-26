from cleaning import prep, store_best_exp
from concept_analysis import concept_similarity, Union, calculate_similarity_across_explanations, count_ANDOR, sum_andor
from concept_getters import get_indiv_concepts, get_grouped_concepts, get_grouped_concepts_per_unit, get_all_grouped_cps, get_lost_concepts, get_preserved_concepts, find_neurons_explaining, get_common_concepts_explained_neurons, get_avg_iou, get_common_neurons

def main():
    clus_1_5 = prep("Cluster1IOUS5%.csv")
    clus_2_5 = prep("Cluster2IOUS5%.csv")
    clus_3_5 = prep("Cluster3IOUS5%.csv")
    cluster_1_iou = get_avg_iou(clus_1_5.best_iou)
    cluster_2_iou = get_avg_iou(clus_2_5.best_iou)
    cluster_3_iou = get_avg_iou(clus_3_5.best_iou)
    print(f"Cluster 1 avg iou: {cluster_1_iou}\nCluster 2 avg iou: {cluster_2_iou}\nCluster 3 avg iou: {cluster_3_iou}")
main()