from cleaning import prep, store_best_exp
from concept_analysis import concept_similarity, Union, calculate_similarity_across_explanations, count_ANDOR, sum_andor
from concept_getters import get_indiv_concepts, get_grouped_concepts, get_all_grouped_cps, get_lost_concepts, get_avg_iou, get_common_neurons, get_new_concepts, get_indiv_concepts_per_cluster, get_grouped_concepts_per_cluster
from record import record_cluster_level_sim, record_lost_concepts, record_new_concepts, record_retained_concepts, record_common_concepts, record_across_concepts
from stats import percent_overlap

def filter_invalid_compositions(all_pruned_concepts,all_nopruned_concepts,all_pwor_concepts):
    for df1, df2, df3 in zip(all_pruned_concepts.values(),all_nopruned_concepts.values(),all_pwor_concepts.values()):
        df1 = count_ANDOR(df1)
        df2 = count_ANDOR(df2)
        df3 = count_ANDOR(df3)
        
        for ands,ors in df1.values():
            if ands==0 and ors==0:
                pass
        
def main():
    concept_retrieve_funcs = [(get_indiv_concepts_per_cluster,get_indiv_concepts) , (get_grouped_concepts_per_cluster,get_all_grouped_cps)]
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
    
    for i,retrieval in enumerate(concept_retrieve_funcs):
        if i == 0:
            print("LOGGING for individual concepts")
        else:
            print("\n\nLOGGING for compositions")
        all_pruned_concepts=retrieval[0](pruned_clusters)
        all_nopruned_concepts=retrieval[0](nopruned_clusters)
        all_pwor_concepts=retrieval[0](pwor_clusters)
        
        #filter_invalid_compositions(all_pruned_concepts,all_nopruned_concepts,all_pwor_concepts)

    
        iou_indiv_cps = record_cluster_level_sim(pruned_clusters, nopruned_clusters, grouped_cps=False)
        iou_grouped_cps = record_cluster_level_sim(pruned_clusters, nopruned_clusters, grouped_cps=True)
        print(f"IOU for individual concepts: {iou_indiv_cps}\nIOU of groups of concepts: {iou_grouped_cps}")

    
        concepts_lost_after_pruning_beforeRT = record_lost_concepts(all_nopruned_concepts, all_pwor_concepts)
        concepts_retained_after_pruning_beforeRT = record_retained_concepts(all_pwor_concepts, all_nopruned_concepts)

        concepts_preserved_throughout = record_common_concepts(all_pruned_concepts, all_nopruned_concepts, all_pwor_concepts, task ='across_prune_runs')

        for j,(k,v) in enumerate(concepts_preserved_throughout.items()):
            union_of_cps = (all_pruned_concepts[k].union(all_nopruned_concepts[k])).union(all_pwor_concepts[k])
            print(f"Cluster{j+1}: % of all concepts preserved from original explanations: {len(v)/len(union_of_cps)}")

        relearned = record_across_concepts(all_nopruned_concepts, all_pruned_concepts, all_pwor_concepts, 'relearned')


        lost_completely = record_across_concepts(all_nopruned_concepts, all_pruned_concepts, all_pwor_concepts, 'lost')

        percent_relearned_from_training = percent_overlap(relearned, concepts_lost_after_pruning_beforeRT)
        print("percent relearned from training: ", percent_relearned_from_training)

        percent_lost_cmplt = percent_overlap(lost_completely, concepts_lost_after_pruning_beforeRT)
        print("percent not relearned from training: ", percent_lost_cmplt)


        new_concepts_in_retrained = record_new_concepts(all_nopruned_concepts, all_pruned_concepts)

        new_concepts_wo_retraining = record_new_concepts(all_nopruned_concepts, all_pwor_concepts)
      
    
        concepts_common_to_all_clusters_inOrig = record_common_concepts(all_nopruned_concepts, task='across_clusters')
        print("concepts_common_to_all_clusters_inOrig: ", len(concepts_common_to_all_clusters_inOrig["Concepts common to all clusters"]))
        
        if i == 0:
            fname = "Analysis/IndivConceptAnalysis/concepts_common_to_all_clusters_inPruned.csv"
        else:
            fname = "Analysis/CompositionalConceptAnalysis/concepts_common_to_all_clusters_inPruned.csv"
            
        concepts_common_to_all_clusters_inPruned = record_common_concepts(all_pruned_concepts, task='across_clusters', fname=fname)
        print("concepts_common_to_all_clusters_inPruned: ", len(concepts_common_to_all_clusters_inPruned["Concepts common to all clusters"]))
        
        if i == 0:
            fname = "Analysis/IndivConceptAnalysis/concepts_common_to_all_clusters_inPrunedBeforeRT.csv"
        else:
            fname = "Analysis/CompositionalConceptAnalysis/concepts_common_to_all_clusters_inPrunedBeforeRT.csv"
          
        concepts_common_to_all_clusters_inPrunedBeforeRT = record_common_concepts(all_pwor_concepts, task='across_clusters', fname=fname)
        print("concepts_common_to_all_clusters_inPrunedBeforeRT: ", len(concepts_common_to_all_clusters_inPrunedBeforeRT["Concepts common to all clusters"]))
        



    
main()

# when u prune wo retran you retain some concepts and lost some. of the ones you retain, do the pruned expls have these? --> x
# with the new ones, hiw many are new --> x
# how many old the new onces propogate to the fully pruned  --> x

#are there groups of concepts where wo pruning you learn them at one cluster but then after pruning you learn them at a different cluster
    #look at cluster 1 concepts and see if they're not in cluster 1 pruned, are they in 2,3,or 4
    
# are there groups of concepts that are learned at multiple clusters --> x



    
    next steps: pune 1% and see results
        
