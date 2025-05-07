import sys
sys.path.append("Analysis/")
import concept_getters
import concept_analysis
import cleaning
import record_local
import record_global
import os, utils



def pipe_avg_ious(filenames :list, prune_iter :int):
    '''
        Calculates and prints average IOU for a pruned and unpruned set
    '''
    dfs_pruned, dfs_og=collect_dfs(folder_p,folder_np)

    i=0
    for np,p in zip(dfs_og, dfs_pruned):
        i+=1
        print(f"Avg IOU 0% Cluster {i}: {concept_getters.get_avg_iou(np.best_iou)}")
        print(f"Avg IOU {0.5*prune_iter}% Cluster {i}: {concept_getters.get_avg_iou(p.best_iou)}")

def pipe_explanation_similiarity(folder_p, folder_np, task='global', get_concepts_func = 'group', fname=None):
    '''
        Records common concepts globally or locally between 2 pruning iterations
        Global comparison looks at concepts across all clusters
        Local comparison compares concepts by cluster
        
        Returns: a list or dict of global or local common concepts (respectively)
    '''
    dfs_pruned, dfs_og= utils.collect_dfs(folder_p,folder_np)

    if task == 'global':
        if get_concepts_func == 'group':
            get_concepts_func = concept_getters.get_all_grouped_cps
        else:
            get_concepts_func = concept_getters.get_indiv_concepts
            
        all_pruned_concepts = get_concepts_func(dfs_pruned)
        all_nopruned_concepts = get_concepts_func(dfs_og)
        
        
        return record_global.record_common_concepts([all_pruned_concepts, all_nopruned_concepts], fname=fname)
    else:
        return record_local.record_common_concepts([dfs_pruned, dfs_og], fname=fname)
        
def pipe_percent_lost(folder_p,folder_np, task='global', get_concepts_func = 'group', fname=None, as_percent=False):
    '''
        Records the number of lost concepts across clusters (globally) and per cluster (locally)
        Feturns a list or dict of lost concepts
    '''
    dfs_pruned, dfs_og=collect_dfs(folder_p,folder_np)

    
    if get_concepts_func == 'group':
        get_concepts_func = concept_getters.get_grouped_concepts_per_cluster
    else:
        get_concepts_func = concept_getters.get_indiv_concepts_per_cluster
    all_pruned_concepts = get_concepts_func(dfs_pruned)
    all_nopruned_concepts = get_concepts_func(dfs_og)
    
    if task == 'global':
        all_nopruned_concepts_set=set()
        for lis in all_nopruned_concepts.values():
            for l in lis:
                all_nopruned_concepts_set.add(l)
        all_nopruned_concepts = all_nopruned_concepts_set

        all_pruned_concepts_set=set()
        for lis in all_pruned_concepts.values():
            for l in lis:
                all_pruned_concepts_set.add(l)
        all_pruned_concepts = all_pruned_concepts_set
       
    
        return record_global.record_lost_concepts(all_nopruned_concepts, all_pruned_concepts, fname, as_percent)
    else:
        return record_local.record_lost_concepts(all_nopruned_concepts, all_pruned_concepts, fname, as_percent)

