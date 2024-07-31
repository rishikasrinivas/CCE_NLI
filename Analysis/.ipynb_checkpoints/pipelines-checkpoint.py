import sys
sys.path.append("Analysis/")
import concept_getters
import concept_analysis
import cleaning
import record_local
import record_global
import os
def pipe_avg_ious(filenames :list, prune_iter :int):
    dfs_pruned=[]
    dfs_og= []
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if 'pruned' in p_np:
                for f in file:
                    dfs_pruned.append(cleaning.prep(f))
            elif p_np == 'original':
                for f in file:
                    dfs_og.append(cleaning.prep(f))
            else:
                raise NameError("Invalid Key ", p_np)
    i=0
    for np,p in zip(dfs_og, dfs_pruned):
        i+=1
        print(f"Avg IOU 0% Cluster {i}: {concept_getters.get_avg_iou(np.best_iou)}")
        print(f"Avg IOU {0.5*prune_iter}% Cluster {i}: {concept_getters.get_avg_iou(p.best_iou)}")
def pipe_explanation_similiarity(filenames :list, task='global', get_concepts_func = 'group', fname=None):
    dfs_pruned=[]
    dfs_og= []
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if 'pruned' in p_np:
                for f in file:
                    dfs_pruned.append(cleaning.prep(f))
            elif p_np == 'original':
                for f in file:
                    dfs_og.append(cleaning.prep(f))
            else:
                raise NameError("Invalid Key ", p_np)
    
    if task == 'global':
        if get_concepts_func == 'group':
            get_concepts_func = concept_getters.get_grouped_concepts_per_cluster
        else:
            get_concepts_func = concept_getters.get_indiv_concepts_per_cluster
            
        all_pruned_concepts = get_concepts_func(dfs_pruned)
        all_nopruned_concepts = get_concepts_func(dfs_og)
        
        
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
        
        return record_global.record_common_concepts([all_pruned_concepts, all_nopruned_concepts], fname=fname)
    else:
        task = 'local'
        return record_local.record_common_concepts([dfs_pruned, dfs_og], fname=fname)
        
def pipe_percent_lost(filenames :list, task='global', get_concepts_func = 'group', fname=None, as_percent=False):
    dfs_pruned=[]
    dfs_og= []
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if 'pruned' in p_np:
                for filename in file:
                    dfs_pruned.append(cleaning.prep(filename))
            elif p_np == 'original':
                for filename in file:
                    dfs_og.append(cleaning.prep(filename))
            else:
                raise NameError("Invalid Key ", p_np)
    
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
        
def pipe_relearned_concepts(filenames :list, task='global', get_concepts_func = 'indiv', fname=None):
    dfs_pruned=[]
    dfs_og= []
    dfs_pruned_beforeRT=[]
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if p_np == 'prunedAfter':
                for f in file:
                    dfs_pruned.append(cleaning.prep(f))
            elif p_np == 'prunedBefore':
                for f in file:
                    dfs_pruned_beforeRT.append(cleaning.prep(f))
            elif p_np == 'original':
                for f in file:
                    dfs_og.append(cleaning.prep(f))
            else:
                raise NameError("Invalid Key ", p_np)
    
    if get_concepts_func == 'group':
        get_concepts_func = concept_getters.get_grouped_concepts_per_cluster
    elif 'indiv' in get_concepts_func:
        get_concepts_func = concept_getters.get_indiv_concepts_per_cluster
        
    all_pruned_concepts = get_concepts_func(dfs_pruned)
    all_nopruned_concepts = get_concepts_func(dfs_og)
    all_pruned_wo_retrain_concepts = get_concepts_func(dfs_pruned_beforeRT)
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
        
        all_pruned_wo_retrain_concepts_set=set()
        for lis in all_pruned_wo_retrain_concepts.values():
            for l in lis:
                all_pruned_wo_retrain_concepts_set.add(l)
        all_pruned_wo_retrain_concepts = all_pruned_wo_retrain_concepts
        
        return record_global.record_across_concepts(all_nopruned_concepts, all_pruned_concepts, all_pruned_wo_retrain_concepts, task='relearned', fname=fname)
    elif task == 'local':
        return record_local.record_across_concepts(all_nopruned_concepts, all_pruned_concepts, all_pruned_wo_retrain_concepts, task='relearned', fname=fname)