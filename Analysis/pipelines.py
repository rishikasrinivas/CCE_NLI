import concept_getters
import concept_analysis
import cleaning
import record_local
import record_global
def pipe_explanation_similiarity(filenames :list, task='global', get_concepts_func = 'group'):
    dfs_pruned=[]
    dfs_og= []
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if p_np == 'pruned':
                for fname in file:
                    dfs_pruned.append(cleaning.prep(fname))
            elif p_np == 'original':
                for fname in file:
                    dfs_og.append(cleaning.prep(fname))
            else:
                raise NameError("Invalid Key ", p_np)
    
    record_common_concepts(concepts : list, task = 'global', fname=None)            
    if task == 'global':
        if get_concepts_func == 'group':
            get_concepts_func = concept_getters.get_grouped_concepts_per_cluster
        else:
            get_concepts_func = concept_getters.get_indiv_concepts_per_cluster
        all_pruned_concepts = get_concepts_func(dfs_pruned)
        all_nopruned_concepts = get_concepts_func(dfs_og)

        all_nopruned_concepts = set(list(all_nopruned_concepts.values())[0])
        all_pruned_concepts = set(list(all_pruned_concepts.values())[0])
        
        return record_local.record_common_concepts([all_pruned_concepts, all_nopruned_concepts], task, fname=None)
    else:
        task = 'local'
        return record_local.record_common_concepts([dfs_pruned, dfs_og], task = task, fname=None)
        
def pipe_percent_lost(filenames :list, task='global', get_concepts_func = 'group'):
    dfs_pruned=[]
    dfs_og= []
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if p_np == 'pruned':
                for fname in file:
                    dfs_pruned.append(cleaning.prep(fname))
            elif p_np == 'original':
                for fname in file:
                    dfs_og.append(cleaning.prep(fname))
            else:
                raise NameError("Invalid Key ", p_np)
    
    if get_concepts_func == 'group':
        get_concepts_func = concept_getters.get_grouped_concepts_per_cluster
    else:
        get_concepts_func = concept_getters.get_indiv_concepts_per_cluster
    all_pruned_concepts = get_concepts_func(dfs_pruned)
    all_nopruned_concepts = get_concepts_func(dfs_og)
    
    if task == 'global':
        all_nopruned_concepts = set(list(all_nopruned_concepts.values())[0])
        all_pruned_concepts = set(list(all_pruned_concepts.values())[0])
       
    
        return record_global.record_lost_concepts(all_nopruned_concepts, all_pruned_concepts, fname, as_percent=True)
    else:
        return record_local.record_lost_concepts(all_nopruned_concepts, all_pruned_concepts, fname, as_percent=True)
        
def pipe_relearned_concepts(filenames :list, task='global', get_concepts_func = 'group'):
    dfs_pruned=[]
    dfs_og= []
    for expl_dict in filenames:
        for p_np, file in expl_dict.items():
            if p_np == 'pruned':
                for fname in file:
                    dfs_pruned.append(cleaning.prep(fname))
            elif p_np == 'original':
                for fname in file:
                    dfs_og.append(cleaning.prep(fname))
            else:
                raise NameError("Invalid Key ", p_np)
    
    if get_concepts_func == 'group':
        get_concepts_func = concept_getters.get_grouped_concepts_per_cluster
    else:
        get_concepts_func = concept_getters.get_indiv_concepts_per_cluster
    all_pruned_concepts = get_concepts_func(dfs_pruned)
    all_nopruned_concepts = get_concepts_func(dfs_og)
    if task == 'global':
        record_global.record_across_concepts(orig_dict, retrained_pruned_dict, noretrain_prune_dict, task, fname=None)