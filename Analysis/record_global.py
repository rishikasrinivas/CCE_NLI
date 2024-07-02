import concept_getters
import utils
def record_lost_concepts(nonpruned_cps: set, pruned_cps : set, fname=None, as_percent = False) -> dict:
    lost_cps= {}
    if not as_percent:
        lost_cps['lost_concepts'] = concept_getters.get_lost_concepts(nonpruned_cps, pruned_cps) 
    else:
        lost_cps[ 'percent_lost' ] = len(concept_getters.get_lost_concepts(nonpruned_cps, pruned_cps)) / len(nonpruned_cps )
        
    if fname != None:
        utils.save_to_csv(lost_cps, fname)
    return lost_cps
def record_common_concepts(concepts : list, fname=None):

    common_overall_bween_pnp = concept_getters.get_preserved_concepts(concepts[0], concepts[1])
    num_total_cps = len(concepts[0].union(concepts[1]))
    common = {'% similarity globally': common_overall_bween_pnp / num_total_cps}
    if fname != None:
        utils.save_to_csv(common,fname)
    return common

def record_across_concepts(orig_cps, retrained_pruned_cps, noretrain_prune_cps task, fname=None):
    concepts_lost_after_pruning_wo_rt = record_lost_concepts(orig_cps,noretrain_prune_cps)
    recordings = {}
    #find concepts lost after pruning in retrrained
    if task == 'relearned':
        task_concepts = retrained_pruned_cps.intersection(concepts_lost_after_pruning_wo_rt.values())
        recordings['relearned'] = task_concepts
    elif task == 'lost':
        task_concepts = concepts_lost_after_pruning_wo_rt.values().difference(retrained_pruned_cps)
        recordings['lost'] = task_concepts
    else:
        raise ValueError("Invalid task argument")
   
    if fname != None:
        utils.save_to_csv(recordings, fname)
    return recordings
        
