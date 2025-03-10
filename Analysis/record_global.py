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
def record_common_concepts(concepts : list, fname=None,percent=False):

    common_overall_bween_pnp = concept_getters.get_preserved_concepts(concepts[0], concepts[1])
    num_total_cps = len(concepts[0].union(concepts[1]))
    if percent:
        common = {'% similarity globally': len(common_overall_bween_pnp) / num_total_cps}
    else:
        common = {'% similarity globally': common_overall_bween_pnp}
    if fname != None:
        utils.save_to_csv(common,fname)
    return common

