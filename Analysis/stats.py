

# find %of concepts lost from initislly learned when 1st pruned  = num concepts lost/union(pruned no rt, orig)
def percent_concepts_lost(lost, pruned_dict, orig_dict):
    i=0
    percent_lost_per_clus = {}
    for p,np in zip(pruned_dict.keys(), orig_dict.keys()):
        i += 1
        pruned_cps = pruned_dict[p]
        nopruned_cps = orig_dict[np] 
        
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
