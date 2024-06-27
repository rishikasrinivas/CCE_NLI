import pandas as pd
import re

#gets individual concepts like hyp:tok:dog, pre:tok:cat
def get_indiv_concepts(df) -> set:
    concepts= set()
    expls = [form for form in df.best_name]
    expls = " ".join(expls)
    concps = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', expls)
    for i,_ in enumerate(concps):
        while concps[i][-1] == ')':
            concps[i] = concps[i][:-1]
        concepts.add(concps[i])
    return concepts
                
def get_indiv_concepts_per_unit(df) -> dict:
    concepts = {}
    for unit,expl in zip(df.unit, df.best_name):
        concps = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', expl)
        for i in range(len(concps)):
            while concps[i][-1] == ')':
                concps[i] = concps[i][:-1]
        concepts[unit] = concps
    return concepts

#gets all the compositions: ((dog and cat ) or tree) and fish --> (dog and cat), ((dog and cat ) or tree), fish, ((dog and cat ) or tree) and fish 
def get_grouped_concepts(formula) -> list:
    stack = []
    results = []
    for i, char in enumerate(formula):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            results.append(formula[start:i+1])
    return results

#get the compositions for each unit
def get_grouped_concepts_per_unit(df) -> dict:
    res={}
    for unit,form in zip(df.unit , df.best_name):
        res[unit] = get_grouped_concepts(form)
    return res

#gets all the compositions across all the units
def get_all_grouped_cps(df) -> set:
    c=set()
    grouped_cps_per_unit = get_grouped_concepts_per_unit(df)
    for k,v in grouped_cps_per_unit.items():
        for i in v:
            c.add(i)
    return c

# returns all the concepts that are in the non-pruned, but not in the pruned and vise versa
def get_lost_concepts(non_pruned : set, pruned : set) -> set:
    concepts_innotP_butnotin_prunedNR = non_pruned.difference(pruned) 
    
    return concepts_innotP_butnotin_prunedNR 

def get_new_concepts(non_pruned : set, pruned: set):
    
    return pruned.difference(non_pruned) 

# returns the concepts common to the non-pruned and pruned explanations
def get_preserved_concepts(non_pruned :set, pruned_not_retrained:set):
    return non_pruned.intersection(pruned_not_retrained) 

# returns the neurons explaining a specific concept 
def find_neurons_explaining(concept_dict, cps) -> list:
    units=[]
    for unit, _ in concept_dict.items():
        if cps in concept_dict[unit]:
            units.append(unit)
    return units
    
#returns a dict that says which neuros explain each concept in pruned and not pruned
def get_common_concepts_explained_neurons(not_pruned_concept_dict, pruned_concept_dict, common) -> dict:
    d={}
    for common_cps in common:
        np_units=find_neurons_explaining(not_pruned_concept_dict, common_cps)
        p_units=find_neurons_explaining(pruned_concept_dict, common_cps)
        
        d[common_cps] = {'not_pruned':np_units, 'pruned':p_units}
    return d

def get_avg_iou(ious):
    ious=ious.tolist()
    return sum(ious)/len(ious)

def get_common_neurons(pruned, not_pruned):
    return set(pruned['unit'].unique()).intersection(set(not_pruned['unit'].unique()))

def get_common_concepts(dfs1, dfs2, dfs3):
    common = {}
    i=0
    for df1, df2, df3 in zip(dfs1, dfs2, dfs3):
        i+=1
        df1 = concept_ret(df1)
        df2 = concept_ret(df2)
        df3 = concept_ret(df3)
        
        common[f"Cluster{i}"] = (df1.intersection(df2)).intersection(df3)
    return common
        
        