import numpy as np
import pandas as pd
import re

def store_best_exp(df):
    '''
    Parse the df and store only the best iou's per neuron
    return: df
    
    '''
    best_exp=pd.DataFrame({})
    for neuron in df['unit'].unique():
        index_of_best_exp = np.argmax(df[df['unit']==neuron]['best_iou'].tolist())
        formula = df[df['unit']==neuron].iloc[index_of_best_exp]['best_name']
        iou = df[df['unit']==neuron].iloc[index_of_best_exp]['best_iou']
        best_exp=pd.concat([best_exp, pd.DataFrame({'unit': [neuron], 'best_name':[formula], "best_iou":[iou]  })])
    return best_exp.reset_index().drop(columns=['index'])

def prep(file, edit):
    df = pd.read_csv(file)
    if edit:
        df = store_best_exp(df)
        df=df.drop_duplicates()
        
    df=df.drop((df[df.best_iou==0]).index)
    df=df.drop_duplicates()
    return df

def neuron_count(df):
    return len(df['unit'].unique())

def get_indiv_concepts(df, with_neurons):
    if not with_neurons:
        concepts= set()
        expls = [form for form in df.best_name]
        expls = "".join(expls)
        concps = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', expls)
        for i,_ in enumerate(concps):
            while concps[i][-1] == ')':
                concps[i] = concps[i][:-1]
                concepts.add(concps[i])
    else:
        concepts = {}
        for unit,expl in zip(df.unit, df.best_name):
            concps = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', expl)
            for i in range(len(concps)):
                while concps[i][-1] == ')':
                    concps[i] = concps[i][:-1]
            concepts[unit] = concps
    return concepts

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

def get_grouped_concepts_per_unit(df) -> dict:
    res={}
    for unit,form in zip(df.unit , df.best_name):
        res[unit] = get_grouped_concepts(form)
    return res

def all_grouped_cps(grouped_cps_per_unit : dict) -> set:
    c=set()
    for k,v in grouped_cps_per_unit.items():
        for i in v:
            c.add(i)
    return c

# identify concepts that were in np but not in pruned anymore
def lost_concepts(non_pruned : set, pruned_not_retrained : set) -> set:
    concepts_innotP_butnotin_prunedNR = non_pruned.difference(pruned_not_retrained) 
    concepts_inprunedNR_butnotin_notP = pruned_not_retrained.difference(non_pruned) 
    
    return concepts_innotP_butnotin_prunedNR, concepts_inprunedNR_butnotin_notP

def preserved_concepts(non_pruned :set, pruned_not_retrained:set):
    return non_pruned.intersection(pruned_not_retrained) 

def find_neurons_explaining(concept_dict, cps) -> list:
    units=[]
    for unit, cncps in concept_dict.item():
        if cps in concept_dict[unit]:
            units.append(unit)
    return units
    
#returns a dict that says which neuros explain each concept in pruned and not pruned
def rearranged_concepts(not_pruned_concept_dict, pruned_concept_dict, common) -> dict:
    d={}
    for common_cps in common:
        np_units=find_neurons_explaining(not_pruned_concept_dict, common_cps)
        p_units=find_neurons_explaining_inpruned_nottrain(pruned_concept_dict, common_cps)
        
        d[common_cps] = {'not_pruned':np_units, 'pruned':p_units}
    return d
def main():
    clus_1_p = prep("Cluster1IOUS1024N.csv",True)
    '''clus_2_p = prep("/content/Cluster2IOUS1024N5%Pruned.csv",True)
    clus_3_p = prep("/content/Cluster3IOUS1024N5%Pruned.csv",True)
    clus_4_p = prep("/content/Cluster4IOUS1024N5%Pruned.csv",True)'''


    clus_1_np = prep("Cluster1IOUSNoPrune.csv", True)
    clus_2_np = prep("Cluster2IOUSNoPrune.csv", True)
    clus_3_np = prep("Cluster3IOUSNoPrune.csv", True)
    clus_4_np = prep("Cluster4IOUSNoPrune.csv", True)
    
    
    #WHAT COMPOSITIONS ARE PRESERVED/LOST AFTER PRUNING
        #concs_1p: concept groups in pruned
        #concs_1np: concept groups in unpruned
        #lc[0]: concept groups in pruned but not in not pruned
        #lc[1]: concept groups in not pruned but not in pruned

    per_unit_compositions = get_grouped_concepts_per_unit(clus_1_p)
    concs_1p = all_grouped_cps(per_unit_compositions)
    per_unit_compositions=get_grouped_concepts_per_unit(clus_1_np)
    concs_1np = all_grouped_cps(per_unit_compositions)


    lc=lost_concepts(concs_1p, concs_1np)

    print("Num concepts after pruning: ", len(concs_1p))
    print("Num concepts before pruning: ", len(concs_1np))
    print("Num concepts in pruned but not in not pruned: ", len(lc[0]))
    print("Num concepts in not pruned but not in pruned: ", len(lc[1]))
    print("Preserved concepts: ", preserved_concepts(concs_1p,concs_1np))
    
    
    #WHAT INDIVIDUAL CONCEPTS ARE LOST/PRESERVED AFTER PRUNING
    clus_1_pruned_concepts = get_indiv_concepts(clus_1_p,False)
    print("Num concepts in pruned: " , len(clus_1_pruned_concepts))
    clus_1_notpruned_concepts=get_indiv_concepts(clus_1_np,False)
    print("Num concepts in not pruned: ", len(clus_1_notpruned_concepts))


    lc=lost_concepts(clus_1_pruned_concepts, clus_1_notpruned_concepts)
    print("Preserved indiv. concepts: ", preserved_concepts(clus_1_pruned_concepts,clus_1_notpruned_concepts))
        #some concepts are preserved but their compositions change. so dog and cat in not pruned, dog and cat remain presentafter pruning, but maybe its dog
        #and not cay or dog or cat 
    concepts_lost_after_pruning=lc[1] 
    print("Concepts lost after pruning: ",concepts_lost_after_pruning)
main()
          
