import re
import pandas as pd
from concept_getters import get_common_neurons
def Union(lst1, lst2):
    final_list = sorted(lst1 + lst2)
    return final_list

def concept_similarity(pruned_exps, not_pruned_exps):
    pruned_concps = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', pruned_exps)
    not_pruned_concps = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', not_pruned_exps)
    for i,_ in enumerate(pruned_concps):
        while pruned_concps[i][-1] == ')':
            pruned_concps[i] = pruned_concps[i][:-1]
    for i,_ in enumerate(not_pruned_concps):
        while not_pruned_concps[i][-1] == ')':
            not_pruned_concps[i] = not_pruned_concps[i][:-1]

    intersection = 0
    for pruned_conc in pruned_concps:
        if pruned_conc in not_pruned_concps:
            intersection += 1
    union = len(set(Union(pruned_concps, not_pruned_concps)))
    return intersection/union

def calculate_similarity_across_explanations(pruned, not_pruned):
    common_neurons = get_common_neurons(pruned, not_pruned)
    sim_df=pd.DataFrame({})
    for unit in common_neurons:
        pruned_exp=pruned[pruned['unit']==unit].best_name.iloc[0]

        not_pruned_exp=not_pruned[not_pruned['unit']==unit]['best_name'].iloc[0]


        concept_similarity(pruned_exp, not_pruned_exp)
        sim_df=pd.concat([sim_df, pd.DataFrame({'unit': [unit], 'sim': [concept_similarity(pruned_exp, not_pruned_exp)]})])
    sim_df=sim_df.reset_index().drop(columns=['index'])
    return sim_df

def count_ANDOR(df):
    d={}
    for i, form in enumerate(df['best_name']):

        ands=form.count("AND")
        ors=form.count("OR")
        nots=form.count("NOT")

        pattern = re.findall(r'\b(?:pre:tok:|hyp:tok:|oth:)\S*', form)
        if ands == 0:
            and_ratio=0
        else:
            and_ratio= (ands)/(ands+ors)
        if ors == 0:
            or_ratio = 0
        else:
            or_ratio = (ors)/(ands+ors)
        d[df['unit'].iloc[i]]={"Ands ": and_ratio, "Ors": or_ratio}

    return d

def sum_andor(diction):
    andsum=0
    orsum=0
    for key,val in diction.items():
        andsum+=diction[key]['Ands ']
        orsum+=diction[key]['Ors']
    return [andsum/(andsum+orsum), orsum/(andsum+orsum)]