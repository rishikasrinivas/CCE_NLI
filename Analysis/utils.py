import pandas as pd
import concept_getters
import csv, os,cleaning
def save_to_csv(dictionary, fname):
    ml=0
    for k,v in dictionary.items():
        if type(v)!= float:
            if len(v)>ml:
                ml=len(v)
        else:
            dictionary[k] = [v]
    if type(v) != float:
        for k,v in dictionary.items():

            dictionary[k]=list(dictionary[k])
            while len(dictionary[k])!=ml:
                dictionary[k].append('')

    pd.DataFrame(dictionary).to_csv(fname)
    
def collect_dfs(folder):
    dfs=[]
    for expl in os.listdir(folder):
        
        if expl[-3:] == "csv" and expl!='result.csv':
            dfs.append(cleaning.prep(folder+"/"+expl))
    return dfs

def save_concepts(folder,glob):
    dfs=collect_dfs(folder)
    if glob:
        concepts=[]
        for df in dfs:
            concepts.extend(concept_getters.get_indiv_concepts(df))
        with open(f'{folder}_concepts.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(['concepts'])
            write.writerows([[c] for c in concepts])
    else:
        cps=concept_getters.get_indiv_concepts_per_cluster(dfs)
        save_to_csv(cps, fname)

def intersection(lst1, lst2):
    intersect=0
    for i,j in zip(lst1,lst2):
        if i==j:
            intersect += 1
    return intersect

def union(lst1,lst2):
    union=0
    for i,j in zip(lst1,lst2):
        if i==1 or j==1:
            union += 1
    return union

# have concepts at each cluster and overall 
   