import pandas as pd

def save_to_csv(dictionary, fname):
    ml=0
    for k,v in dictionary.items():
        if len(v)>ml:
            ml=len(v)
    for k,v in dictionary.items():
        dictionary[k]=list(dictionary[k])
        while len(dictionary[k])!=ml:
            dictionary[k].append('')
    pd.DataFrame(dictionary).to_csv(fname)
    
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

 