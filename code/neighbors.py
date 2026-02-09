
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#todo: make clusters
from __future__ import unicode_literals
#comment
import multiprocessing as mp
import os
import re
from collections import Counter, defaultdict
import metrics
import numpy as np
#import onmt.opts as opts
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import settings

def load_vecs(path):
    vecs = []
    vecs_stoi = {}
    vecs_itos = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tok, *nums = line.split(" ")
            nums = np.array(list(map(float, nums)))

            assert tok not in vecs_stoi
            new_n = len(vecs_stoi)
            vecs_stoi[tok] = new_n
            vecs_itos[new_n] = tok
            vecs.append(nums)
    vecs = np.array(vecs)
    return vecs, vecs_stoi, vecs_itos


# Load vectors
VECS, VECS_STOI, VECS_ITOS = load_vecs(settings.VECPATH)


NEIGHBORS_CACHE = {}


def get_neighbors(lemma):
    """
    Get neighbors of lemma given glove vectors.
    """
    if lemma not in VECS_STOI:
        # No neighbors
        return set()
    if lemma in NEIGHBORS_CACHE:
        return NEIGHBORS_CACHE[lemma]
    lemma_i = VECS_STOI[lemma]
    lvec = VECS[lemma_i][np.newaxis]
    dists = cdist(lvec, VECS, metric="cosine")[0]
    # first dist will always be the vector itself
    nearest_i = np.argsort(dists)[1 : settings.EMBEDDING_NEIGHBORHOOD_SIZE + 1]
    nearest = [VECS_ITOS[i] for i in nearest_i]
    NEIGHBORS_CACHE[lemma] = nearest
 
    return set(nearest)
def get_indiv_concepts(formula) -> list:
    concepts = []
    concps = re.findall(r'(?<!\bNOT\s)(?:\b(?:hyp|pre|oth):[^\s)]+)', formula)
    for c in concps:
        try:
            end_idx = c.index(')')
        except:
            end_idx = len(c)
        concepts.append(c[:end_idx])
    return concepts


def load_csv_data(filepath):
    """Load CSV and extract unit-concept mappings."""
    df = pd.read_csv(filepath)
    unit_concepts = defaultdict(set)
    raw_concepts= []
    for _, row in df.iterrows():
        unit = row['unit']
        formula = row['best_name']
        concepts = get_indiv_concepts(formula)
        
        unit_concepts[unit].update(concepts)
        raw_concepts.extend(concepts)
    modified_raw_concepts = [word.split(":")[-1] for word in raw_concepts]
        
    return unit_concepts, set(modified_raw_concepts)
#how concepts removal same across algortihms 
mapping, all_conceptsc3 = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.6841850626856109%Pruned/Cluster3IOUS1024N.csv')
mapping, all_conceptsc1 = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.6841850626856109%Pruned/Cluster1IOUS1024N.csv')
mapping, all_conceptsc2 = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.6841850626856109%Pruned/Cluster2IOUS1024N.csv')

all_concepts = all_conceptsc3 | all_conceptsc2 | all_conceptsc1
#for each concept get neighbors (dict {concept: neighbors})
concept_neighbors_dict = {}
for concept in set(all_concepts):
    concept_neighbors_dict[concept] = get_neighbors(concept)
    
import pandas as pd

concepts = list(concept_neighbors_dict.keys())
N = len(concepts)
sim_matrix = pd.DataFrame(0.0, index=concepts, columns=concepts)

for i, c1 in enumerate(concepts):
    for j, c2 in enumerate(concepts):
        if i >= j:
            continue
        set1 = concept_neighbors_dict[c1]
        set2 = concept_neighbors_dict[c2]
        if c1.startswith("nn") or c1.startswith("vb") or c1.startswith("jj"):
            sim_matrix.loc[c1, 'POS']=8
            sim_matrix.loc['POS', c1]=8
        elif c2.startswith("nn") or c2.startswith("vb") or c2.startswith("jj"):
            sim_matrix.loc[c2, 'POS']=8
            sim_matrix.loc['POS', c2]=8
            
            
        if not set1 and not set2:
            continue

        sim = len(set1 & set2)
        sim_matrix.loc[c1, c2] = sim
        sim_matrix.loc[c2, c1] = sim
        
import networkx as nx

G = nx.Graph()
threshold = 1 # e.g., only connect if overlap > 0.3
concepts.append("POS")
for c1 in concepts:
    for c2 in concepts:
        if c1 == c2:
            continue
        if sim_matrix.loc[c1, c2] > threshold:
            G.add_edge(c1, c2)

clusters = list(nx.connected_components(G))
print(clusters)
#{concept1: [concept5, concept2,....]}
#group concepts into ones that have similar neighbors  {[concept1, concept2, concept3...], [concept4, concept5,....]}
#can embed these using w2v so plot it
#then repeat for other sparsities
