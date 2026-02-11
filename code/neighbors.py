
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

neighborhood_size= 10 #settings.EMBEDDING_NEIGHBORHOOD_SIZE
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
    nearest_i = np.argsort(dists)[1 : neighborhood_size + 1]
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
mapping, all_conceptsc3 = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.7789752992375562%Pruned/Cluster3IOUS1024N.csv')
mapping, all_conceptsc1 = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.7789752992375562%Pruned/Cluster1IOUS1024N.csv')
mapping, all_conceptsc2 = load_csv_data('BERT/exp/CoFi/Run0.25_5/Expls/0.7789752992375562%Pruned/Cluster2IOUS1024N.csv')

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
threshold =2 # e.g., only connect if overlap > 0.3
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



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import umap

# pick the first clustering



from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def alignment(coords, labels):
    red = coords[np.array(labels)==1]
    blue = coords[np.array(labels)==0]
    D = cdist(red, blue)
    row_ind, col_ind = linear_sum_assignment(D)

    matching_dist = D[row_ind, col_ind].mean()
    print("Mean optimal matching distance:", matching_dist)
    return matching_dist


with open('data/analysis/snli_1.0_dev.tok', 'r') as f:
    samples = f.readlines()

for sentence in samples:
    sentences.append(sentence.split())
w2v = Word2Vec(sentences, vector_size=128, window=5, epochs=10, workers=4, min_count=1)

def sentence_embedding(words, w2v):
    vecs = []
    for w in words:
        if w in w2v.wv:
            v = w2v.wv[w]
            v = v / np.linalg.norm(v)
            vecs.append(v)

    if not vecs:
        return np.zeros(w2v.vector_size)

    v = np.mean(vecs, axis=0)
    return v / np.linalg.norm(v)

group = {}
groupings=[]
coords_list=[]
embeddings = []

for index,rel in enumerate(rels):

    # ---- collect embeddings ----
    
    rel = sorted(rel)
    offset = len(group)
    print(offset)
   
    for i, cluster in enumerate(rel):
        #if i > subset: break
        
        embedding = torch.zeros((128,))
        embeddings.append(sentence_embedding(cluster, w2v))
        group[i+offset] = index
        groupings.append(cluster)
    

    # ---- word -> cluster mapping ----
    word_to_cluster = {}
    for cid, cluster in enumerate(rel):
        word_to_cluster[cid] = cid

print(group)
labels = [1 if i > offset else 0 for i in group]

# ---- UMAP projection ----
embeddings = np.array(embeddings)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=3, random_state=42)

coords = reducer.fit_transform(embeddings)  # supervised


    # ---- colors ----



num_abstractions = len(embeddings) #len(rels[0]) + len(rels[1])
colors_list = ['red', 'blue'] #cc.glasbey_hv[:num_abstractions]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
import string

point_labels = list(string.ascii_uppercase) + [f"{i}+" for i in list(string.ascii_uppercase)] + [f"{i}--" for i in list(string.ascii_uppercase)] +  [f"{i}*" for i in list(string.ascii_uppercase)]

for i, ((x, y,z), l) in enumerate(zip(coords, labels)):
    

    ax.scatter(x, y, z,alpha=0.7, color=colors_list[l])
    ax.text(x, y, z, point_labels[i], fontsize=10, weight='bold')


legend_elements = [
    Patch(
        facecolor=colors_list[labels[i]],
        label=f"{point_labels[i]}: {', '.join(groupings[i])}"
    )
    for i in range(min(len(coords), len(groupings)))
]

ax.legend(
    handles=legend_elements,
    title="Topics",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    ncol=2
)
fig.savefig("WordMapping0to68_Cofi.png", dpi=300, bbox_inches="tight")
alignment(coords, labels)