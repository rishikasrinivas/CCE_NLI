

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
import networkx as nx
import csv

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import umap
from matplotlib.colors import hsv_to_rgb

import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

import os
from gensim.models import Word2Vec
import colorcet as cc

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

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
print(f"number of words in ds : {len(VECS_STOI)}")
import pickle

# Example dictionary

# Write dictionary to a pickle file
with open("VECS_STOI.pkl", "wb") as f:
    pickle.dump(VECS_STOI, f)
with open("code/tok_feats_vocab.pkl", "rb") as f:
    tok_feats_vocab= pickle.load(f)
print("Dictionary saved to VECS_STOI.pkl")



NEIGHBORS_CACHE = {}

neighborhood_size= 7 #2*settings.EMBEDDING_NEIGHBORHOOD_SIZE
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


def buildneighborgraph(concpetset):
    import networkx as nx
    G = nx.Graph()
    
    for word in concpetset:
        neighbors = get_neighors(word)
        for n in neighbors:
            G.add_edge(word,n)
    return list(nx.connected_components(G))


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

def get_all_concepts(folder = f'BERT/exp/CoFi/Run0.25_5/Expls/0.7789752992375562%Pruned'):
    all_concepts = set()
    for cluster in range(1,4):
        mapping, all_concepts_cluster = load_csv_data(os.path.join(folder, f'Cluster{cluster}IOUS1024N.csv'))
        all_concepts = all_concepts | all_concepts_cluster
    return all_concepts

def issubset(ns, group):
    for n in ns:
        if n not in group:
            return False
    return True

def collect_neighbors(concept_set):
    groups = []  # list of sets
    dictionary={}
    inv = 0
    concept_to_group = {}
    print(len(concept_set))
    for concept in concept_set:
        
        try:
            concept = concept.split(":")[-1]
        except:
            concept = concept
        ns = set(get_neighbors(concept))
        
        
        if not ns:
            print("c ", concept)
            inv += 1
            continue
        
        
        ns.update([concept])
        
        groups.append(ns)
        dictionary[concept] = ns


    groups = [g for g in groups if g]
    unq = set()
    for g in groups:
        unq.update(g)
    print("Number of concepts used: " ,len(dictionary), inv, len(groups))
    #assert len(dictionary) == len(unq), f'didnt catch all words, got {len(dictionary)}/{len(unq)}'
    return groups, dictionary

'''import pickle


# Write dictionary to a pickle file
with open("code/NEIGHBORSGLOBAL.pkl", "wb") as f:
    pickle.dump(collect_neighbors(tok_feats_vocab), f)

print("Dictionary saved to NEIGHBORSGLOBAL.pkl")'''

POS= ["''",
 '-lrb-',
 '-rrb-',
 '``',
 'cc',
 'cd',
 'dt',
 'fw',
 'jj',
 'jjr',
 'jjs',
 'md',
 'nn',
 'nnp',
 'nnps',
 'nns',
 'overlap25',
 'overlap50',
 'overlap75',
 'pdt',
 'pos',
 'prp',
 'prp$',
 'rb',
 'rbr',
 'rbs',
 'rp',
 'sym',
 'uh',
 'vb',
 'vbd',
 'vbg',
 'vbn',
 'vbp',
 'vbz',
 'wdt',
 'wp',
 'wp$',
 'wrb']
def build_similarity_matrix(concept_neighbors_dict):
    try:
        concepts = list(concept_neighbors_dict.keys())
        
    except:
        concepts= concept_neighbors_dict
    if len(set(concepts) & set(POS)) == 0:
        concepts.extend(POS)
    N = len(concepts)
    sim_matrix = pd.DataFrame(0.0, index=concepts, columns=concepts)

    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i >= j:
                continue
            
            
            if c1 in POS:
                sim_matrix.loc[c1, 'POS']=8
                sim_matrix.loc['POS', c1]=8
                continue
            elif c2 in POS:
                sim_matrix.loc[c2, 'POS']=8
                sim_matrix.loc['POS', c2]=8
                continue
    
        
            set1 = concept_neighbors_dict.get(c1, set())
            set2 = concept_neighbors_dict.get(c2, set())
            if not set1 and not set2:
                continue
            
            
            set1 = set(set1)
            set2 = set(set2)
            sim = len(set1 & set2)
            sim_matrix.loc[c1, c2] = sim
            sim_matrix.loc[c2, c1] = sim
            
    return sim_matrix
      
def build_graph(concepts, sim_matrix):
    if 'POS' not in concepts:
        concepts.append('POS')

    G = nx.Graph()
    threshold = 2 # e.g., only connect if overlap > 0.3

    for c1 in concepts:
        for c2 in concepts:
            if c1 == c2:
                continue
            if sim_matrix.loc[c1, c2] >= threshold:
                G.add_edge(c1, c2)

    abstractions = list(nx.connected_components(G))
    return abstractions


conc , concept_neighbors_dict = collect_neighbors(tok_feats_vocab)
#sim_matrix = build_similarity_matrix(concept_neighbors_dict)
#sim_matrix.to_csv("code/sim.csv")
sim_matrix = pd.read_csv("code/sim.csv")
sim_matrix = sim_matrix.set_index('Unnamed: 0')

abstractions = build_graph(concepts =  list(concept_neighbors_dict.keys()), sim_matrix = sim_matrix)
'''with open("code/concept_neighbors_dict.pkl", "wb") as f:
    pickle.dump(concept_neighbors_dict, f)
exit(1)'''

# UTILS
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import umap

from scipy.spatial.distance import cdist
import numpy as np

def alignment(coords, labels, groupings):
    coords = np.array(coords)
    labels = np.array(labels)

    dense_idx = np.where(labels == 0)[0]
    sparse_idx = np.where(labels == 1)[0]

    dense_coords = coords[dense_idx]
    sparse_coords = coords[sparse_idx]

    # distance matrix: sparse x dense
    D = cdist(sparse_coords, dense_coords)

    results = []

    for i, sparse_i in enumerate(sparse_idx):
        nearest_dense_local = np.argmin(D[i])
        nearest_dense_global = dense_idx[nearest_dense_local]

        sparse_topic = groupings[sparse_i]
        dense_topic = groupings[nearest_dense_global]
        distance = D[i, nearest_dense_local]

        results.append((sparse_topic, dense_topic, distance))

    return results

def sentence_embedding_W2V(words, w2v):
    vecs = []
    for w in words:
        if w in w2v.wv.key_to_index:
            v = w2v.wv[w]
            v = v / np.linalg.norm(v)
            vecs.append(v)

    if not vecs:
        return np.zeros(w2v.vector_size)

    v = np.mean(vecs, axis=0)
    
    return v / np.linalg.norm(v)

def sentence_embedding_ST(concepts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(concepts)
    return embeddings

model = 'BERT'
method = 'lottery_ticket'
fname = 'Run0.25_5'


print(f"LOADING VALIDATION PAIRS")
with open('code/data/analysis/snli_1.0_dev.tok', 'r') as f:
    samples = f.readlines()
print(f"PREPARING DATA FOR W2V")
sentences=[]
for sentence in samples:
    sentences.append(sentence.split())

print(f"TRAINING W2V")
w2v = Word2Vec(sentences, vector_size=128, window=5, epochs=10, workers=4, min_count=1)
print(f"FINDING ABSTRACTIONS FOR EACH MODEL")


save_path=f"{model.upper()}/exp/{method}/{fname}/abstractions.pkl"
if not os.path.exists(save_path):
    print(f"FINDING ABSTRACTIONS FOR EACH MODEL")


    root_dir = f'{model}/exp/{method}/{fname}/Expls/'
    abstractions_dict= {}
    for sparsity in os.listdir(root_dir):
        if '%Pruned' not in sparsity: continue
        model_dir = os.path.join(root_dir, sparsity)
        all_concepts = get_all_concepts(folder = model_dir)

        concept_neighbors_dict = collect_neighbors(all_concepts)
        sim_matrix = build_similarity_matrix(concept_neighbors_dict)
        abstractions = build_graph(concepts = list(concept_neighbors_dict.keys()), sim_matrix = sim_matrix)
        
        abstractions_dict[sparsity] = abstractions

    with open(save_path, "wb") as f:
        pickle.dump(abstractions_dict, f)
else:
    with open(save_path, "rb") as f:
        abstractions_dict = pickle.load(f)
    
for s, a in abstractions_dict.items():
    num = 0

    for g in a:
        for c in g:
        
            num += 1
    print(f"{model}, {method}, {fname}, Sparsity {s} has {num} concepts covered via abstractions")

dense_coverage = abstractions_dict['0.0%Pruned']

dense_rels = sorted(dense_coverage)
def map_via_concept_alignment(sparse_rels, dense_rels):
    """
    For each sparse abstraction (set of concepts),
    find the dense abstraction with highest Jaccard similarity.

    Returns:
        list of tuples:
        (sparse_index, dense_index, similarity_score)
    """

    results = []

    for i, sparse_set in enumerate(sparse_rels):

        best_score = -1
        best_j = None

        for j, dense_set in enumerate(dense_rels):

            intersection = len(sparse_set & dense_set)
            union = len(sparse_set | dense_set)

            score = intersection / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_j = j
                best_dense=dense_set
        if best_score <= 0:
            results.append((sparse_set, 'No Match', best_score))
        else:
            results.append((sparse_set, best_dense, best_score))

    return results

#COLLECT EMBEDDINGS AND ASSOCIATED MODEL
for sparse_coverage in abstractions_dict:
    print(f"EMBEDDING {sparse_coverage} ABSTRACTIONS")
    group = {}
    groupings=[]
    coords_list=[]
    embeddings = []
    if sparse_coverage == '0.0%Pruned': continue
        
    sparse_rels = sorted(abstractions_dict[sparse_coverage])
    with open(f"{model}_{method}_{sparse_coverage}_difEmbedding_alignment_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sparse_topic", "dense_topic", "score"])
        writer.writerows(map_via_concept_alignment(sparse_rels, dense_rels))
    continue
    # ---- collect embeddings ----
    
    print(sparse_rels, len(sparse_rels))
    for index,relationship in enumerate([dense_rels, sparse_rels]):
        #if i > subset: break
        offset = len(group)
        for i, topic in enumerate(relationship):
            embeddings.append(sentence_embedding(topic,  w2v))
            group[i+offset] = index
            groupings.append(topic)
    print("Numner of abstractions ", len(group))
     
    
    '''labels = [1 if i >= offset else 0 for i in group]
    print(f"Number of labels {labels}")
    from sklearn.decomposition import PCA
    print(f"DIMENSIONALITY REDUCTION FOR {sparse_coverage}")
    # ---- UMAP projection ----
    embeddings = np.array(embeddings)

    #coords = PCA(n_components=3).fit_transform(embeddings)
    reducer = umap.UMAP(n_neighbors=min(len(embeddings)-1,10), min_dist=0.1, n_components=3, random_state=42)
    coords = reducer.fit_transform(embeddings)  # supervised


    #CREATE VISUALIZATION
    print(f"CREATING VISUALIZATION FOR {sparse_coverage}")
    num_abstractions = len(embeddings) #len(rels[0]) + len(rels[1])
    colors_list = ['blue', 'red']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    point_labels = list(string.ascii_uppercase) + [f"{i}+" for i in list(string.ascii_uppercase)] + [f"{i}--" for i in list(string.ascii_uppercase)] +  [f"{i}*" for i in list(string.ascii_uppercase)] + [f"{i}&" for i in list(string.ascii_uppercase)] + [f"{i}^" for i in list(string.ascii_uppercase)]

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

    print(f"SAVING TO WordMapping0to{sparse_coverage}_{method}.png")
    fig.savefig(f"WordMapping0to{sparse_coverage}_{method}.png", dpi=300, bbox_inches="tight")'''
    
    matches = alignment(coords, labels,groupings)
    with open(f"{model}_{method}_{sparse_coverage}_alignment_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sparse_topic", "dense_topic", "distance"])
        writer.writerows(matches)
        
    
