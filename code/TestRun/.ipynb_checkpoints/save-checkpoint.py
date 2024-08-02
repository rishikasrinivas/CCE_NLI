import os,csv
import torch
import sys
sys.path.append("code/")
import models,settings
import numpy as np
import sklearn.cluster as scikit_cluster
import pickle
import pandas as pd

def cluster_activ_ranges(num_clusters, neuron, final_actives):
    clustering=scikit_cluster.KMeans(n_clusters= num_clusters, random_state=1234).fit(final_actives[neuron].reshape(-1,1))
    activation_range= {}
    for index, label in enumerate(clustering.labels_):
        if label in activation_range.keys():
            activation_range[label].append(final_actives[neuron][index])
        else:
            activation_range[label] = [final_actives[neuron][index]]
        activation_range[label].sort()
    return sort_dict(activation_range)

def sort_dict(activation_range):
    ranges = []
    activation_range_ = activation_range.values()
    activation_range_= sorted(activation_range_, key=lambda item: item[0])
    for i in activation_range_:
        ranges.append((min(i), max(i)))
    return ranges

def find_explaining_neurons(cluster):
    df = pd.read_csv(f"code/TestRun/Cluster{cluster}IOUS1024N.csv")
    return df['unit'].tolist()

def get_samples_for_neuron(num_clusters, final_activations, neuron, ranges):
    pairs = [] #dict of sets
    for cluster in range(num_clusters):
        pairs_ = []
        for sample, activation in enumerate(final_activations[neuron]):
            if activation >= ranges[cluster][0] and activation <= ranges[cluster][1]:
                pairs_.append(sample)
        pairs.append(pairs_)
    return pairs

def get_formula_for_neuron(cluster, neuron):
    df = pd.read_csv(f"code/TestRun/Cluster{cluster}IOUS1024N.csv")
    return df[df['unit']==neuron].best_name.item()

def get_sentences(pairs_indices, sentences):
    for i, pairs in enumerate(pairs_indices):
        for pair in pairs[:2]:
            yield i+1, pair, sentences[2*pair], sentences[(2*pair) + 1]

def write_file(file,fields,data):
    if os.path.exists(file):
        with open(file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(data)
    else:
        with open(file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fields)
            writer.writerows(data)
import random        
def main():
    with open("code/TestRun/OriginalActivations.pkl", 'rb') as f:
        final_activs=pickle.load(f)
    with open("code/Sentences.pkl", 'rb') as f:
        sents=pickle.load(f)

    num_clusters=[3,4,5,10]
   
    neurons=find_explaining_neurons(cluster=1)
    neurons=random.sample(neurons, k=10)
    for num_cluster in num_clusters:
        print(f"Working on {num_cluster} clusters")
        for neuron in neurons:
            print("Neuron: ", neuron)
            ranges = cluster_activ_ranges(num_clusters=num_cluster, neuron=neuron, final_actives=final_activs)
            pairs_indices=get_samples_for_neuron(num_clusters=num_cluster, final_activations=final_activs, neuron=neuron, ranges=ranges)
            field=['neuron','cluster','formula', 'pair_num', 'pre', 'hyp']
            file=f'Results{num_cluster}Cluster.csv'
            for cluster, pair_num, p, h in get_sentences(pairs_indices, sents):
                formula = get_formula_for_neuron(cluster, neuron)
                write_file(file,field,[[neuron, cluster, formula, pair_num, p, h]])
main()