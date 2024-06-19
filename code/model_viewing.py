import sys

import data
import data.snli
import settings
import os
import pandas as pd
import pickle 
def get_masked_connections(new_weights):
    masked_connections={}
    for out_n, weights in enumerate(new_weights):
        for in_n, val in enumerate(weights):
            if val == 0:
                if out_n in masked_connections.keys():
                    masked_connections[out_n].append(in_n)
                else:
                    masked_connections[out_n] = [in_n]
   

    with open('code/5%masked_connecs.pkl', 'wb') as f:
        pickle.dump(masked_connections, f)
    return masked_connections
def get_min_max(masked_connections):
    max_l=0
    min_l=60000
    for k,v in masked_connections.items():
        if len(v) > max_l:
            max_l=len(v)
        elif len(v) < min_l:
            min_l = len(v)
    print("Max L: ", max_l, "\nMin L: ", min_l) #every neuron has 102 connections removed
    
def main():
    model, dataset = data.snli.load_for_analysis(
        settings.MODEL,
        settings.DATA,
        model_type=settings.MODEL_TYPE,
        cuda=settings.CUDA,
    )
    for n, p in model.named_parameters():
        if n == 'mlp.0.weight_orig':
            orig=p
            break
    for n,p in model.named_buffers():
        if n == 'mlp.0.weight_mask':
            mask=p
            break

    new_weights=orig*mask

    print(new_weights.shape) #1024x2048
    print(os.listdir("code/"))
    if "5%masked_connecs.pkl" in os.listdir("code/"):
        with open('code/5%masked_connecs.pkl', 'rb') as f:
            masked_connections= pickle.load(f)
    else:
        masked_connections = get_masked_connections(new_weights)
    print(masked_connections)
    for k,v in masked_connections.items():
        print(f"{k}: {masked_connections[k][:9]}")
main()
        

                
