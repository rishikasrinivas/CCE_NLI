from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from argparse import ArgumentParser

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_similarity_between_exp(pruned_exp, not_pruned_exp):
    sentences = [pruned_exp, not_pruned_exp]
    #Compute embedding for both lists
    embedding_1= model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    return util.pytorch_cos_sim(embedding_1, embedding_2)
    ## tensor([[0.6003]])
    
def get_similarity_scores(pruned_df, not_pruned_df):
    common_neurons = set(pruned_df['unit'].unique()).intersection(set(not_pruned_df['unit'].unique()))
    sim_df=pd.DataFrame({})
    for unit in common_neurons:
        pruned_exp=pruned_df[pruned_df['unit']==unit]['best_name']
        not_pruned_exp=not_pruned_df[not_pruned_df['unit']==unit]['best_name']
        sim_score = compute_similarity_between_exp(pruned_exp, not_pruned_exp).item()
        temp_df=pd.DataFrame({"Unit": unit, "Similarity Score": sim_score, "Pruned Explanation": pruned_exp, "Not Pruned Explanation": not_pruned_exp})
        sim_df = pd.concat([sim_df, temp_df])
    return sim_df

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--cluster1Pruned',help='Cluster1PrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster2Pruned',help='Cluster2PrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster3Pruned',help='Cluster3PrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster4Pruned',help='Cluster4PrunedIOUS1024N.csv')
    
    arg_parser.add_argument('--cluster1NotPruned',help='Cluster1NotPrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster2NotPruned',help='Cluster2NotPrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster3NotPruned',help='Cluster3NotPrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster4NotPruned',help='Cluster4NotPrunedIOUS1024N.csv')
   

    
    pruned_files = [args.cluster1Pruned, args.cluster2Pruned, args.cluster3Pruned, args.cluster4Pruned]
    unpruned_files = [args.cluster1NotPruned, args.cluster2NotPruned, args.cluster3NotPruned, args.cluster4NotPruned]
    
    for p,np in zip(pruned_files,unpruned_files):
        if not os.path.isfile(p) or not os.path.isfile(np):
            return "Invalid csv provided"
        
    for i in range(len(pruned_files)):
        pruned_df = pd.read_csv(pruned_files[i])
        not_pruned_df = pd.read_csv(unpruned_files[i])
        sim_df = get_similarity_scores(pruned_df, not_pruned_df)
        sim_df.to_csv(f"Cluster{i}PruningVsNoPruningSimilarity.csv")
        
        
    return 0
    
        
    
