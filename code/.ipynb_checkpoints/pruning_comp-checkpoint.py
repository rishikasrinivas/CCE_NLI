from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def neuron_count(df):
  return len(df['unit'].unique())

def active_neurons(df):
  units = []
  units.append(df['unit'].unique())
  return units
def calc_avg_iou(ious):
  ious=ious.tolist()
  return sum(ious)/len(ious)

def get_avg_ious(dfs,labels):
    avg_list=[]
    for df,label in zip(dfs,labels):
      avg_list.append(get_avg_iou(df['best_iou']))
    print(len(avg_list[4:]))
    data= {
        "Clusters":["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"],
        "Avgs_NotPruned":avg_list[:4],
        "Avgs_Pruned":avg_list[4:],
    }
    
    df  = pd.DataFrame(data)
    ax = df.plot(x="Clusters", y=["Avgs_NotPruned", "Avgs_Pruned"], kind="bar", rot=45)
    ax.set_title("5% Pruning: Avg IOU Across Clusters in Pruned vs Not pruned")
    
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
        pruned_exp=pruned_df[pruned_df['unit']==unit].best_name.iloc[0]
        not_pruned_exp=not_pruned_df[not_pruned_df['unit']==unit]['best_name'].iloc[0]
        sim_score = compute_similarity_between_exp(pruned_exp, not_pruned_exp).item()
        temp_df=pd.DataFrame({"Unit": [unit], "Similarity Score": [sim_score], "Pruned Explanation": [pruned_exp], "Not Pruned Explanation": [not_pruned_exp]})
        sim_df = pd.concat([sim_df, temp_df])

    return sim_df

def clean_file(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates().reset_index()
    df = df.drop(df[df.best_iou == 0.0].index)
    df=store_best_exp(df)
    return df

def store_best_exp(df):
    best_exp=pd.DataFrame({})
    for neuron in df['unit'].unique():
        index_of_best_exp = np.argmax(df[df['unit']==neuron]['best_iou'].tolist())
        formula = df[df['unit']==neuron].iloc[index_of_best_exp]['best_name']
        iou = df[df['unit']==neuron].iloc[index_of_best_exp]['best_iou']
        best_exp=pd.concat([best_exp, pd.DataFrame({'unit': [neuron], 'best_name':[formula], "best_iou":[iou]  })])
    return best_exp.reset_index().drop(columns=['index'])

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--cluster1Pruned',default ='../Cluster1IOUS1024N5%Pruned.csv')
    arg_parser.add_argument('--cluster2Pruned',default='../Cluster2IOUS1024N5%Pruned.csv')
    arg_parser.add_argument('--cluster3Pruned',default='../Cluster3IOUS1024N5%Pruned.csv')
    arg_parser.add_argument('--cluster4Pruned',default='../Cluster4IOUS1024N5%Pruned.csv')
    
    arg_parser.add_argument('--cluster1NotPruned',default='../Cluster1NotPrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster2NotPruned',default='../Cluster2NotPrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster3NotPruned',default='../Cluster3NotPrunedIOUS1024N.csv')
    arg_parser.add_argument('--cluster4NotPruned',default='../Cluster4NotPrunedIOUS1024N.csv')
    args = arg_parser.parse_args()
   
    pruned_files = [args.cluster1Pruned, args.cluster2Pruned, args.cluster3Pruned, args.cluster4Pruned]
    unpruned_files = [args.cluster1NotPruned, args.cluster2NotPruned, args.cluster3NotPruned, args.cluster4NotPruned]
    
    for p,np in zip(pruned_files,unpruned_files):
        if not os.path.isfile(p) or not os.path.isfile(np):
            return "Invalid csv provided"
        
    print(pruned_files)   
    for i in range(len(pruned_files)):
        pruned_df = clean_file(pruned_files[i])
        not_pruned_df = clean_file(unpruned_files[i])
        sim_df = get_similarity_scores(pruned_df, not_pruned_df)
        sim_df.to_csv(f"Cluster{i+1}PruningVsNoPruningSimilarity.csv")
        
        
    return 0
main()
    
        
    
