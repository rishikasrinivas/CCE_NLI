import os
import pandas as pd
import concept_getters,csv
import sys
sys.path.append("/workspace/CCE_NLI/code/")
import settings
max_ = -1
for pruning_folder in os.listdir("Analysis/LHExpls/Run1"):
    if pruning_folder =='.ipynb_checkpoints':
        continue
    dfs=[]
    for clus in range(1,settings.NUM_CLUSTERS+1):
        filename = f"Analysis/LHExpls/Run1/{pruning_folder}/Min_Acts_500_No_Filters/3Clusters/Cluster{clus}IOUS1024N.csv"
        dfs.append(pd.read_csv(filename))
    #print(concept_getters.get_indiv_concepts_per_cluster(dfs))
        concepts_dict=concept_getters.get_indiv_concepts_per_cluster(dfs)
        for k in concepts_dict.keys():
            concepts_dict[k] = list(concepts_dict[k])
            length = len(concepts_dict[k])
            
            if length > max_:
                max_=length
        for k,v in concepts_dict.items():
            if len(list(concepts_dict[k])) < max_:
                for i in range(max_-len(list(concepts_dict[k]))):
                    concepts_dict[k].append("")
    pd.DataFrame(concepts_dict).to_csv(f"Analysis/LHExpls/Run1/{pruning_folder}/Min_Acts_500_No_Filters/concepts_per_cluster.csv")
  
                                  
print("Clustered Concepts saved")
        