import os
import pandas as pd
import concept_getters,csv
import sys,utils
sys.path.append("/workspace/CCE_NLI/code/")
import settings
max_ = -1
main_dir = "exp/Run2"
thresh='Local_Threshold'
utils.save_concepts(f"{main_dir}/Expls0.0_Pruning_Iter/{thresh}/3Clusters", True)
utils.save_concepts(f"{main_dir}/Expls20.0_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls36.0_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls48.8_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls59.04_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls67.232_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls73.786_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls79.029_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls83.223_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls86.578_Pruning_Iter/{thresh}/3Clusters",True)
utils.save_concepts(f"{main_dir}/Expls89.263_Pruning_Iter/{thresh}/3Clusters",True)
#utils.save_concepts(f"{main_dir}/Expls91.41_Pruning_Iter/{thresh}/3Clusters",True)
#utils.save_concepts(f"{main_dir}/Expls93.128_Pruning_Iter/{thresh}/3Clusters",True)
#utils.save_concepts(f"{main_dir}/Expls94.502_Pruning_Iter/{thresh}/3Clusters",True)
#utils.save_concepts(f"{main_dir}/Expls95.602_Pruning_Iter/{thresh}/3Clusters",True)

'''
for pruning_folder in os.listdir(main_dir):
    if pruning_folder =='.ipynb_checkpoints' or '91' in pruning_folder:
        continue
    dfs=[]
    for clus in range(1,settings.NUM_CLUSTERS+1):
        filename = f"{main_dir}/{pruning_folder}/{thresh}/3Clusters/Cluster{clus}IOUS1024N.csv"
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
    pd.DataFrame(concepts_dict).to_csv(f"{main_dir}/{pruning_folder}/{thresh}/concepts_per_cluster.csv")
'''
                                  
print("Clustered Concepts saved")
        