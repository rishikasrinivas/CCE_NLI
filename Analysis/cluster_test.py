import concept_getters
import pandas as pd

files=['Analysis/LHExpls/Expls91.41_Pruning_Iter/4Clusters/Cluster1IOUS1024N.csv',
      "Analysis/LHExpls/Expls91.41_Pruning_Iter/4Clusters/Cluster2IOUS1024N.csv",
      "Analysis/LHExpls/Expls91.41_Pruning_Iter/4Clusters/Cluster3IOUS1024N.csv",
      "Analysis/LHExpls/Expls91.41_Pruning_Iter/4Clusters/Cluster4IOUS1024N.csv",
      #"Analysis/LHExpls/Expls91.41_Pruning_Iter/5Clusters/Cluster5IOUS1024N.csv",
      #"Analysis/LHExpls/Expls0.0_Pruning_Iter/6Clusters/Cluster6IOUS1024N.csv"
      ]

def best_cluster():
    concepts=set()
    for file in files:
        concepts_len=len(concepts)
        indiv=concept_getters.get_indiv_concepts(pd.read_csv(file))
 
        concepts.update(indiv)
        concepts_len_updated=len(concepts)
        print(concepts_len_updated-concepts_len)

best_cluster()  