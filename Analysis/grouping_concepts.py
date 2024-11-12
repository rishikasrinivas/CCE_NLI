import os
import pandas as pd
functional_concepts= set(["at", "in", "is", "there", "on", "to", "are", "for", "has", "not", "their" ,"is", "while", "and", "by", "with", "from", "are", "one", "no", "another", "this", "these", "that", "those", "my", "our", "your", "his", "her", "its", "their", "some", "any", "every", "other", "many", "more", "most", "enough", "few", "less", "much", "either", "neither", "several", "all", "both", "each", "one", "two", "three", "four", "first", "second", "third" ,"last", "can", "have", "be", "do", "could", "has", "am", "does", "will", "had", "is", "did", "would", "having", "are", "done", "shall", "was", "doing", "should", "were", "may", "been", "might", "being", "must", "but", "or", "yet", "nor", "for", "so", "before", "once", "since", "until", "when", "while", "as", "like",  "although", "though", "whereas", "while", "except", "because", "since", "if", "where", "when", "why", "whom", "whose", "which", "what", "how"])

# ================ Global concepts =======================
main_dir = "exp/Run2"
thresh='Local_Threshold'
for pruning_folder in os.listdir(main_dir):
    if pruning_folder == ".ipynb_checkpoints" or '91' in pruning_folder:
        continue
    ct_structural_cps = 0
    filename = f"{main_dir}/{pruning_folder}/{thresh}/3Clusters_concepts.csv"
    global_concepts = set(pd.read_csv(filename)['concepts'])
    
    for gc in global_concepts:
        if ":tag:" in gc or "oth:overlap:" in gc:
            ct_structural_cps += 1
        for sc in functional_concepts:
            sc = f":tok:{sc}"
            hyp_sc = "hyp"+sc
            pre_sc = "pre"+sc
            if gc == hyp_sc or gc ==pre_sc:
                ct_structural_cps += 1
    length=len(global_concepts)
    #print(f"% of structural concepts at {pruning_folder[5:8]}: {100*ct_structural_cps/length}")
    
# ================= Local concepts ==========================
local_strucutred= {}
for pruning_folder in os.listdir(main_dir):
    if pruning_folder == ".ipynb_checkpoints" or '91' in pruning_folder:
        continue
    filename = f"{main_dir}/{pruning_folder}/{thresh}/concepts_per_cluster.csv"
    df = pd.read_csv(filename)
    c1,c2,c3 = df['Cluster1'], df['Cluster2'], df['Cluster3']
    concepts = [c1,c2,c3]
    c1len,c2len,c3len=len(c1),len(c2),len(c3)
    num_func=[0,0,0]
    percent_func=[0,0,0]
   
    lens=[c1len,c2len,c3len]
    for idx in range(3):
        concepts[idx] = concepts[idx].dropna()
        for concept in list(concepts[idx]):
            #check if convept is a tag or overlap
            if ":tag:" in concept or "oth:overlap:" in concept:
                num_func[idx] += 1
            #check if the concept is a functional concept
            for sc in functional_concepts:
                sc = f":tok:{sc}"
                hyp_sc = "hyp"+sc
                pre_sc = "pre"+sc
                if concept == hyp_sc or concept ==pre_sc:
                    num_func[idx] += 1
        percent_func[idx] = num_func[idx]/lens[idx]
    local_strucutred[pruning_folder[5:8]] = [percent_func,  num_func] 
sorted_dict = dict(sorted(local_strucutred.items(), key=lambda item: item[0][0:2]))
for k,v in sorted_dict.items():
    print("Percent functional concepts:", k, ": ", v)
   