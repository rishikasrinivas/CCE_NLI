import utils,files, code

import record_global,cleaning,concept_getters
import pandas as pd
import sys
import os

# Add the path where wandb_utils is located
sys.path.append(os.path.abspath('code/lotteryTicket/'))

# Now you can import the module
import wandb_utils
#get the 3 files
run_folder = "Local_Threshold"  #'No_Min_Acts_No_Filters' #
root_dir="exp/Run1"
l_k={'b':20.0, 'c':36.0, 'd':48.8, 'e':59.04, 'f':67.232, 'g':73.786, 'h':79.028, 'i':83.223, 'j':86.578,  'k':89.263, 'l': 91.41, 'm':93.128, 'n':94.502, 'o':95.602}

def compute_iou(np,p):
    cps=set()
    overlap=0
    for i in np:
        cps.add(i)
    for i in p:
        cps.add(i)
        if i in np:
            overlap += 1
    return overlap/len(cps)
    
def compare_units(np, p,f):
    df=pd.DataFrame()
    np=pd.read_csv(np)
    p=pd.read_csv(p)
    k=[]
    ious=[]
    np=concept_getters.get_indiv_concepts_per_unit(np)
    p=concept_getters.get_indiv_concepts_per_unit(p)
    for keys,val in np.items():
        if keys in p.keys():
            k.append(keys)
            ious.append(compute_iou(val,p[keys]))
    df['unit']=k
    df['iou']=ious
    df.to_csv(f)
    
def main_unitwise_iou():
    for k,v in l_k.items():
        compare_units(files.get_fname(0.0, 1), files.get_fname(v, 1), f"{root_dir}/Expls{v}_Pruning_Iter/IOU0-{v}_C1.csv")
        compare_units(files.get_fname(0.0, 2), files.get_fname(v, 2), f"{root_dir}/Expls{v}_Pruning_Iter/IOU0-{v}_C2.csv")
        compare_units(files.get_fname(0.0, 3), files.get_fname(v, 3), f"{root_dir}/Expls{v}_Pruning_Iter/IOU0-{v}_C3.csv")
  
'''  
utils.save_concepts(f"{root_dir}/Expls0.0_Pruning_Iter/{run_folder}/3Clusters", True)
utils.save_concepts(f"{root_dir}/Expls20.0_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls36.0_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls48.8_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls59.04_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls67.232_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls73.786_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls79.028_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls83.223_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls86.578_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls89.263_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls91.41_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls93.128_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls94.502_Pruning_Iter/{run_folder}/3Clusters",True)
utils.save_concepts(f"{root_dir}/Expls95.602_Pruning_Iter/{run_folder}/3Clusters",True)
'''
files = [
    f"{root_dir}/Expls0.0_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls20.0_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls36.0_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls48.8_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls59.04_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls67.232_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls73.786_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls79.028_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls83.223_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls86.578_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls89.263_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls91.41_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls93.128_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls94.502_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    f"{root_dir}/Expls95.602_Pruning_Iter/{run_folder}/3Clusters_concepts.csv",
    
]

a=pd.read_csv(files[0]).concepts
a=set(a)
b=pd.read_csv(files[1]).concepts
b=set(b)
c=pd.read_csv(files[2]).concepts
c=set(c)
d=pd.read_csv(files[3]).concepts
d=set(d)
e=pd.read_csv(files[4]).concepts
e=set(e)
f=pd.read_csv(files[5]).concepts
f=set(f)
g=pd.read_csv(files[6]).concepts
g=set(g)
h=pd.read_csv(files[7]).concepts
h=set(h)
i=pd.read_csv(files[8]).concepts
i=set(i)
j=pd.read_csv(files[9]).concepts
j=set(j)
k=pd.read_csv(files[10]).concepts
k=set(k)
l=pd.read_csv(files[11]).concepts
l=set(l)
m=pd.read_csv(files[12]).concepts
m=set(m)
n=pd.read_csv(files[13]).concepts
n=set(n)
o=pd.read_csv(files[14]).concepts #some miscount with a-f
o=set(o)

def main_rec_common_concepts():
    zero_twenty=record_global.record_common_concepts([a,b], fname=None, percent=False)
    zero_36 = record_global.record_common_concepts([a,c], fname=None, percent=False)
    zero_48 = record_global.record_common_concepts([a,d], fname=None, percent=False)
    zero_59 = record_global.record_common_concepts([a,e], fname=None, percent=False)
    zero_67 = record_global.record_common_concepts([a,f], fname=None, percent=False)
    zero_73 = record_global.record_common_concepts([a,g], fname=None, percent=False)
    zero_95 = record_global.record_common_concepts([a,h], fname=None, percent=False)
    '''
    print("\nConcepts common 0-20: ", zero_twenty)
    print("\nConcepts common 0-36: ", zero_36)
    print("\nConcepts common 0-48: ", zero_48)
    print("\nConcepts common 0-59: ", zero_59)
    print("\nConcepts common 0-67: ", zero_67)
    print("\nConcepts common 0-73: ", zero_73)
    print("\nConcepts common 0-95: ", zero_95)
    '''
#default expls
cluster1_pd=pd.read_csv(f"{root_dir}/Expls0.0_Pruning_Iter/{run_folder}/3Clusters/Cluster1IOUS1024N.csv")
cluster2_pd=pd.read_csv(f"{root_dir}/Expls0.0_Pruning_Iter/{run_folder}/3Clusters/Cluster2IOUS1024N.csv")
cluster3_pd=pd.read_csv(f"{root_dir}/Expls0.0_Pruning_Iter/{run_folder}/3Clusters/Cluster3IOUS1024N.csv")
c1_np=list(concept_getters.get_indiv_concepts(cluster1_pd))
c2_np=list(concept_getters.get_indiv_concepts(cluster2_pd))
c3_np=list(concept_getters.get_indiv_concepts(cluster3_pd))

def track_concepts():
    wandb=wandb_utils.wandb_init("CCE_NLI", "Tracking Neurons")
    for s,l in zip([b,c,d,e,f,g,h,i,j,k,l,m,n, o],['b','c','d','e','f','g','h', 'i','j', 'k', 'l', 'm', 'n', 'o']):
        lost_cps_loc={'Cluster_1':[], 'Cluster_2': [], 'Cluster_3': []}
        new_cps_loc={'Cluster_1':[], 'Cluster_2': [], 'Cluster_3': []}
        cluster1_pd_pruned=pd.read_csv(f"{root_dir}/Expls{l_k[l]}_Pruning_Iter/{run_folder}/3Clusters/Cluster1IOUS1024N.csv")
        cluster2_pd_pruned=pd.read_csv(f"{root_dir}/Expls{l_k[l]}_Pruning_Iter/{run_folder}/3Clusters/Cluster2IOUS1024N.csv")
        cluster3_pd_pruned=pd.read_csv(f"{root_dir}/Expls{l_k[l]}_Pruning_Iter/{run_folder}/3Clusters/Cluster3IOUS1024N.csv")

        c1_p=list(concept_getters.get_indiv_concepts(cluster1_pd_pruned))
        c2_p=list(concept_getters.get_indiv_concepts(cluster2_pd_pruned))
        c3_p=list(concept_getters.get_indiv_concepts(cluster3_pd_pruned))
        for val in list(record_global.record_lost_concepts(a,s).values())[0]:
            if val in c1_np :
                lost_cps_loc['Cluster_1'].append(val)
            if val in c2_np:
                lost_cps_loc['Cluster_2'].append(val)
            if val in c3_np:
                lost_cps_loc['Cluster_3'].append(val)
        for val in concept_getters.get_new_concepts(a, s):
            if val in c1_p:
                new_cps_loc['Cluster_1'].append(val)
            if val in c2_p:
                new_cps_loc['Cluster_2'].append(val)
            if val in c3_p:
                new_cps_loc['Cluster_3'].append(val) 
        total_new,total_lost=0,0
        for count in new_cps_loc.values():
            total_new += len(count)
        for count in lost_cps_loc.values():
            total_lost += len(count)
        c1=[len(new_cps_loc['Cluster_1'])/ len(c1_p), len(lost_cps_loc['Cluster_1'])/len(c1_np)]   
        c2=[len(new_cps_loc['Cluster_2'])/ len(c1_p), len(lost_cps_loc['Cluster_2'])/len(c1_np)]
        c3=[len(new_cps_loc['Cluster_3'])/ len(c1_p), len(lost_cps_loc['Cluster_3'])/len(c1_np)]

        print(f"\n\nNew 0 and {l_k[l]} ", new_cps_loc)
        print("\nPercent new in Cluster 1: ", len(new_cps_loc['Cluster_1'])/ len(c1_p))
        print("Percent new in Cluster 2: ", len(new_cps_loc['Cluster_2'])/len(c2_p))
        print("Percent new in Cluster 3: ", len(new_cps_loc['Cluster_3'])/len(c3_p))
        print("\n\nPercent lost in Cluster 1: ",len( lost_cps_loc['Cluster_1'])/len(c1_np))
        print("Percent lost in Cluster 2: ", len(lost_cps_loc['Cluster_2'])/len(c2_np))
        print("Percent lost in Cluster 3: ", len(lost_cps_loc['Cluster_3'])/len(c3_np))
        print(f"\nLost 0 and {l_k[l]}: {lost_cps_loc}")



        wandb.log({'C1 New':c1[0],'c1 Lost': c1[1]}) #plot new % and lost%
        wandb.log({'c2 New':c2[0],'c2 Lost': c2[1]}) #plot new % and lost%
        wandb.log({'c3 New':c3[0],'c3 Lost': c3[1]}) #plot new % and lost%
        
def recovered_concept_track():
    
    percents=list(l_k.values())
    lostAt20 = list(record_global.record_lost_concepts(a,b).values())[0]
    for percent in percents[1:]:
        recov=0
        concepts=pd.read_csv(f"{root_dir}/Expls{percent}_Pruning_Iter/{run_folder}/3Clusters_concepts.csv")['concepts'].tolist()
       
        for i in concepts:
            if i in lostAt20:
                recov += 1
        print(f"{percent}, {recov/len(concepts)}")
    
     
recovered_concept_track()
'''
print("\nLos concepts 0 to 36 ", record_global.record_lost_concepts(a,c))
print("\nLos concepts 0 to 48 ", record_global.record_lost_concepts(a,d))
print("\nLos concepts 0 to 59 ", record_global.record_lost_concepts(a,e))
print("\nLos concepts 0 to 67 ", record_global.record_lost_concepts(a,f))
print("\nLos concepts 0 to 73 ", record_global.record_lost_concepts(a,g))
print("\nNew concepts 0-73 ", concept_getters,get_new_concepts(a,g))
print("\nLos concepts 0 to 95 ", record_global.record_lost_concepts(a,h))

print("common lost\n")
for concept in list(record_global.record_lost_concepts(a,c).values())[0]:
    if concept in list(record_global.record_lost_concepts(a,b).values())[0]:
        print(concept)
print("Los concepts 20 to 95 ", record_global.record_lost_concepts(b,c))

print("Preserved concepts") #in 0 and 20 so if in 20 and 95 as well its in all 3
for vals in list(zero_twenty.values())[0]:
    if vals in list(zero_36.values())[0] and vals in list(zero_48.values())[0] and vals in list(zero_59.values())[0] and vals in list(zero_67.values())[0] and vals in list(zero_73.values())[0] and vals in list(zero_95.values())[0]:
        print(vals)



'''