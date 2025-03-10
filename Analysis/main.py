import utils
import record_global,cleaning
import pandas as pd
#get the 3 files

utils.save_concepts("Analysis/LHExpls/Run1/Expls0.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters", True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls20.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls95.602_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
files = [
    "Analysis/LHExpls/Run1/Expls0.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls20.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls95.602_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv"
]
a=pd.read_csv(files[0]).concepts
a=set(a)
b=pd.read_csv(files[1]).concepts
b=set(b)
c=pd.read_csv(files[2]).concepts
c=set(c)

zero_twenty=record_global.record_common_concepts([a,b], fname=None, percent=False)
twenty_95 = record_global.record_common_concepts([a,c], fname=None, percent=False)
print("Concepts common: ", zero_twenty)
print("Concepts common as a %: ", record_global.record_common_concepts([a,b], fname=None, percent=True))

print("Los concepts 0 to 20 ", record_global.record_lost_concepts(a,b))
print("Los concepts 0 to 95 ", record_global.record_lost_concepts(a,c))

print("common lost\n")
for concept in list(record_global.record_lost_concepts(a,c).values())[0]:
    if concept in list(record_global.record_lost_concepts(a,b).values())[0]:
        print(concept)
print("Los concepts 20 to 95 ", record_global.record_lost_concepts(b,c))

print("Preserved concepts") #in 0 and 20 so if in 20 and 95 as well its in all 3
for vals in list(zero_twenty.values())[0]:
    if vals in list(twenty_95.values())[0]:
        print(vals)


