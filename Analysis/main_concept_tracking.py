import utils
import record_global,cleaning
import pandas as pd
#get the 3 files

utils.save_concepts("Analysis/LHExpls/Run1/Expls0.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters", True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls20.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls36.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls48.8_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls59.04_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls67.232_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls73.786_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
utils.save_concepts("Analysis/LHExpls/Run1/Expls95.602_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters",True)
files = [
    "Analysis/LHExpls/Run1/Expls0.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls20.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls36.0_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls48.8_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls59.04_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls67.232_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls73.786_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    "Analysis/LHExpls/Run1/Expls95.602_Pruning_Iter/Min_Acts_500_No_Filters/3Clusters_concepts.csv",
    
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

zero_twenty=record_global.record_common_concepts([a,b], fname=None, percent=False)
zero_36 = record_global.record_common_concepts([a,c], fname=None, percent=False)
zero_48 = record_global.record_common_concepts([a,d], fname=None, percent=False)
zero_59 = record_global.record_common_concepts([a,e], fname=None, percent=False)
zero_67 = record_global.record_common_concepts([a,f], fname=None, percent=False)
zero_73 = record_global.record_common_concepts([a,g], fname=None, percent=False)
zero_95 = record_global.record_common_concepts([a,h], fname=None, percent=False)

print("\nConcepts common 0-20: ", zero_twenty)
print("\nConcepts common 0-36: ", zero_36)
print("\nConcepts common 0-48: ", zero_48)
print("\nConcepts common 0-59: ", zero_59)
print("\nConcepts common 0-67: ", zero_67)
print("\nConcepts common 0-73: ", zero_73)
print("\nConcepts common 0-95: ", zero_95)
'''
print("Concepts common as a %: ", record_global.record_common_concepts([a,b], fname=None, percent=True))
'''
print("\nLos concepts 0 to 20 ", record_global.record_lost_concepts(a,b))
print("\nLos concepts 0 to 36 ", record_global.record_lost_concepts(a,c))
print("\nLos concepts 0 to 48 ", record_global.record_lost_concepts(a,d))
print("\nLos concepts 0 to 59 ", record_global.record_lost_concepts(a,e))
print("\nLos concepts 0 to 67 ", record_global.record_lost_concepts(a,f))
print("\nLos concepts 0 to 73 ", record_global.record_lost_concepts(a,g))
print("\nNew concepts 0-73 ", concept_getters,get_new_concepts(a,g))
print("\nLos concepts 0 to 95 ", record_global.record_lost_concepts(a,h))
'''
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
