import cleaning,os,collections

files=['Expls0.0_Pruning_Iter', 'Expls20.0_Pruning_Iter','Expls36.0_Pruning_Iter','Expls48.8_Pruning_Iter','Expls59.04_Pruning_Iter',
      'Expls67.232_Pruning_Iter','Expls73.786_Pruning_Iter']
main_dir="exp/Run2/"
thresh = "Local_Threshold"


def comp(vals, base):
    new=[]
    common=[]
    for v in vals:
        if v not in base:
            new.append(v)
        else:
            common.append(v)
    return new,common
d=collections.defaultdict(list)
for file in files:
    for clus in [1,2,3]:
        cluster=cleaning.prep(main_dir+file+ "/"+thresh+f"/3Clusters/Cluster{clus}IOUS1024N.csv")
        neurons=cluster.unit
        if file[5:8] == '0.0':
            base_neurons=list(neurons) 
        d[file[5:9]].extend(list(neurons))

for vals in list(d.values())[1:]:
    new,common=comp(vals, base_neurons)
    print("New: ", len(new)/len(vals), "\nCommon:",len(common)/len(vals))
    
    