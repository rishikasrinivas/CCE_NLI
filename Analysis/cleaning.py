import numpy as np
import pandas as pd

def store_best_exp(df):
    best_exp=pd.DataFrame({})
    for neuron in df['unit'].unique():
        index_of_best_exp = np.argmax(df[df['unit']==neuron]['best_iou'].tolist())
        formula = df[df['unit']==neuron].iloc[index_of_best_exp]['best_name']
        iou = df[df['unit']==neuron].iloc[index_of_best_exp]['best_iou']
        best_exp=pd.concat([best_exp, pd.DataFrame({'unit': [neuron], 'best_name':[formula], "best_iou":[iou]  })])
    return best_exp.reset_index().drop(columns=['index'])

def prep(file):
    df = pd.read_csv(file)
    df = store_best_exp(df)
    df=df.drop((df[df.best_iou==0]).index)
    df=df.drop_duplicates()
    return df

def get_percent_dif(run1, run2, run3):
    pairs = [(run1, run2), (run1, run3), (run2, run3)]
    index2key={0:"Dif Between Run 0 and 1", 1:"Dif Between Run 1 and 2", 2:"Dif Between Run 2 and 3" }
    dif_percent = {"Dif Between Run 0 and 1":0 , "Dif Between Run 1 and 2":0, "Dif Between Run 2 and 3":0}
    for i,pair in enumerate(pairs):
        int_runArunB_df = pd.merge(pair[0], pair[1])
        union_runArunB_df = pd.concat([pair[0], pair[1]])
        int_runArunB_df=int_runArunB_df.drop_duplicates()
        union_runArunB_df=union_runArunB_df.drop_duplicates()
        sim=len(int_runArunB_df)/len(union_runArunB_df)
        dif_percent[index2key[i]] = (1-sim)
    return dif_percent