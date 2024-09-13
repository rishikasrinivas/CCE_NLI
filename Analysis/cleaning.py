import numpy as np
import pandas as pd
import os
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

