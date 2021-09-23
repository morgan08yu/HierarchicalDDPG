#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:00:40 2020

@author: Morgans
"""

from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (15,6)
plt.style.use('ggplot')

# numeric
import numpy as np
from numpy import random
import pandas as pd

import glob

from tqdm import tqdm_notebook as tqdm
import os, json

# '/Users/Morgans/Desktop/trading_system/HFT_data/additional dataset/*.csv'
dfs=[]
for infile in glob.glob('/Users/Morgans/Desktop/trading_system/HFT_data/ETF/*.csv'):
    df = pd.read_csv(infile)
    
    # date
    df.index=pd.to_datetime(df.Date)
    del df['Date']
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']] #'Volume'
    #df=df.resample('1T').first(
    # name cols
    name = os.path.splitext(os.path.basename(infile))[0]
#     df.columns = ['%s|%s'%(name,col) for col in df.columns]
    df.name = name
    
    dfs.append(df)

dfs.sort(key=lambda x:len(x), reverse=True)
[(df.name,df.index[0]) for df in dfs]


"check NA"
   # df =  pd.read_csv('/Users/Morgans/Desktop/trading_system /HFT_data/GSPC2018jan.csv')
   # df.index= pd.to_datetime(df.Date)
   # dftest=dfs   
"select data "
dfs1= [df for df in dfs if df.index.min() < pd.Timestamp('2020-07-30')]
#dfs1= [df for df in dfs1 if df.name.endswith('BTC')]
print([str(min(df.index)) for df in dfs1])

"reindex of data"
mi = dfs1[0].index.copy()
for i in range(len(dfs1)):
    name = dfs1[i].name
    dfs[i] = dfs1[i].reindex(mi, method='pad')
    dfs[i][np.isnan(dfs[i])] = 0
    dfs[i].name = name

df = pd.concat(dfs1, axis=1, keys=[df.name for df in dfs1], names=['ETFs','Price'])
df

print('cropped from', len(df))
t = max([min(df1.index) for df1 in dfs1])
df = df[df.index > t]
print('to', len(df))
# check NA or not
df.isnull().any()
df = df.fillna(method="pad")
df.describe()
assert np.isfinite(df.as_matrix()).all()


"split"
test_split = 0.20
c = int(len(df.index)*test_split)
split_time = df.index[-c]


df_test = df[df.index > split_time]
df_train = df[df.index <= split_time]
print('test#:',len(df_test), 'train#:', len(df_train), 'test_frac:', len(df_test)/len(df), 'cutoff_time:',split_time)
# df_train.to_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/additional dataset/poloniex_fc.hf', key='train', mode='w', append=False)
# df_test.to_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/additional dataset/poloniex_fc.hf', key='test', mode='a', append=False)
df_train



df_train1 = df_train.drop(['Volume'],axis=1,level='Price')
df_train1.columns = pd.MultiIndex.from_tuples(df_train1.columns.tolist(), names=df_train1.columns.names) # update index to remove dropped cols
df_train1 = df_train1.sort_index(axis=1)

df_test1 = df_test.drop(['Volume'],axis=1,level='Price')
df_test1.columns = pd.MultiIndex.from_tuples(df_test1.columns.tolist(), names=df_test1.columns.names)
df_test1 = df_test1.sort_index(axis=1)
df_test1




" view timeseries "
plt.figure(figsize=(15,16))
for i, d in enumerate(dfs1):
    name = d.name
    x=d.dropna().index
    y=[-i]*len(x)
    plt.scatter(x, y, label=name[:20], s=1)
plt.legend()
plt.show()


data_window = df.copy()
open = data_window.xs('Open', axis=1, level='Price')
plt.show()
# data_window = data_window.divide(open.iloc[-1], level='Pair')
# data_window = data_window.drop('Open', axis=1, level='Price')
data_window.xs('Close', axis=1, level='Price').plot()
plt.show()
data_window.xs('Volume', axis=1, level='Price').plot()
plt.show()




