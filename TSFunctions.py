import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import math
import talib as tb
from talib import abstract
from talib.abstract import *
matplotlib.style.use('ggplot')

np.set_printoptions(threshold=np.nan)

# READS FOREX HISTORIC DATA INTO A NUMPY ARRAY AND NORMALIZES IF REQUIRED
def readts(path, norm="FALSE"):
    df = pd.read_csv(path, sep=',', header=None, parse_dates=[['date', 'time']], usecols=[0, 1, 2, 3, 4, 5],
                         names=['date', 'time', 'value0', 'value1', 'value2', 'value3'])
    df = df.iloc[:, [0, 3]].as_matrix()
    ts = pd.Series(df[:, 1], index=df[:, 0])
    #if norm == "TRUE": df[:, 1] = df[:, 1] / np.linalg.norm(df[:, 1]) not yet
    return ts

# SPLITS NUMPY ARRAY INTO TRAINING AND TESTING SETS
def split(dataset, percentagetraining):
    train_size = int(len(dataset) * percentagetraining)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, test

# generates examples of the form X1,X2,...,Xn Y from dataset, for supervised learning
def generate(dataset, n, timestamps="FALSE"):
    rows = math.trunc(len(dataset)/(n+1))
    dataset = dataset[0:rows*(n+1)]
    dataset= np.reshape(dataset,(-1,(n+1) *2 ))
    print(dataset)

# visualize price time series together with indicator in the same plot
def visualize(ts, TALibFuncName):
  inputs = {'close': np.array(ts.values,dtype='f8')}
  TALibFunction= abstract.Function(TALibFuncName[0])
  TALibResult= TALibFunction(inputs,60) #parameter for SMA
  concat= np.column_stack([ts.values, TALibResult])
  df = pd.DataFrame(concat, index=ts.index, columns= ['Price',TALibFuncName[0]])
  df.plot(); plt.show()


#generate matrix of Xs to be input in the Neural Net
# this tests the addition of TALib functions as columns to the Xs
def testGenMatrix(ts, TALibFuncNames):
    hotOneTALib = np.zeros(150)
    hotOneTALib[1] = 1 
    hotOneTALib[0] = 1 
    hotOneTALib[3] = 1 
    xs = generateMatrix(ts, hotOneTALib, TALibFuncNames) 
    print(xs)

# converted = ts.asfreq('60Min', method='pad') 
#ts=ts.ix[1:50]
#train, test = split(s, 0.6)
#generate(train,5)
#print(ts.values)
#print(SMA)

TALibFuncNames = np.array(pd.read_csv('TA-list.txt')) 
ts = readts('EURAUDSmall.txt', norm="TRUE") 
ts = ts.ix[1:500]
 testGenMatrix(ts, TALibFuncNames) 
#visualize(ts, TALibFuncNames[1])
