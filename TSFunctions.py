import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import math
import talib as tb
matplotlib.style.use('ggplot')

np.set_printoptions(threshold=np.nan)

# READS FOREX HISTORIC DATA INTO A NUMPY ARRAY AND NORMALIZES IF REQUIRED
def readts(path, norm="FALSE"):
    df = pd.read_csv(path, sep=',', header=None, parse_dates=[['date', 'time']], usecols=[0, 1, 2, 3, 4, 5],
                         names=['date', 'time', 'value0', 'value1', 'value2', 'value3'])
    df = df.iloc[:, [0, 3]].as_matrix()
    ts = pd.Series(df[:, 1], index=df[:, 0])
    #if norm == "TRUE": df[:, 1] = df[:, 1] / np.linalg.norm(df[:, 1])
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


ts = readts('C:\Users\Adrian\Desktop\EURAUDSmall.txt', norm="TRUE")
#converted = ts.asfreq('60Min', method='pad')

#print(ts.values)
SMA= tb.SMA(np.array(ts.values,dtype='f8'))
pds =pd.Series(SMA)
#print(len(ts))
#print(len(SMA))
#print(np.randn(1000, 4))
#df = pd.DataFrame([ts.values,SMA], index=ts.index, columns=list('AB'))
pds.plot(legend=True);
plt.show()
#train, test = split(s, 0.6)
#generate(train,5)


