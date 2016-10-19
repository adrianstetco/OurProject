import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from talib import abstract
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

# generate examples from matrix

def generateExamples(xs, howMuchLookAhead):
    examples = np.zeros(shape=(xs.shape[0], xs.shape[1] + 1))
    for i in range(xs.shape[0] - howMuchLookAhead):
        sum = 0
        x = xs[i, :];  # current row
        x[np.isnan(x)] = 0  # how do we replace nans?; currently with zeros
        # -1 sell; 0 hold; 1 buy
        # look ahead, if howMany=3 and the next 3 prices are rising then Buy (2), if next 3 dropping = Sale(-1) else Hold (0)
        for j in range(howMuchLookAhead):
            if (xs[i + j, 0] < xs[i + j + 1, 0]): sum = sum + 1
            if (xs[i + j, 0] > xs[i + j + 1, 0]): sum = sum - 1
        examples[i, 0:len(x)] = x
        if (sum == howMuchLookAhead):  examples[i, len(x)] = 1
        if (sum == -howMuchLookAhead): examples[i, len(x)] = 0
        if ((sum < howMuchLookAhead) and (sum > -howMuchLookAhead)): examples[i, 0:len(x)] = 0
    return examples


#generate matrix of Xs to be input in the Neural Net
def generateMatrix(ts, requiredTALibFunctions, TALibFuncName):
    xs = ts.values
    inputs = {'close': np.array(xs, dtype='f8')}
     # create ndarray with price values
    # requiredTALibFunctions = [0, 1, 0 ,0, ... , 0, 0]
    for i in range(len(requiredTALibFunctions)):
        if requiredTALibFunctions[i] != 0:
            TALibFunction = abstract.Function(TALibFuncName[i][0])
           # print(TALibFuncName[i][0])
            TALibResult = TALibFunction(inputs)  # !!parameters need to be adjusted for each TALib function
            xs = np.column_stack((xs, TALibResult)) # appends columns containing TALib func results
                                                    # to the currency pair closing price value
    return xs.astype(np.float)


# test generate Matrix
# this tests the addition of TALib functions as columns to the Xs
def testGenMatrix(ts, TALibFuncNames):
    hotOneTALib = [1] * 31
    #hotOneTALib[2]=0
    xs = generateMatrix(ts, hotOneTALib, TALibFuncNames)
    return xs