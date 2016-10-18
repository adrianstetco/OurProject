import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import math
from talib import abstract
from numpy import *
from ModelPKG import ModelFactory
matplotlib.style.use('ggplot')
from PreProc import *
from Visualise import *


TALibFuncNames = np.array(pd.read_csv('TA-list2.txt'))
ts = readts('EURAUDSmall.txt', norm="TRUE")
ts = ts.ix[1:155000]
#test = testGenMatrix(ts, TALibFuncNames)

ActiveTALibFeatures = [1] * 31
DataSet = generateMatrix(ts, ActiveTALibFeatures, TALibFuncNames)


examples = generateExamples(DataSet,4)

examples = np.vstack({tuple(row) for row in examples})

print(examples.shape)
train, test = split(examples, 0.7)

DeepNet = ModelFactory.Factory.create("DeepNet")
DeepNet.train(train)
DeepNet.test(test)

#LinearModel = ModelFactory.Factory.create("DeepNet")
#LinearModel.train(train, test)
#LinearModel.test(test)






