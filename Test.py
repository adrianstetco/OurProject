import numpy as np
import pandas as pd
from ModelPKG import ModelFactory
from PreProc import readts, generateMatrix, generateExamples, split, split2, blockshaped, label

TALibFuncNames = np.array(pd.read_csv('TA-list2.txt'))
ts = readts('EURAUDSmall.txt', norm="TRUE")
ts = ts.ix[1:155000]
#test = testGenMatrix(ts, TALibFuncNames)

ActiveTALibFeatures = [1] * 31
DataSet = generateMatrix(ts, ActiveTALibFeatures, TALibFuncNames)


examples = generateExamples(DataSet,4)
examples = np.vstack({tuple(row) for row in examples})


#DeepNet = ModelFactory.Factory.create("DeepNet")
#trainDN, testDN = split(examples, 0.7)
#DeepNet.train(trainDN)
#DeepNet.test(testDN)

LSTM = ModelFactory.Factory.create("LSTM")
windowSize = 33 #how far ahead are we looking to get the "whole" picture
featureSize = 33 #these are the TALib features (for LSTM, these should be taken integrally i.e. now = 33
trainLSTM = split2(examples, windowSize) #Select the maximum amount of columns that are divisible with the size of the frame
trainLSTM = blockshaped(trainLSTM.T, featureSize, windowSize) #return an array of "pictures" of size featureSize x windowSize
labelsLSTM = label(trainLSTM)
print(labelsLSTM)
LSTM.train(trainLSTM)
#LSTM.test(test)

#LinearModel = ModelFactory.Factory.create("DeepNet")
#LinearModel.train(train, test)
#LinearModel.test(test)






