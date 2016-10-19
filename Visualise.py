import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from talib import abstract
matplotlib.style.use('ggplot')




# visualize price time series together with indicator in the same plot
def visualize(ts, TALibFuncName):
  inputs = {'close': np.array(ts.values,dtype='f8')}
  TALibFunction = abstract.Function(TALibFuncName[0])
  TALibResult = TALibFunction(inputs,60) #parameter for SMA
  concat = np.column_stack([ts.values, TALibResult])
  df = pd.DataFrame(concat, index=ts.index, columns= ['Price',TALibFuncName[0]])
  df.plot(); plt.show()
