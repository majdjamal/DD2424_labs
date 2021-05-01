
import numpy as np
import time
from data.getData import getData
from model import ConvNet
from utils.utils import Params

start = time.time()

data = getData()
params = Params(
    n1 = 4, n2  = 4,
    k1 = 5, k2  = 5,
    eta  = 0.001, roh = 0.9,
    epochs = 0, n_batches = 20)

cnn = ConvNet()
cnn.fit(data, params)


end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms

"""
d: height
k: width
n: filters
"""
