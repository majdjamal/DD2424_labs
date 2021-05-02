
import numpy as np
import time
from data.getData import getData
from model import ConvNet
from utils.utils import Params

start = time.time()

data = getData()
params = Params(
    n1 = 10, n2  = 10,
    k1 = 5, k2  = 5,
    eta  = 0.001, roh = 0.9,
<<<<<<< HEAD
    epochs = 5, n_batches = 5)
=======
    epochs = 5, n_batches = 130)
>>>>>>> 8e5cc9a69312625cf22a96db304da2b54bc3727e

cnn = ConvNet()
cnn.fit(data, params)


end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms

"""
d: height
k: width
n: filters
"""
