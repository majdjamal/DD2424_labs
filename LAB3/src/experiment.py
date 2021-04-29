
import numpy as np
import time
from data.getData import getData
from model import ConvNet
from utils.utils import Params

start = time.time()

data = getData()
params = Params(
    NF1 = 4, NF2  = 4,
    widthF1 = 5, widthF2  = 5,
    eta  = 0.001, roh = 0.9, epochs = 10, n_batches = 20)

cnn = ConvNet()
cnn.fit(data, params)


end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms
