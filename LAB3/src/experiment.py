
import numpy as np
import time
from data.getData import getData
from model import ConvNet
from utils.utils import Params, plotter

start = time.time()

data = getData()
params = Params(
    n1 = 20, n2  = 20,
    k1 = 5, k2  = 3,
    eta  = 0.001, roh = 0.9,
    epochs = 15, n_batches = 130)

cnn = ConvNet()
cnn.fit(data, params)
weights = cnn.getWeights()  #[F1, F2, W]
np.save('data/weights/weights.npy', weights)
loss_ind, loss = cnn.getLoss()
plotter(loss_ind, loss)


end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms

"""
d: height
k: width
n: filters
"""
