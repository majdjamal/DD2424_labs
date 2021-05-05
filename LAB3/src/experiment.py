
import numpy as np
import time
from data.getData import getData
from model import ConvNet
from utils.utils import Params, plotter

start = time.time()

data = getData()
params = Params(
    n1 = 10, n2  = 10,
    k1 = 5, k2  = 3,
    eta  = 0.003, roh = 0.9,
    epochs = 100, n_batches = 100)

cnn = ConvNet()
cnn.fit(data, params)
weights = cnn.getWeights()  #[F1, F2, W]
np.save('data/weights/weights.npy', weights)
loss_ind, loss = cnn.getLoss()
plotter(loss_ind, loss)
np.save('loss.npy', [loss_ind, loss])

end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms

"""
d: height
k: width
n: filters
"""
