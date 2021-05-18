
__author__ = 'Majd Jamal'

import numpy as np
from data.getData import getData
from utils.utils import Params, plotter
from model import VRNN

data = getData()
params = Params(
    m = 100, seq_length = 25, eta = 0.1, sig = 0.1, epochs = 10
    )

vrnn = VRNN()
vrnn.fit(data, params)
loss = vrnn.getLoss()
plotter(loss[0], loss[1])
weigths = vrnn.getWeigths()

np.save('weigths.npy', weigths)
np.save('loss.npy', loss)
