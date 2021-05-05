
import numpy as np
import time
from data.getData import getData
from model import ConvNet
from utils.utils import Params, plotter

start = time.time()

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#   Setup
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
data = getData()
params = Params(
    n1 = 20, n2  = 20,
    k1 = 5, k2  = 3,
    eta  = 0.001, roh = 0.9,
    epochs = 3, n_batches = 108)
cnn = ConvNet()

#=-=-=-=-=-=-=
#   Training
#=-=-=-=-=-=-=
cnn.fit(data, params)
weights = cnn.getWeights()  #[F1, F2, W]
loss_ind, loss = cnn.getLoss()
plotter(loss_ind, loss)

#Save weights
np.save('data/weights/weights.npy', weights)
np.save('loss.npy', [loss_ind, loss])

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#   Classification test with
#   the best network settings
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print('=-=-=-=-=-=-=-=-=-=-=-=- Test -=-=-=-=-=-=-=-=-=-=-=-=')
test = ['linda', 'per', 'majd', 'alba', 'steve']
for name in test:
    labels, probabilities = cnn.predict(name)
    print('Name: ', name)
    print('lbl ', labels)
    print('pr ', probabilities)
    print('\n')


end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms
