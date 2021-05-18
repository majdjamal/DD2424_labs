
import numpy as np
import matplotlib.pyplot as plt
from functions import softmax, ComputeGradsNum, ComputeGradsNumSlow, montage, save_as_mat
from data.data import LoadBatch
from utils import centering, softmax
from model import SLN, GDparams

class Data:
    """ This class is used to send training and validation data
        to a function with an object, instead of multiple parameters.
        Train - training, val - validation"""
    def __init__(self, X_train, Y_train, y_train, X_val, Y_val, y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.y_train = y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.y_val = y_val


def plotter(train, val, title):
    """ Plotter function to plot training and validation loss.
    :param train: training loss, this should be a list.
    :param val: validaition loss, this should be a list."""
    plt.style.use('seaborn')
    plt.xlabel('epoch')
    plt.ylabel(str(title))
    plt.plot(train, color = 'green', label = "training " + str(title))
    plt.plot(val, color = 'red', label = "validation " + str(title))
    plt.legend()
    plt.show()
    #plt.savefig('result/loss')
    #plt.close()

#=-=-=-=-=-=-=-
# Experiments
#=-=-=-=-=-=-=-

#=-=-=-=-=-=-=-=-
# 1. Load Data
#=-=-=-=-=-=-=-=-
X_train, Y_train, y_train = LoadBatch('data/data_batch_1')
X_val, Y_val, y_val = LoadBatch('data/data_batch_2')
X_test, Y_test, y_test = LoadBatch('data/test_batch')


#=-=-=-=-=-=-=-=-=-=-=-=-
# 2. Pre-processing.
#=-=-=-=-=-=-=-=-=-=-=-=-
X_train = centering(X_train)
X_val = centering(X_val)
X_test = centering(X_test)



#=-=-=-=-=-=
# 3. Train.
#=-=-=-=-=-=

#Test 1
#params = GDparams(n_batches = 100, eta = 0.1, n_epochs = 40, lmd=0.)

#Test 2
#params = GDparams(n_batches = 100, eta = 0.001, n_epochs = 40, lmd=0.)

#Test 3
params = GDparams(n_batches = 100, eta = 0.001, n_epochs = 40, lmd=0.1)

#Test 4
#params = GDparams(n_batches = 100, eta = 0.001, n_epochs = 40, lmd=1.)

data = Data(X_train, Y_train, y_train, X_val, Y_val, y_val)

slp = SLN()
slp.fit(data, params)

cost_train, cost_val = slp.getCost()
loss_train, loss_val = slp.getLoss()


W, b = slp.getParams()



score = slp.TestAccuracy(X_test, Y_test, y_test)
print('Score: ' + str(score))

#montage(W)
plotter(cost_train, cost_val, 'cost')
plotter(loss_train, loss_val, 'loss')
