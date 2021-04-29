
import numpy as np

class Params:

	def __init__(self, NF1, NF2, widthF1, widthF2, eta, roh, epochs,n_batches):

		self.NF1 = NF1
		self.NF2 = NF2

		self.widthF1 = widthF1
		self.widthF2 = widthF2

		self.eta = eta
		self.roh = roh

		self.epochs = epochs
		self.n_batches = n_batches

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x):
    """ Standard definition of the ReLU function """
    return np.maximum(x, 0)

def BatchCreator(j, n_batches):
    #Create mini_batch
    j_start = (j-1)*n_batches + 1
    j_end = j*n_batches + 1
    ind = np.arange(start= j_start, stop=j_end, step=1)
    return ind
