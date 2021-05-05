
import numpy as np
import matplotlib.pyplot as plt

class Params:
	""" Object that are used to pass arguments to the deep network.
	"""
	def __init__(self, n1, n2, k1, k2, eta, roh, epochs, n_batches):

		self.n1 = n1
		self.n2 = n2

		self.k1 = k1
		self.k2 = k2

		self.eta = eta
		self.roh = roh

		self.epochs = epochs
		self.n_batches = n_batches

def softmax(x):
	""" Standard definition of the softmax function """
	e_x = np.exp(x - np.max(x))
	return e_x / np.sum(e_x, axis=0)

def ReLU(x):
    """ Standard definition of the ReLU function """
    return np.maximum(x, 0)

def BatchCreator(j, n_batches):
    #Create mini_batch
    j_start = (j-1)*n_batches + 1
    j_end = j*n_batches + 1
    ind = np.arange(start= j_start, stop=j_end, step=1)
    return ind

def vecX(x, d, nlen):
    """ Vectorizes an x-data point in a similar fashion to Matlab.
	:param x: data point, with shape = (Ndim, )
	:param d: height of the original shape
	:param nlen: width - "" -
	:return: vectorized version of data point x
    """
    return x.reshape((d, nlen)).flatten(order = 'F')

def vecF(F):
    """ Vectorizes an x-data point in a similar fashion to Matlab.
	:param x: Filter, with shape (#filters, width, height)
	:return: vectorized filter in 1D
    """
    nf,_,_ = F.shape
    for filter in range(nf):

    	if filter == 0:
    		F_flattened = F[filter].flatten(order = 'F')
    	else:
    		F_flattened = np.hstack((F_flattened, F[filter].flatten(order = 'F')))

    return F_flattened

def plotter(X, Y):
	""" Plotting the loss function.
	:param X: x-coordinates, which are used as update steps
	:param Y: loss values
	"""
	plt.style.use('seaborn')
	plt.xlabel('update steps')
	plt.ylabel('loss')
	plt.plot(X, Y, color = 'red', label = 'loss')
	plt.legend()
	plt.savefig('result/loss')
	plt.close()
