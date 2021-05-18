
import pickle
import numpy as np

def LoadBatch(file):
	"""Loads data from files
	:param file: path to database
	:return X: Data matrix, shape = (Ndim, Npts)
	:return Y: One hot vector Matrix, shape = (Nout, Npts)
	:return y: Labels for data points
	"""

	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')

	#dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
	storage = dict

	#X
	X = storage[b'data']
	X = X.T
	#y
	y = storage[b'labels']
	y = np.array(y).reshape(-1,1)

	#Y
	Y = np.zeros((y.size, y.max()+1))
	Y[np.arange(y.size), np.reshape(y, (1,-1))] = 1
	Y = Y.T

	return X, Y, y
