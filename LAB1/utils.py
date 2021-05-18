
import numpy as np
import numpy.matlib as mb
import matplotlib.pyplot as plt

def centering(X):
	"""Transform data matrix to have zero mean.
	:return X: Data matrix, shape = (Ndim, Npts)
	:return X: Data matrix with zero mean.
	"""
	Ndim, Npts = X.shape

	mu = np.mean(X, axis=1).reshape(-1,1)
	sig = np.std(X, axis=1).reshape(-1,1)

	X = X - mu
	X = X / sig

	return X

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)
	#e_x = np.exp(x - np.max(x))
	#return e_x / e_x.sum(axis=0)
