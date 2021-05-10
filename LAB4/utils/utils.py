
__author__ = 'Majd Jamal'

import numpy as np

class Params:

    def __init__(self, m, seq_length, eta, sig):

        self.m = m # hidden units
        self.seq_length = seq_length
        self.eta = eta  # learning rate
        self.sig = sig  # variance when initializing weights

def softmax(x):
	""" Standard definition of the softmax function """
	e_x = np.exp(x - np.max(x))
	return e_x / np.sum(e_x, axis=0)

def tanh(x):
    """ Standard definition of the tanH function """
    return np.sinh(x) / np.cosh(x)
