
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt

class Params:

    def __init__(self, m, seq_length, eta, sig, epochs):

        self.m = m # hidden units
        self.seq_length = seq_length
        self.eta = eta  # learning rate
        self.sig = sig  # variance when initializing weights
        self.epochs = epochs

def softmax(x):
	""" Standard definition of the softmax function """
	e_x = np.exp(x - np.max(x))
	return e_x / np.sum(e_x, axis=0)

def tanh(x):
    """ Standard definition of the tanH function """
    return np.sinh(x) / np.cosh(x)

def plotter(step, val):
    """ Plots validation loss from a training session.
    :param step: update steps
    :param val: validation loss
    """
    plt.style.use('seaborn')
    plt.plot(step, train, color = 'red', label = 'Validation Loss')
    plt.xlabel('Update step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('utils/results/loss')
    plt.close()
