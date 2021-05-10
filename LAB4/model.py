
__author__ = 'Majd Jamal'

import numpy as np
from utils.utils import tanh, softmax

class VRNN:
    """ Vanilla Recurrent Neural Network
        used for natural language processing.
    """
    def __init__(self):

        self.W = None
        self.V = None
        self.U = None
        self.b = None
        self.c = None

        self.h_0 = None

        self.char_to_ind = None
        self.ind_to_char = None

    def forward(self, X, h_t_1):
        """ Computes the forward pass
        """

        a_t = self.W @ h_t_1 0 self.U@X + self.b
        h_t = tanh(a_t)
        o_t = self.V@h_t + self.c
        p_t = softmax(o_t)

        return o_t, p_t

    def backward(self):
        """
        ##  TODO: Write backward pass with instructions
        ##  from lecture 9. Solve the gradient for the bias terms.
        ##
        """
        pass


    def loss(self, Y, P):
        """ Computes loss
        """
        return np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P)))

    def fit(self, data, params):

        ##
        ## Unpacking data
        ##
        book_data = data.book_data
        K = data.NUnique
        self.char_to_ind = data.char_to_ind
        self.ind_to_char = data.ind_to_char

        ##
        ##  Unpacking params
        ##
        m = params.m
        seq_length = params.seq_length
        eta = params.eta
        sig = params.eta

        e = 0 # Tracker

        ##
        ##  Initialize weights and
        ##  biases
        ##
        self.U = np.random.randn(m, K) * sig
        self.W = np.random.randn(m, m) * sig
        self.V = np.random.randn(K, m) * sig

        X_chars = book_data[e:seq_lenght]
        Y_chars = book_data[e + 1 :seq_lenght + 1]
