
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

        self.h_t_1 = None
        self.h_t = None

        self.char_to_ind = None
        self.ind_to_char = None

        self.AdaGradTerm = 0

    def forward(self, X):
        """ Computes the forward pass with
            tanH and softmax activations.
        :param X: matrix representation of
                  a character sequence.
                  shape = (K, Npts)
        :return o_t: non-normalized output probabilities,
                  shape = (K, Npts)
        :return p_t: normalized probabilities,
                    shape = (K, Npts)
        """

        a_t = self.W @ self.h_t_1 + self.U@X + self.b
        h_t = tanh(a_t)
        o_t = self.V@h_t + self.c
        p_t = softmax(o_t)

        return a_t, h_t, o_t, p_t

    def backward(self, Y, P, h_t):
        """
        ##  TODO: Write backward pass with instructions
        ##  from lecture 9. Solve the gradient for the bias terms.
        ##
        """

        G = - (Y - P)

        dL_dV = h_t @ G

        pass

    def update(self, dL_dV, dL_dW, dL_dU, eta, eps = 0.001):

        m = self.AdaGradTerm + np.square(dL_dV)

        self.V -= eta/np.sqrt(m + eps) * dL_dV




    def loss(self, Y, P):
        """ Computes loss
        """
        return np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P)))

    def fit(self, data, params):

        ##
        ## Unpacking data
        ##
        book_data = data.book_data
        X = data.X
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
        self.b = np.zeros((m,1))
        self.c = np.zeros((K,1))

        ##
        ##  Initializing hidden states
        ##
        self.h_t_1 = np.zeros((m,1))

        ##
        ##  Forward pass
        ##
        X_train = X[:, e:e+seq_length]
        Y_train = X[:, e + 1 :e + 1 +seq_length]
        a_t, h_t, o_t, p_t = self.forward(X_train)
        loss = self.loss(Y_train, p_t)
