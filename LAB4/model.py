
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

        self.AdaGradTerm = {}

        self.lossData = [[],[],[]] # [step, train_loss, val_loss]

    def forward(self, X, V, U, W, b, c):
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

        a_t = W @ self.h_t_1 + U@X + b
        H = tanh(a_t)
        O = V@H + c
        P = softmax(O)

        return a_t, H, O, P

    def backward(self, X, Y, P, H, a_t):
        """
        ##  TODO: Write backward pass with instructions
        ##  from lecture 9. Solve the gradient for the bias terms.
        ##
        """

        _, Npts = Y.shape
        dL_dU, dL_dW, dL_db = 0,0,0

        G = - (Y - P) # dL_do

        dL_dV = G @ H.T * (1/Npts)

        dL_dc = G @ np.ones((Npts, 1)) * (1/Npts)

        #"""

        new_G = 0
        prev_a_T = 0
        dL_dh = 0


        for i in range(Npts-1, -1, -1):

            g = G[:, i].reshape(-1,1)


            if i == Npts - 1:

                dL_dh = self.V.T @ g    # EQ: dL_do * V

                #prev_a_T = diag.T @ prev_a_T  # EQ: dL_do
                new_G = dL_dh

            else:
                diag = np.diag(1 - np.square(np.tanh(a_t[:, i + 1])))
                dL_dh = self.V.T @ g + self.W.T @ (diag.T @ dL_dh)# EQ: dL_do * V + dL_da_{t+1} * W
                #prev_a_T = diag.T @ dL_dh

                new_G = np.hstack((new_G, dL_dh))

        G = new_G
        #G = np.fliplr(new_G)

        # dh = g @ V + da_prev @ W
        # da = dh * diag
        #"""

        #G = self.V.T @ G    #dL_dh
        #print(G.shape)

        #G = G * np.diag(1 - np.square(np.tanh(a_t))) #dL_dat

        dL_db = G @ np.ones((Npts, 1)) * (1/Npts)

        dL_dW = G @ self.h_t_1.T * (1/Npts)
        dL_dU = G @ X.T * (1/Npts)
        #"""

        return dL_dV, dL_dU, dL_dW, dL_db, dL_dc

    def update(self, dL_dV, dL_dU, dL_dW, dL_db, dL_dc, eta, eps = 1e-8):

        m_V = self.AdaGradTerm['V'] + np.square(dL_dV) + eps
        m_U = self.AdaGradTerm['U'] + np.square(dL_dU) + eps
        m_W = self.AdaGradTerm['W'] + np.square(dL_dW) + eps
        m_b = self.AdaGradTerm['b'] + np.square(dL_db) + eps
        m_c = self.AdaGradTerm['c'] + np.square(dL_dc) + eps

        self.V -= eta/np.sqrt(m_V) * dL_dV
        self.U -= eta/np.sqrt(m_U) * dL_dV
        self.W -= eta/np.sqrt(m_W) * dL_dV

        self.b -= eta/np.sqrt(m_b) * dL_db
        self.c -= eta/np.sqrt(m_c) * dL_dc

    def loss(self, X, Y, V, U, W, b, c):
        """ Computes loss
        """
        _,_,_, P = self.forward(X, V, U, W, b, c)
        return np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P)))

    def getLoss(self):
        """ Returns loss from the training
        """
        return self.lossData

    def NumericalVSAnalytic(self, grad_num, grad_an):
        """Computes absolute difference between numerical and analytical gradient
        """
        diff = np.sum(np.abs(np.subtract(grad_an, grad_num)))
        diff /= np.sum(np.add(np.abs(grad_an), np.abs(grad_num)))
        return diff

    def ComputeGradsNum(self, X, Y, V, U, W, b, c, h = 1e-4):

        # V
        dL_dV = np.zeros(V.shape)
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):

                V_try = np.array(V)
                V_try[i, j] -= h
                c1 = self.loss(X, Y, V_try, U, W, b, c)

                V_try = np.array(V)
                V_try[i, j] += h
                c2 = self.loss(X, Y, V_try, U, W, b, c)

                dL_dV[i,j] = (c2 - c1) / (2 * h)

        # U
        dL_dU = np.zeros(U.shape)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_try = np.array(U)
                U_try[i, j] -= h
                c1 = self.loss(X, Y, V, U_try, W, b, c)

                U_try = np.array(U)
                U_try[i, j] += h
                c2 = self.loss(X, Y, V, U_try, W, b, c)

                dL_dU[i,j] = (c2 - c1) / (2 * h)

        # W
        dL_dW = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i, j] -= h
                c1 = self.loss(X, Y, V, U, W_try, b, c)

                W_try = np.array(W)
                W_try[i, j] += h
                c2 = self.loss(X, Y, V, U, W_try, b, c)

                dL_dW[i,j] = (c2 - c1) / (2 * h)

        # b
        dL_db = np.zeros(b.shape)
        for i in range(b.size):
            b_try = np.array(b)
            b_try[i] -= h
            c1 = self.loss(X, Y, V, U, W, b_try, c)

            b_try = np.array(b)
            b_try[i] += h
            c2 = self.loss(X, Y, V, U, W, b_try, c)

            dL_db[i] = (c2 - c1) / (2 * h)

        # c
        dL_dc = np.zeros(c.shape)
        for i in range(c.size):
            c_try = np.array(c)
            c_try[i] -= h
            c1 = self.loss(X, Y, V, U, W, b, c_try)

            c_try = np.array(c)
            c_try[i] += h
            c2 = self.loss(X, Y, V, U, W, b, c_try)

            dL_dc[i] = (c2 - c1) / (2 * h)

        return dL_dV, dL_dU, dL_dW, dL_db, dL_dc

    def AnalyzeGradients(self, X, Y):

        dL_dV_num, dL_dU_num, dL_dW_num, dL_db_num, dL_dc_num = self.ComputeGradsNum(
        X, Y, self.V, self.U, self.W, self.b, self.c
        )

        a_t, h_t, o_t, p_t = self.forward(
        X, self.V, self.U, self.W, self.b, self.c)

        dL_dV_an, dL_dU_an, dL_dW_an, dL_db_an, dL_dc_an = self.backward(
        X, Y, p_t, h_t, a_t
        )

        """
        print('*')
        print(dL_db_num)
        print('*')
        print('\n'*2)
        print('*')
        print(dL_db_an)
        """

        V_d = self.NumericalVSAnalytic(dL_dV_num, dL_dV_an)
        U_d = self.NumericalVSAnalytic(dL_dU_num, dL_dU_an)
        W_d = self.NumericalVSAnalytic(dL_dW_num, dL_dW_an)
        b_d = self.NumericalVSAnalytic(dL_db_num, dL_db_an)
        c_d = self.NumericalVSAnalytic(dL_dc_num, dL_dc_an)

        print(" =-=-=-=- Numerical vs Analytic gradients -=-=-=-= ")
        print(" =-=- V: ", V_d)
        print(" =-=- U: ", U_d)
        print(" =-=- W: ", W_d)
        print(" =-=- b: ", b_d)
        print(" =-=- c: ", c_d)
        print(" =-=-=-=- @ -=-=-=-= ")

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
        sig = params.sig
        epochs = params.epochs

        e = 0 # Tracker

        ##
        ##  Initialize weights and
        ##  biases
        ##
        #np.random.seed(400)
        self.U = np.random.randn(m, K) * sig
        self.W = np.random.randn(m, m) * sig
        self.V = np.random.randn(K, m) * sig
        self.b = np.zeros((m,1))
        self.c = np.zeros((K,1))

        ##
        ##  Initializing hidden states
        ##
        #self.h_t_1 = np.zeros((m,1))
        self.h_t_1 = np.random.randn(m,seq_length - 10)

        ## Analyze gradients
        self.AnalyzeGradients(X[:, 0:seq_length - 10], X[:, 1:1 + seq_length - 10])

        print(' =-=-=-=- Network parameters -=-=-=-= ')
        print(' =- epochs: ', epochs, ' learning_rate: ', eta)
        print(' =- hidden_units: ', m, ' seq_length: ', seq_length)
        print(' =-=-=-=- Starting training -=-=-=-= \n')

        """ Training
        for epoch in range(epochs):
            for itr in range(X.shape[1] - seq_length - 1):
                X_train = X[:, e:e+seq_length]
                Y_train = X[:, e + 1 :e + 1 +seq_length]

            print('Epoch: ', epoch)

        print('=-=- Training Completed -=-=')
        """
