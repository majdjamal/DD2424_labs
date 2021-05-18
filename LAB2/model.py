
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import ReLU, softmax, BatchCreator

class Params:

    """Class used to organizate parameters as an object.
    """
    def __init__(self, epochs, n_batch, n_hidden, lmd, n_s):

        self.epochs = epochs
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.lmd = lmd
        self.n_s = n_s

class MLP:

    """ Two Layer Network used for classification. """

    def __init__(self):

        self.W1 = None  #weight matrix nearest input
        self.W2 = None

        self.b1 = None
        self.b2 = None

        self.n_s = None

        self.memory = {             # Used to store
        'training': [[],[],[]],     # training and valdiation loss
        'validation': [[],[],[]],
        'itr': []}

    def CyclicLearning(self, Npts, Npts_batch, n_max, n_min, iteration):
        """ Cyclic Learning. Computes a learning rate for an iteration.
		:param Npts: Number of total data points
        :param Npts_batch: Number of data points per batch
        :param n_max: maximal learning rate
        :param n_min: minimal learning rate
        :param iterartion: current iteration
        :return: a learning rate, constant
        """

        l = np.floor((iteration + 1)/( 2 * self.n_s))

        if 2 * l * self.n_s <= iteration and iteration <= (2*l + 1)*self.n_s:
            return n_min + ((iteration - 2 * l * self.n_s) / self.n_s ) * (n_max - n_min)
        elif (2*l + 1)*self.n_s <= iteration and iteration <= 2 * (l + 1)*self.n_s:
            return n_max - ((iteration - (2 * l + 1) * self.n_s) / self.n_s ) * (n_max - n_min)
        elif (iteration + 1) % self.n_s == 0:
            return n_min

    def forward(self, X, V, W, bV, bW):
        """ Computes the forward pass.
		:param X: Data matrix, shape = (Ndim, Npts)
        :param V: Weight matrix nearest input, shape = (Nhidden, Ndim)
        :param W: Weight matrix nearest output, shape = (Nout, Nhidden)
        :param bV: Bias term nrst. input, shape = (Nhidden, 1)
        :param bW: Bias term nrst. output, shape = (Nout, 1)
        :return P: Output probabilities, shape = (Nout, Npts)
        :return H: Values of the hidden layer, shape = (Nhidden, Npts)
        """
        H = ReLU(V @ X + bV)
        P = softmax(W @ H + bW)
        return P, H

        self.b1 = np.zeros((hidden_units, 1))
        self.b2 = np.zeros((Nout, 1))

    def loss(self, X, Y, V, W, bV, bW, lmd):
        """ Computes the forward pass.
		:param X: Data matrix, shape = (Ndim, Npts)
        :param Y: One hot matrix, shape =(Nout, Npts)
        :param V: Weight matrix nearest input, shape = (Nhidden, Ndim)
        :param W: Weight matrix nearest output, shape = (Nout, Nhidden)
        :param bV: Bias term nrst. input, shape = (Nhidden, 1)
        :param bW: Bias term nrst. output, shape = (Nout, 1)
		:param lmd: Regularization value
        :return cost: Cost function value, shape = constant
        """

        P, _ = self.forward(X, V, W, bV, bW)
        loss = np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P)))
        #loss = np.mean(-np.log(np.diag(Y.T @ P)))
        reg = lmd * (np.sum(np.square(W)) + np.sum(np.square(V)))
        cost = loss + reg
        return cost, loss

    def backward(self, X, Y, P, H, lmd):
        """  Backward pass to compute the gradients.
		:param X: Data matrix, shape = (Ndim, Npts)
        :param Y: One hot matrix, shape =(Nout, Npts)
        :param P: Output probabilities, shape = (Nout, Npts)
        :param H: Values of the hidden layer, shape = (Nhidden, Npts)
        :param lmd: Regularization value
        :return dL_dW2: Gradients for Weight 2
        :return dL_db2: Gradients for Bias term 2
        :return dL_dW1: Gradients for Weight 1
        :return dL_db1: Gradients for Bias term 1
        """
        _, Npts = P.shape

        G = - (Y - P)

        dL_dW2 = G @ H.T * (1/Npts) + 2 * lmd * self.W2
        dL_db2 = G @ np.ones((Npts, 1)) * (1/Npts)

        G = self.W2.T @ G
        G = G * np.where(H > 0, 1, 0)

        dL_dW1 = G @ X.T * (1/Npts) + 2 * lmd * self.W1
        dL_db1 = G @ np.ones((Npts, 1)) * (1/Npts)

        return dL_dW2, dL_db2, dL_dW1, dL_db1

    def update(self, dL_dW2, dL_db2, dL_dW1, dL_db1, eta):
        """  Updates weights
        :param dL_dW2: Gradients for Weight 2
        :param dL_db2: Gradients for Bias term 2
        :param dL_dW1: Gradients for Weight 1
        :param dL_db1: Gradients for Bias term 1
        :param eta: Learning rate
        """

        self.W2 -= eta * dL_dW2
        self.b2 -= eta * dL_db2

        self.W1 -= eta * dL_dW1
        self.b1 -= eta * dL_db1

    def ComputeAccuracy(self, P, y):
        """Computes a score indicating the network accuracy.
        :param P: Probabilities, shape = (Nout, Npts)
        :param y: One hot matrix, shape = (Npts, 1)
        :return: error rate. low is better.
        """
        out = np.argmax(P, axis=0).reshape(-1,1)
        return 1 - np.mean(np.where(y==out, 0, 1))

    def getMemory(self):
        """ Return loss, cost, and accuracy curves
        from the training phase.
        """
        return self.memory


    def ComputeGradsNum(self, X, Y, W1, W2, b1, b2, lam, h):
        """ Computes numerical gradients. OBS! This code was taken
        from Canvas Discussion.
        """
        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)

        c , _= self.loss(X, Y, W1, W2, b1, b2, lam)

        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] += h
            c2, _ = self.loss(X, Y, W1, W2, b1_try, b2, lam)
            grad_b1[i] = (c2 - c) / h

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i,j] += h
                c2, _ = self.loss(X, Y, W1_try, W2, b1, b2, lam)
                grad_W1[i,j] = (c2 - c) / h

        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] += h
            c2, _ = self.loss(X, Y, W1, W2, b1, b2_try, lam)
            grad_b2[i] = (c2 - c) / h

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i,j] += h
                c2, _ = self.loss(X, Y, W1, W2_try, b1, b2, lam)
                grad_W2[i,j] = (c2 - c) / h

        return [grad_W1, grad_W2, grad_b1, grad_b2]


    def ComputeGradsNumSlow(self, X, Y, W1, W2, b1, b2, lam, h):
        """ Computes numerical gradients. OBS! This code was taken
        from Canvas Discussion.
        """
        grad_W1 = np.zeros(W1.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_W2 = np.zeros(W2.shape)
        grad_b2 = np.zeros(b2.shape)

        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] -= h
            c1, _ = self.loss(X, Y, W1, W2, b1_try, b2, lam)

            b1_try = np.array(b1)
            b1_try[i] += h
            c2, _ = self.loss(X, Y, W1, W2, b1_try, b2, lam)

            grad_b1[i] = (c2 - c1) / (2 * h)

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i,j] -= h
                c1, _ = self.loss(X, Y, W1_try, W2, b1, b2, lam)

                W1_try = np.array(W1)
                W1_try[i,j] += h
                c2, _ = self.loss(X, Y, W1_try, W2, b1, b2, lam)

                grad_W1[i,j] = (c2 - c1) / (2 * h)

        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] -= h
            c1, _ = self.loss(X, Y, W1, W2, b1, b2_try, lam)

            b2_try = np.array(b2)
            b2_try[i] += h
            c2, _ = self.loss(X, Y, W1, W2, b1, b2_try, lam)

            grad_b2[i] = (c2 - c1) / (2 * h)

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i,j] -= h
                c1, _ = self.loss(X, Y, W1, W2_try, b1, b2, lam)

                W2_try = np.array(W2)
                W2_try[i,j] += h
                c2, _ = self.loss(X, Y, W1, W2_try, b1, b2, lam)

                grad_W2[i,j] = (c2 - c1) / (2 * h)

        return [grad_W1, grad_W2, grad_b1, grad_b2]


    def NumericalDifference(self, grad_num, grad_an):
        """ Computes absolute difference between numerical and
        analytical gradient
        :param grad_num: Numerical gradient
        :param grad_an: Analytical gradient
        :return diff: The absolute difference
        """
        diff = np.sum(np.abs(np.subtract(grad_an, grad_num)))
        diff /= np.sum(np.add(np.abs(grad_an), np.abs(grad_num)))
        return diff

    def analyzeGradient(self, X, Y, lmd):
        """ Computes and prints the difference between
        numerical and analytical gradients.
        :param X: Data matrix, shape = (Ndim, Npts)
        :param Y: One hot matrix, shape = (Nout, Npts)
        :param lmd: Lambda, regularization value
        :param h:
        """

        ind = np.arange(100)    # Generates indicies for 100 points.

        grad_num_W1, grad_num_W2, grad_num_b1, grad_num_b2 = self.ComputeGradsNumSlow(
        X[:, ind], Y[:, ind],
        self.W1, self.W2, self.b1, self.b2, lmd, 1e-5)

        PBatch, HBatch = self.forward(X[:, ind], self.W1,
        self.W2, self.b1, self.b2)

        grad_an_W2, grad_an_b2, grad_an_W1, grad_an_b1 = self.backward(
        X[:, ind], Y[:, ind], PBatch, HBatch, lmd)

        w1_diff = self.NumericalDifference(grad_num_W1, grad_an_W1)
        w2_diff = self.NumericalDifference(grad_num_W2, grad_an_W2)
        b1_diff = self.NumericalDifference(grad_num_b1, grad_an_b1)
        b2_diff = self.NumericalDifference(grad_num_b2, grad_an_b2)

        print(w1_diff, w2_diff, b1_diff, b2_diff)

    def evaluate(self, data, lmd):
        #Training X, Y, lmd
        c, l = self.loss(data.X_train, data.Y_train, self.W1, self.W2, self.b1, self.b2, lmd)  #Cost and Loss
        P, _ = self.forward(data.X_train, self.W1, self.W2, self.b1, self.b2,)
        acc = self.ComputeAccuracy(P, data.y_train) #Accuracy

        #Storing loss, cost, and accuracy in memory
        self.memory['training'][0].append(c)
        self.memory['training'][1].append(l)
        self.memory['training'][2].append(acc)

        #Validation
        c_val, l_val = self.loss(data.X_val, data.Y_val, self.W1, self.W2, self.b1, self.b2, lmd)  #Cost and Loss
        P_val, _ = self.forward(data.X_val, self.W1, self.W2, self.b1, self.b2)
        acc_val = self.ComputeAccuracy(P_val, data.y_val) #Accuracy

        #Storing loss, cost, and accuracy in memory
        self.memory['validation'][0].append(c_val)
        self.memory['validation'][1].append(l_val)
        self.memory['validation'][2].append(acc_val)

    def BatchCreator(self, j, n_batches):
        """Creates indicies for a mini_batch, given
        an iteration state and number of data points in a batch.
        :param j: Iteration index
        :param n_batches: Number of data points in a batch.
        :return ind: Data point indicies to create a mini_batch
        """

        j_start = (j-1)*n_batches + 1
        j_end = j*n_batches + 1
        ind = np.arange(start= j_start, stop=j_end, step=1)
        return ind

    def fit(self, data, params):
        """ This function is called to start trining.
        :param data: Object containing training and validation data
        :param params: Object containing parameters used for training, i.e.
        epochs, n_batch, n_hidden, lmd, n_s.
        """

        #Data
        X = data.X_train
        Y = data.Y_train
        y = data.y_train

        #Parameters
        Ndim, Npts = X.shape
        Nout, _ = Y.shape
        hidden_units = params.n_hidden
        lmd = params.lmd
        epochs = params.epochs
        n_batches = params.n_batch
        self.n_s = params.n_s

        np.random.seed(400)

        #Weights
        self.W1 = np.random.normal(0, 1/np.sqrt(Ndim),
        size = (hidden_units, Ndim))

        self.W2 = np.random.normal(0, 1/np.sqrt(hidden_units),
        size =  (Nout, hidden_units))

        #Biases
        self.b1 = np.zeros((hidden_units, 1))
        self.b2 = np.zeros((Nout, 1))

        # Open to analyze gradients
        #self.analyzeGradient(X, Y, lmd)

        print('=-=- Starting Training -=-=')
        for epoch in range(epochs):

            for j in range(round(Npts/n_batches)):

                #Cyclic Learning
                itr = j + (Npts*epoch/n_batches + 1)
                eta = self.CyclicLearning(Npts, n_batches, 1e-1, 1e-5, itr)

                #Create mini_batch
                ind = self.BatchCreator(j, n_batches)
                XBatch = X[:, ind]
                YBatch = Y[:, ind]

                #Training
                PBatch, HBatch = self.forward(XBatch, self.W1, self.W2, self.b1, self.b2)
                gradients = self.backward(XBatch, YBatch, PBatch, HBatch, lmd)
                self.update(*gradients, eta)

                #Evaluation
                #Makes computation for every 100 update step
                if itr == 1 or itr % 100 == 0:
                    self.evaluate(data, lmd)
                    self.memory['itr'].append(itr)


            print('Epoch: ', epoch)

        print('=-=- Training Completed -=-=')


    def TestAccuracy(self, X, Y, y):
        """ Computes accuracy of the model, i.e. error rate.
        :param X: Test Data matrix, shape = (Ndim, Npts)
        :param Y: Test One hot matrix, shape =(Nout, Npts)
        :param y: Test One hot matrix, shape = (Npts, 1)
        """
        P, _ = self.forward(X, self.W1, self.W2, self.b1, self.b2)
        score = self.ComputeAccuracy(P, y)
        return score
