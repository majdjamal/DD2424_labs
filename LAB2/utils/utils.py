
import numpy as np
import numpy.matlib as mb
import matplotlib.pyplot as plt


class Data:
    """ This class is used to send training and validation data
        to a function with an object, instead of multiple parameters.
        Train - training, val - validation"""
    def __init__(self, X_train, Y_train, y_train, X_val, Y_val, y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.y_train = y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.y_val = y_val

class Params:

    def __init__(self, epochs, n_batch, n_hidden, lmd, n_s):

        self.epochs = epochs
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.lmd = lmd
        self.n_s = n_s

def plotter(Xaxis, train, val, title):
    """ Plotter function to plot training and validation loss.
    :param train: training loss, this should be a list.
    :param val: validaition loss, this should be a list."""
    #plt.style.use('seaborn')
    plt.xlabel('Update step')
    plt.ylabel(str(title))
    #plt.ylim(ymin=0, ymax=np.max(train)*1.2)
    #plt.xlim(xmin=0, xmax=1050)
    plt.plot(Xaxis, train, color = 'green', label = "training " + str(title))
    plt.plot(Xaxis, val, color = 'red', label = "validation " + str(title))
    plt.legend()
    plt.savefig('result/' + str(title))
    plt.close()


def BatchCreator(j, n_batches):
    #Create mini_batch
    j_start = (j-1)*n_batches + 1
    j_end = j*n_batches + 1
    ind = np.arange(start= j_start, stop=j_end, step=1)
    return ind


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

def ReLU(x):
    """ Standard definition of the ReLU function """
    return np.maximum(x, 0)

def ComputeGradsNum(X, Y, W, b, lamda, h):
    """ OBS. This function is given by the instruction. It is used
    to evaluate the numerical gradient values."""
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c, _ = self.ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2, _ = self.ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] += h
            c2, _ = self.ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i,j] = (c2-c) / h

    return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1, _ = self.ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2, _ = self.ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] -= h
            c1, _ = self.ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i,j] += h
            c2, _ = self.ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i,j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]



"""
plt.style.use('seaborn')
plt.plot(tot_loss, color ='red', label = "Training Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


lst = []
for i in range(3000):
    lst.append(self.CyclicLearning(Npts, n_batches, 1e-1, 1e-5, i + 1))

print(lst)
plt.ylabel('eta')
plt.xlabel('Update step')
plt.plot(lst)
plt.show()

"""
