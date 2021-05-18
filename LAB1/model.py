
import numpy as np
import matplotlib.pyplot as plt
from functions import softmax, LoadBatch, ComputeGradsNumSlow, montage, save_as_mat
from data.data import LoadBatch
from utils import centering, softmax

class GDparams:
	""" This class is used to train the network with parameters as an object."""
	def __init__(self, n_batches, eta, n_epochs, lmd):
		"""
		:param n_batches: Number of images in each batch.
		:param eta: Learning rate
		:param n_epochs: Number of training iterations
		:lmd: Constant used in the regularization term.
		"""
		self.n_batches = n_batches
		self.eta = eta
		self.n_epochs = n_epochs
		self.lmd = lmd

class SLN:
	""" One Layer Network used for classification. """
	def __init__(self):

		self.W = None
		self.b = None

		self.cost_train = []	#training cost
		self.cost_val = []		#validation cost

		self.loss_train = []	#training loss
		self.loss_val = []		#validation loss

		self.X_val = None
		self.Y_val = None
		self.y_val = None

	def ComputeGradsNum(self, X, Y, W, b, lamda, h):
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

	def EvaluateClassifier(self, X, W, b):
		"""Computes the forwards pass, using Softmax.
		:param X: Data matrix, shape = (Ndim, Npts)
		:param W: Weight Matrix, shape = (Nout, Ndim)
		:param b: bias term, shape = (Nout, 1)
		:return y: values for the forward pass
		"""
		return softmax(W @ X + b)

	def ComputeCost(self, X, Y, W, b, lmd):
		"""Loss function. Computes a scalar value with
		L2-regularization.
		:param X: Data matrix, shape = (Ndim, Npts)
		:param Y: One hot matrix, shape =(Nout, Npts)
		:param W: Weight Matrix, shape = (Nout, Ndim)
		:param b: bias term, shape = (Nout, 1)
		:param lmd: Regularization value
		:return cost: scalar indicating loss
		"""
		Ndim, Npts = X.shape
		P = self.EvaluateClassifier(X, W, b)
		loss = np.mean(-np.log(np.diag(Y.T @ P)))
		reg = lmd * np.sum(np.square(W))
		cost = loss + reg
		#J = np.mean(-np.log(np.diag(Y.T @ P))) + lmd * np.sum(np.square(W))
		return cost, loss

	def ComputeAccuracy(self, X, y, W, b):
		"""Computes a score indicating the network accuracy.
		:param X: Data matrix, shape = (Ndim, Npts)
		:param y: One hot matrix, shape = (Npts, 1)
		:param W: Weight Matrix, shape = (Nout, Ndim)
		:param b: bias term, shape = (Nout, 1)
		:return: error rate. low is better.
		"""
		p = self.EvaluateClassifier(X, W, b)
		out = np.argmax(p, axis=0).reshape(-1,1)
		return np.mean(np.where(y==out, 0, 1))

	def ComputeGradients(self, X, Y, W, b, lmd):
		"""Computes analytic gradients, for weights and bias term.
		:param X: Data matrix, shape = (Ndim, Npts)
		:param Y: One hot matrix, shape =(Nout, Npts)
		:param W: Weight Matrix, shape = (Nout, Ndim)
		:param b: bias term, shape = (Nout, 1)
		:param lmd: Regularization value
		:return grad_W: Gradients for the Weights
		:return grad_b: Gradients for the bias term
		"""

		P = self.EvaluateClassifier(X, W, b)
		_, Npts = P.shape

		G = - (Y - P)

		grad_W = G @ X.T * (1/Npts) + 2 * lmd * W
		grad_b = G @ np.ones((Npts, 1)) * (1/Npts)

		return grad_W, grad_b

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

	def ComputeGradsNumSlow(self, X, Y, W, b, lamda, h):
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

	def AnalyzeGradients(self, X, Y):
		""" Generates numerical and analytical gradients and
			compares their abolsute difference.
		:param X: Data matrix, shape = (Ndim, Npts)
		:param Y: One hot matrix, shape =(Nout, Npts)
		return diff_W: Difference between numerical and analytical
						gradient for the weights.
		:return diff_W: Difference between numerical and analytical
						gradient for the bias term.
		"""
		Ndim, Npts = X.shape

		grad_num_W, grad_num_b = self.ComputeGradsNum(
			X[:,[0,1,2,3,4]], Y[:,[0,1,2,3,4]],
			self.W, self.b, 0, 1e-6)

		grad_an_W, grad_an_b = self.ComputeGradients(
			X[:,[0,1,2,3,4]], Y[:,[0,1,2,3,4]],
			self.W, self.b, 0)

		diff_W_small = self.NumericalDifference(grad_an_W, grad_num_W)
		diff_b_small = self.NumericalDifference(grad_an_b, grad_num_b)
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
		print('With lamda = 0 and 5 points, the differences in gradients was:\n' + ' W - ' + str(diff_W_small) + ' b - ' + str(diff_b_small))
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

		ind = np.random.choice(Npts, 100, replace=False)
		#ind = np.arange(50)

		grad_num_W, grad_num_b = self.ComputeGradsNum(
			X[:,ind], Y[:,ind],
			self.W, self.b, 0, 1e-6)

		grad_an_W, grad_an_b = self.ComputeGradients(
			X[:,ind], Y[:,ind],
			self.W, self.b, 0)

		diff_W = self.NumericalDifference(grad_an_W, grad_num_W)
		diff_b = self.NumericalDifference(grad_an_b, grad_num_b)
		print('\n')
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
		print('With lamda = 0 and 100 points, the differences in gradients was:\n' + ' W - ' + str(diff_W) + ' b - ' + str(diff_b))
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

		grad_num_W, grad_num_b = self.ComputeGradsNumSlow(
			X[:,ind], Y[:,ind],
			self.W, self.b, 0.3, 1e-6)

		grad_an_W, grad_an_b = self.ComputeGradients(
			X[:,ind], Y[:,ind],
			self.W, self.b, 0.3)

		diff_W = self.NumericalDifference(grad_an_W, grad_num_W)
		diff_b = self.NumericalDifference(grad_an_b, grad_num_b)
		print('\n')
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
		print('With lamda = 0.1 and 100 points, the differences in gradients was:\n' + ' W - ' + str(diff_W) + ' b - ' + str(diff_b))
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

	def getCost(self):
		""" Returns training and validation loss
		:return self.costtrain: training loss
		:return self.costval: validation loss
		"""
		return self.cost_train, self.cost_val

	def getLoss(self):
		""" Returns training and validation loss
		:return self.costtrain: training loss
		:return self.costval: validation loss
		"""
		return self.loss_train, self.loss_val


	def getParams(self):
		""" Returns the model parameters, i.e. weights and bias.
		:return self.W: Weights
		:return self.b: Bias
		"""
		return self.W, self.b

	def MiniBatchGD(self, X, Y, y, GDparams, W, b):
		""" Train the model with Mini Batches.
		:param X: Data matrix, shape = (Ndim, Npts)
		:param Y: One hot matrix, shape =(Nout, Npts)
		:param y: One hot matrix, shape = (Npts, 1)
		:param GDparams: Parameters used for the training, i.e.
			n_batches, eta, n_epochs, lambda.
		:param W: Weight Matrix, shape = (Nout, Ndim)
		:param b: bias term, shape = (Nout, 1)
		"""
		n_batches = GDparams.n_batches
		eta = GDparams.eta
		epochs = GDparams.n_epochs
		lmd = GDparams.lmd
		Ndim, Npts = X.shape

		for epoch in range(epochs):

			# Shuffle training data
			ind = np.arange(Npts)
			np.random.shuffle(ind)

			X, Y = X[:, ind], Y[:, ind]

			for j in range(round(Npts/n_batches)):

				#Create mini_batch
				j_start = (j-1)*n_batches + 1
				j_end = j*n_batches + 1
				ind = np.arange(start= j_start, stop=j_end, step=1)

				XBatch = X[:, ind]
				YBatch = Y[:, ind]

				#Compute gradients
				Wstar, bstar = self.ComputeGradients(
				XBatch, YBatch, self.W, self.b, lmd)

				#Update parameters
				self.W -= eta * Wstar
				self.b -= eta * bstar

			#Store loss values
			cost, loss = self.ComputeCost(X,Y, self.W, self.b, lmd)
			cost_validation, loss_validation = self.ComputeCost(self.X_val,self.Y_val, self.W, self.b, lmd)

			self.cost_train.append(cost)
			self.cost_val.append(cost_validation)

			self.loss_train.append(loss)
			self.loss_val.append(loss_validation)

	def fit(self, data, params):
		""" This function is called to start trining.
		:param data: Object containing training and validation data
		:param params: Object containing parameters used for training, i.e.
			n_batches, eta, n_epochs, lambda.
		"""
		X = data.X_train
		Y = data.Y_train
		y = data.y_train

		self.X_val = data.X_val
		self.Y_val = data.Y_val
		self.y_val = data.y_val

		Ndim, Npts = X.shape
		Nout, _ = Y.shape

		np.random.seed(400)
		self.W = np.random.normal(0, 0.01, size=(Nout, Ndim))
		self.b = np.random.normal(0, 0.01, size=(Nout, 1))

		#self.AnalyzeGradients(X, Y)	# Open to print differences
								# in numerical and analytic gradients.


		self.MiniBatchGD(X, Y, y, params, self.W, self.b)

	def TestAccuracy(self, X, Y, y):
		""" Computes accuracy of the model, i.e. error rate.
		:param X: Test Data matrix, shape = (Ndim, Npts)
		:param Y: Test One hot matrix, shape =(Nout, Npts)
		:param y: Test One hot matrix, shape = (Npts, 1)
		"""
		score = self.ComputeAccuracy(X, y, self.W, self.b)
		return 1 - score
