
import numpy as np

def ComputeGradsNum(X, Y, W, b, lmbda, h):

	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((b.shape[0], 1))

	c, _ = ComputeCost(X, Y, W, b, lmbda)

	for j in range(len(b)):

		for i in range(len(b[j])):

			b_try = np.array(b)
			b_try[i][j] += h
			c2, _ = ComputeCost(X, Y, W, b_try, lmbda)
			grad_b[j][i] = (c2 - c) / h

	for j in range(len(W)):
		for i in range(len(W[j])):
			W_try = np.array(W)
			W_try[j][i] += h
			c2, _ = ComputeCost(X, Y, W_try, b, lmbda)

			grad_W[j][i] = (c2 - c) / h

	return [grad_W, grad_b]
