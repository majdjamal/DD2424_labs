

class Params:

	def __init__(self, NF1, NF2, widthF1, widthF2, eta, roh):

		self.NF1 = NF1
		self.NF2 = NF2

		self.widthF1 = widthF1
		self.widthF2 = widthF2

		self.eta = eta
		self.roh = roh

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x):
    """ Standard definition of the ReLU function """
    return np.maximum(x, 0)
