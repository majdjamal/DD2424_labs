
import numpy as np
from utils.utils import softmax, ReLU


class ConvNet:

    def __init__(self):

        self.W = None

        self.nlen = None
        self.dx = None

    def forward(self, MF, x_input):
        return MF @ x_input.T

    def backward(self):
        pass

    def update(self):
        pass

    def convert2matrix(self, x, height, width):
        x = x.reshape((height, width), order='F')
        return x

    def MakeMFMatrix(self, F, nlen):

    	nf,d,k = F.shape

    	zero = np.zeros((nf, self.nlen))

    	for filter in range(nf):
    		if filter == 0:
    			VF = F[filter].flatten(order = 'F')
    		else:
    			VF = np.vstack((VF, F[filter].flatten(order = 'F')))

    	MF = np.zeros(((self.nlen - k + 1)*nf, nlen*d))

    	step = 0
    	Nelements = VF[0].size

    	for i in range((self.nlen - k + 1)):

    		for j in range(nf):

    			ind = j + i*nf
    			MF[ind, step:Nelements + step] = VF[j]
    			#print(VF[j])

    		step += d

    	return MF

    def MakeMXMatrix(self, x_input, d, k, nf):

    	x_input = x_input.reshape((self.dx, self.nlen))

    	I_nf = np.identity(nf)

    	MX = np.zeros((
    	(self.nlen-k+1)*nf,
    	(k*nf*d)
    	))


    	for i in range(self.nlen-k+1):

    		vec = x_input[:, i:i+k].T.flatten()

    		vec = np.kron(I_nf, vec)

    		if i == 0:
    			MX = vec
    			#print(vec.shape)
    		else:
    			MX = np.vstack((MX, vec))

    	#print(MX.shape)

    	return MX

    def fit(self, data, params):

        X = data.X
        Y = data.Y
        y = data.y

        Ndim, Npts = X.shape
        self.dx, self.nlen = data.NUnique, data.NLongest

        Nout = Y.shape


        F1 = np.random.random((params.NF1, data.NUnique, params.widthF1))
        F2 = np.random.random((params.NF2, params.NF1, params.widthF2))
        W = np.random.random((data.NClasses, Npts))

        F = np.random.random((1, 28, 6))
        x_input = self.convert2matrix(X[:, 0], data.NUnique, data.NLongest)


        for filter in range(params.NF1):
        	if filter == 0:
        		F_flattened = F1[filter].flatten(order = 'F')
        	else:
        		F_flattened = np.hstack((F_flattened, F1[filter].flatten(order = 'F')))

        MF = self.MakeMFMatrix(F1, self.nlen)
        print(F1.shape)
        MX = self.MakeMXMatrix(x_input.flatten(), data.NUnique, params.widthF1, params.NF1)

        epochs = 10
        for epoch in range(epochs):
            for i in range(400):
                if i == 0:
                    x_input = self.convert2matrix(X[:, i], data.NUnique, data.NLongest).flatten(order='F')
                else:
                    x_input = np.vstack((x_input, self.convert2matrix(X[:, i], data.NUnique, data.NLongest).flatten(order='F')))

            S = self.forward(MF, x_input)
            print(S.shape)


        #s1 = MX @ F_flattened
        #s2 = MF @ x_input.T
        #print(s2.shape)
        #print(np.all(s1 == s2))

        #toVec = self.convert2vec(X[:, 1000], data.NUnique, data.NLongest)
        #print(data.names[1000])

        # Complete the convolutional matricies
