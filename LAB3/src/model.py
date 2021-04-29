
import numpy as np
from utils.utils import softmax, ReLU, BatchCreator


class ConvNet:

    def __init__(self):

        self.W = None

        self.nlen = None
        self.dx = None

    def convert2matrix(self, x, height, width):
        x = x.reshape((height, width), order='F')
        return x

    def MakeMFMatrix(self, F, nlen):

    	nf,d,k = F.shape

    	zero = np.zeros((nf, nlen))

    	for filter in range(nf):
    		if filter == 0:
    			VF = F[filter].flatten(order = 'F')
    		else:
    			VF = np.vstack((VF, F[filter].flatten(order = 'F')))

    	MF = np.zeros(((nlen - k + 1)*nf, nlen*d))

    	step = 0
    	Nelements = VF[0].size

    	for i in range((nlen - k + 1)):

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

    def forward(self, MF1, MF2, W, x_input):
        S1 = ReLU(MF1 @ x_input.T)
        S2 = ReLU(MF2 @ S1)

        S = W@S2
        return S1, S2, S

    def backward(self,S1, S2, P, W, XBatch, YBatch, data, params):

        _, Npts = P.shape

        G = - (YBatch - P)

        dW = G @ S2.T * (1/Npts)

        G = W.T @ G
        G = G * np.where(S2 > 0, 1, 0)

        for j in range(Npts):
            g = G[:, j]
            x = S1[:, j]
            mf = self.MakeMXMatrix(x.flatten(), data.NUnique, params.widthF1, params.NF1)

        pass

    def update(self):
        pass

    def fit(self, data, params):

        X = data.X
        Y = data.Y
        y = data.y

        Ndim, Npts = X.shape
        self.dx, self.nlen = data.NUnique, data.NLongest

        Nout = Y.shape


        F1 = np.random.random((params.NF1, data.NUnique, params.widthF1))
        F2 = np.random.random((params.NF2, params.NF1, params.widthF2))
        W = np.random.random((data.NClasses, 44))

        F = np.random.random((1, 28, 6))
        x_input = self.convert2matrix(X[:, 0], data.NUnique, data.NLongest)


        for filter in range(params.NF1):
        	if filter == 0:
        		F_flattened = F1[filter].flatten(order = 'F')
        	else:
        		F_flattened = np.hstack((F_flattened, F1[filter].flatten(order = 'F')))

        MF1 = self.MakeMFMatrix(F1, self.nlen)

        MF2 = self.MakeMFMatrix(F2, (self.nlen - params.widthF1 + 1))

        MX = self.MakeMXMatrix(x_input.flatten(), data.NUnique, params.widthF1, params.NF1)

        for epoch in range(params.epochs):

            for j in range(round(Npts/params.n_batches)):



                ind = BatchCreator(j, params.n_batches).astype(int)

                #print(mini_batch[-1])
                mini_batch = X[:, ind]

                for i in range(mini_batch.shape[1]):
                    if i == 0:
                        x_input = self.convert2matrix(mini_batch[:, i], data.NUnique, data.NLongest).flatten(order='F')
                    else:
                        x_input = np.vstack((x_input, self.convert2matrix(mini_batch[:, i], data.NUnique, data.NLongest).flatten(order='F')))

                XBatch = x_input
                YBatch = Y[:, ind]
                S1, S2, S = self.forward(MF1, MF2, W, XBatch)
                self.backward(S1, S2, S, W, XBatch, YBatch, data, params)

        print(MF1.shape)



        """
        epochs = 10
        for epoch in range(epochs):
            for i in range(400):
                if i == 0:
                    x_input = self.convert2matrix(X[:, i], data.NUnique, data.NLongest).flatten(order='F')
                else:
                    x_input = np.vstack((x_input, self.convert2matrix(X[:, i], data.NUnique, data.NLongest).flatten(order='F')))

            S = self.forward(MF, x_input)
            print(S.shape)

        """
        #s1 = MX @ F_flattened
        #s2 = MF @ x_input.T
        #print(s2.shape)
        #print(np.all(s1 == s2))

        #toVec = self.convert2vec(X[:, 1000], data.NUnique, data.NLongest)
        #print(data.names[1000])

        # Complete the convolutional matricies
