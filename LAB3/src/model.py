
import numpy as np
from utils.utils import softmax, ReLU, BatchCreator


class ConvNet:

    def __init__(self):

        self.W = None

        self.nlen = None
        self.dx = None

        self.F1 = None
        self.F2 = None

        self.MF1 = None
        self.MF2 = None

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

    def MakeMXMatrix(self, x_input, d, k, nf, dx, nlen):

        x_input = x_input.reshape((dx, nlen))

        I_nf = np.identity(nf)

        MX = np.zeros((
        (nlen-k+1)*nf,
        (k*nf*d)
        ))

        for i in range(nlen-k+1):

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
        P = softmax(S)
        return S1, S2, S, P

    def backward(self, S1, S2, S, P, W, XBatch, YBatch):

        _, Npts = P.shape

        G = - (YBatch - P)

        dW = G @ S2.T * (1/Npts)

        G = W.T @ G
        G = G * np.where(S2 > 0, 1, 0)


        #MakeMXMatrix(self, x_input, d, k, nf):
        nf,d,k = self.F2.shape

        dF2 = 0
        for j in range(Npts):
            g = G[:, j]
            x = S1[:, j]
            mx = self.MakeMXMatrix(x, d, k, nf, 4, self.nlen - 5 + 1)
            v = g.T @ mx
            dF2 += v
        dF2 *= (1/Npts)

        G = self.MF2.T @ G
        G = G * np.where(S1 > 0, 1, 0)


        nf,d,k = self.F1.shape

        dF1 = 0
        for j in range(Npts):
            g = G[:, j]
            x = XBatch[j]
            mx = self.MakeMXMatrix(x, d, k, nf, self.dx, self.nlen)

            v = g.T @ mx
            dF1 += v

        dF1 *= (1/Npts)

        return dW, dF2, dF1

    def vecX(self, x):
        """
            OBS: To classify x data points, they need to pass through this
            vector-converter.
        """
        return x.reshape((self.dx, self.nlen)).flatten(order = 'F')

    def vecF(self, F):
        """
            OBS: To classify with filter F, it need to be flattened
            through this converter.
        """
        nf,_,_ = F.shape
        for filter in range(nf):

        	if filter == 0:
        		F_flattened = F[filter].flatten(order = 'F')
        	else:
        		F_flattened = np.hstack((F_flattened, F[filter].flatten(order = 'F')))

        return F_flattened


    def fit(self, data, params):

        ##
        ##  Data
        ##
        X = data.X
        Y = data.Y
        y = data.y

        ##
        ##  Parameters
        ##

        #General
        Ndim, Npts = X.shape
        self.dx, self.nlen = data.NUnique, data.NLongest
        Nout, _ = Y.shape
        epochs = params.epochs
        n_batches = params.n_batches


        #F1
        d = data.NUnique
        k = params.widthF1
        nf = params.NF1 #number of filters

        #F2
        d2 = params.NF1
        k2 = params.widthF2
        nf2 = params.NF2

        ##
        ##  Filters & Weights
        ##
        F1 = np.random.random((nf,d,k))
        F2 = np.random.random((nf2, d2, k2))
        W = np.random.random((Nout, 44))

        self.F1, self.F2 = F1, F2

        for epoch in range(epochs):

            for j in range(round(Npts/n_batches)):

                ind = BatchCreator(j, n_batches).astype(int)
                mini_batch = X[:, ind]

                for i in range(mini_batch.shape[1]):
                    if i == 0:
                        x_input = self.vecX(mini_batch[:, i])
                    else:
                        x_input = np.vstack((x_input, self.vecX(mini_batch[:, i])))

                XBatch = x_input
                YBatch = Y[:, ind]
                MF1 = self.MakeMFMatrix(F1, self.nlen)
                MF2 = self.MakeMFMatrix(F2, (self.nlen - params.widthF1 + 1))

                self.MF1, self.MF2 = MF1, MF2

                S1, S2, S, P = self.forward(MF1, MF2, W, XBatch)

                dW, dF2, dF1 = self.backward(S1, S2, S, P, W, XBatch, YBatch)

                F1 += dF1.reshape(F1.shape) *params.eta
                F2 += dF2.reshape(F2.shape) *params.eta
                W += dW.reshape(W.shape) *params.eta

            print(epoch)

        for i in range(Npts):
            if i == 0:
                x_input = self.vecX(X[:, i])
            else:
                x_input = np.vstack((x_input, self.vecX(X[:, i])))

        _,_,_, P = self.forward(MF1, MF2, W, x_input)

        out = np.argmax(P, axis=0).reshape(-1,1)
        print(1 - np.mean(np.where(y==out, 0, 1)))
        #        out = np.argmax(P, axis=0).reshape(-1,1)
