
import numpy as np
from utils.utils import softmax, ReLU, BatchCreator, vecX, vecF


class ConvNet:

    def __init__(self):

        self.nlen = None
        self.dx = None

        self.F1 = None
        self.F2 = None
        self.W = None

        self.MF1 = None
        self.MF2 = None

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

    def forward(self, x_input):

        S1 = ReLU(self.MF1 @ x_input)
        S2 = ReLU(self.MF2 @ S1)
        S = self.W@S2
        P = softmax(S)
        return S1, S2, S, P

    def backward(self, S1, S2, S, P, W, XBatch, YBatch, X_MX):

        _, Npts = P.shape
        #Npts = 1

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
            x = X_MX[:, j]
            mx = self.MakeMXMatrix(x, d, k, nf, self.dx, self.nlen)
            v = g.T @ mx
            dF1 += v

        dF1 *= (1/Npts)

        return dW, dF2, dF1


    def fit(self, data, p):

        ##
        ##  Data
        ##
        X = data.X
        Y = data.Y
        y = data.y - 1

        ##
        ##  Parameters
        ##

        #General
        Ndim, Npts = X.shape
        self.dx, self.nlen = data.NUnique, data.NLongest
        Nout, _ = Y.shape
        epochs = p.epochs
        n_batches = p.n_batches

        #height of F1
        d = data.NUnique

        nlen = data.NLongest
        nlen1 = nlen - p.k1 + 1
        nlen2 = nlen1 - p.k2 + 1

        ##
        ##  Filters & Weights
        ##
        ##  F1, F2, W, MF1, and MF2
        ##
        self.F1 = np.random.normal(
        0, 0.1,
        size = (p.n1, d, p.k1))

        self.F2 = np.random.normal(
        0, 0.9,
        size = (p.n2, p.n1, p.k2
        ))
        self.W = np.random.normal(
        0, 0.5,
        size = (Nout, (p.n2 * nlen2)
        ))

        self.MF1 = self.MakeMFMatrix(self.F1, nlen)
        self.MF2 = self.MakeMFMatrix(self.F2, nlen1)

        #=-=-=-=-=- Correct 100% -=-=-=-=-=

        """
        for i in range(Npts):

            x = vecX(X[:, i], d, nlen)

            if i == 0:
                X_vectorized = x
            else:
                X_vectorized = np.vstack((X_vectorized, x))
            print(i)

        np.save('X_vectorized.npy', X_vectorized)
        """
        X_vectorized = np.load('X_vectorized.npy').T
        #print(X_vectorized.shape)

        for i in range(epochs):
            for j in range(round(Npts/n_batches)):

                ind = BatchCreator(j, n_batches).astype(int)
                XBatch = X_vectorized[:, ind]
                YBatch = Y[:, ind]
                X_MX = X[:, ind]

                S1, S2, S, P = self.forward(XBatch)

                dW, dF2, dF1 = self.backward(S1, S2, S, P, self.W, XBatch, YBatch, X_MX)

                self.W -= dW.reshape(self.W.shape) * p.eta
                self.F2 -= dF2.reshape(self.F2.shape) * p.eta
                self.F1 -= dF1.reshape(self.F1.shape) * p.eta

                self.MF1 = self.MakeMFMatrix(self.F1, nlen)
                self.MF2 = self.MakeMFMatrix(self.F2, nlen1)
                
            S1, S2, S, P = self.forward(X_vectorized)
            print(np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P))))
        S1, S2, S, P = self.forward(X_vectorized)

        out = np.argmax(P, axis=0).reshape(-1,1)
        print(np.argmax(P, axis=0)[:100])

        print(y[:100])
        print(1 - np.mean(np.where(y == out, 0, 1)))


            #print(i)dW, dF2, dF1
#    def backward(self, S1, S2, S, P, W, XBatch, YBatch):
        """ Checker
        S1 = MF @ self.vecX(X[:, 0])
        S2 = MX @ self.vecF(F1)

        print(np.all(S1==S2))
        """

        """
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
                MF2 = self.MakeMFMatrix(F2, (self.nlen - p.k1 + 1))

                self.MF1, self.MF2 = MF1, MF2

                S1, S2, S, P = self.forward(MF1, MF2, W, XBatch)

                dW, dF2, dF1 = self.backward(S1, S2, S, P, W, XBatch, YBatch)

                F1 += dF1.reshape(F1.shape) *p.eta
                F2 += dF2.reshape(F2.shape) *p.eta
                W += dW.reshape(W.shape) *p.eta

            print(epoch)

        _,_,_, P = self.forward(MF1, MF2, W, x_input)

        out = np.argmax(P, axis=0).reshape(-1,1)
        print(out)
        print(1 - np.mean(np.where(y==out, 0, 1)))
        #        out = np.argmax(P, axis=0).reshape(-1,1)
        """
