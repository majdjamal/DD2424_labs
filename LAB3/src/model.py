
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

        self.counts = None

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

        x_input = x_input.reshape((dx, nlen), order='F')

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

    def forward(self, x_input, MF1, MF2, W):
        # =-=-=- All Correct -=-=-= 100 %

        S1 = ReLU(MF1 @ x_input)
        S2 = ReLU(MF2 @ S1)
        S = W@S2
        P = softmax(S)
        return S1, S2, S, P

    def backward(self, S1, S2, S, P, W, XBatch, YBatch):

        _, Npts = P.shape

        G = - (YBatch - P)

        #dW = G @ S2.T * (1/Npts)
        #dW1 = dW

        for j in range(Npts):
            g = G[:, j].reshape(-1, 1)
            s2 = S2[:, j].reshape(-1, 1)
            y = YBatch[:, j].argmax()

            py = (1/self.counts[y]) * (1/18)

            if j == 0:
                dW = g@s2.T * py
            else:
                dW += g@s2.T * py

        # =-=-=- ^ Correct -=-=-= 100 %

        G = W.T @ G
        G = G * np.where(S2 > 0, 1, 0)

        nf,d,k = self.F2.shape

        dF2 = 0
        for j in range(Npts):
            g = G[:, j]
            x = S1[:, j]
            mx = self.MakeMXMatrix(x, d, k, nf, 10, self.nlen - 5 + 1)
            v = g.T @ mx
            y = YBatch[:, j].argmax()

            py = (1/self.counts[y]) * (1/18)

            dF2 += v * py

        G = self.MF2.T @ G
        G = G * np.where(S1 > 0, 1, 0)


        nf,d,k = self.F1.shape

        dF1 = 0
        for j in range(Npts):
            g = G[:, j]
            x = XBatch[:, j]
            mx = self.MakeMXMatrix(x, d, k, nf, self.dx, self.nlen)
            v = g.T @ mx
            y = YBatch[:, j].argmax()

            py = (1/self.counts[y]) * (1/18)

            dF1 += v * py

        return dW, dF2, dF1

    def update(self, dW, dF2, dF1, eta):

        self.W -= dW.reshape(self.W.shape) * eta
        self.F2 -= dF2.reshape(self.F2.shape) * eta
        self.F1 -= dF1.reshape(self.F1.shape) * eta

    def ComputeCost(self, X, Y, MF1, MF2, W):

        _,_,_, P  = self.forward(X, MF1, MF2, W)
        _, Npts = P.shape

        #loss = np.mean(-np.log(np.einsum('ij,ji->i', Y.T, P)))
        #loss = np.mean(-np.log(np.diag(Y.T @ P)))

        loss = 0

        for j in range(Npts):

            y = Y[:, j]
            p = P[:, j]
            ind = y.argmax()
            py = (1/self.counts[ind]) * (1/18)
            loss -= np.log(y.T @ p) * py

        return loss

    def ComputeAccuracy(self, P, y):
        out = np.argmax(P, axis=0).reshape(-1,1)
        np.savetxt('out.txt', out)
        return 1 - np.mean(np.where(y==out, 0, 1))


    def ComputeGradsNumSlow(self, X, Y, W, F2, F1, h):

        MF2 = self.MakeMFMatrix(F2, 15)
        MF1 = self.MakeMFMatrix(F1, 19)

        grad_W1 = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W1_try = np.array(W)
                W1_try[i,j] -= h
                c1 = self.ComputeCost(X, Y, MF1, MF2, W1_try)

                W1_try = np.array(W)
                W1_try[i,j] += h
                c2 = self.ComputeCost(X, Y, MF1, MF2, W1_try)

                #print((c2 - c1) / (2 * h))

                grad_W1[i,j] = (c2 - c1) / (2 * h)

        grad_F2 = np.zeros(F2.shape)
        for k in range(F2.shape[0]):
            for i in range(F2.shape[1]):
                for j in range(F2.shape[2]):

                    F2_try = np.array(F2)
                    F2_try[k, i, j] -= h
                    MF2_try = self.MakeMFMatrix(F2_try, 15)
                    c1 = self.ComputeCost(X, Y, MF1, MF2_try, W)

                    F2_try = np.array(F2)
                    F2_try[k, i, j] += h
                    MF2_try = self.MakeMFMatrix(F2_try, 15)
                    c2 = self.ComputeCost(X, Y, MF1, MF2_try, W)


                    grad_F2[k, i, j] = (c2 - c1) / (2 * h)

        grad_F1 = np.zeros(F1.shape)
        for k in range(F1.shape[0]):
            for i in range(F1.shape[1]):
                for j in range(F1.shape[2]):

                    F1_try = np.array(F1)
                    F1_try[k, i, j] -= h
                    MF1_try = self.MakeMFMatrix(F1_try, 19)
                    c1 = self.ComputeCost(X, Y, MF1_try, MF2, W)

                    F1_try = np.array(F1)
                    F1_try[k, i, j] += h
                    MF1_try = self.MakeMFMatrix(F1_try, 19)
                    c2 = self.ComputeCost(X, Y, MF1_try, MF2, W)

                    #print((c2 - c1) / (2 * h))
                    grad_F1[k, i, j] = (c2 - c1) / (2 * h)

        return grad_W1, grad_F2, grad_F1


    def TestMFandMX(self):

        X_test = np.arange(1*6*4) + 1
        X_test = X_test.reshape(1,6,4)
        #print(X_test)
        X_input = X_test.flatten(order = 'F')
        #print(X_input)

        F_test = np.arange(4*6*3) + 1
        F_test = F_test.reshape(4,6,3)
        #print(F_test)

        nf,d,k = F_test.shape

        MF_test = self.MakeMFMatrix(F_test, 4)
        #print(MF_test)

        MX_test = self.MakeMXMatrix(X_input, d, k, nf, 6, 4)


        s1 = MF_test @X_input
        s2 = MX_test @ vecF(F_test)
        print(np.all(s1 == s2)) # >>> True

    def AnalyzeGradients(self, X, Y):

        XBatch = X[:, :100]
        YBatch = Y[:, :100]

        S1, S2, S, P = self.forward(XBatch, self.MF1, self.MF2, self.W)

        grad_an, grad_an_F2, grad_an_F1 = self.backward(S1, S2, S, P, self.W, XBatch, YBatch)
        grad_num, grad_num_F2, grad_num_F1 = self.ComputeGradsNumSlow(XBatch, YBatch, self.W, self.F2, self.F1, 1e-5)

        grad_an_F1 = grad_an_F1.reshape(self.F1.shape)
        grad_an_F2 = grad_an_F2.reshape(self.F2.shape)

        #print(grad_an_F1)
        diff = np.sum(np.abs(np.subtract(grad_an, grad_num)))
        diff /= np.sum(np.add(np.abs(grad_an), np.abs(grad_num)))
        print('W: ', diff)


        diff = np.sum(np.abs(np.subtract(grad_an_F2, grad_num_F2)))
        diff /= np.sum(np.add(np.abs(grad_an_F2), np.abs(grad_num_F2)))
        print('F2: ',diff)

        diff = np.sum(np.abs(np.subtract(grad_an_F1, grad_num_F1)))
        diff /= np.sum(np.add(np.abs(grad_an_F1), np.abs(grad_num_F1)))
        print('F1: ',diff)


    def fit(self, data, p):

        ##
        ##  Data
        ##
        X = data.X[:,:5000]
        Y = data.Y[:,:5000]
        y = data.y[:5000] - 1

        X_train = data.X_train
        Y_train = data.Y_train
        y_train = data.y_train - 1

        X_val = data.X_val
        Y_val = data.Y_val
        y_val = data.y_val - 1


        ##
        ##  Parameters
        ##

        #General
        Ndim, Npts = X_train.shape
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
        0, 1/np.sqrt(d),
        size = (p.n1, d, p.k1))

        self.F2 = np.random.normal(
        0, 1/np.sqrt(p.n1),
        size = (p.n2, p.n1, p.k2
        ))
        self.W = np.random.normal(
        0, 1/np.sqrt(Nout),
        size = (Nout, (p.n2 * nlen2)
        ))

        #self.F2 = np.random.randn((p.n2, p.n1, p.k2), p.n2*p.n1*p.k2*np.sqrt(2/(p.n2*p.n1*p.k2 * nlen2+p.n1*d*p.k1)))
        #self.F1 = np.random.randn((p.n1, d, p.k1), Nout*p.n2 * nlen2*np.sqrt(2/(Nout*p.n2 * nlen2+p.n1*d*p.k1)))
        #self.W = np.random.randn((Nout, (p.n2 * nlen2),1*np.sqrt(2/(Nout*p.n2 * nlen2))))
        #print(self.F2.shape)
        #print(self.F1.shape)
        #print(self.W.shape)

        self.MF1 = self.MakeMFMatrix(self.F1, nlen)
        self.MF2 = self.MakeMFMatrix(self.F2, nlen1)
        _, self.counts = np.unique(y, return_counts = True)
        #print(unique, counts)
        #=-=-=-=-=- Correct 100% -=-=-=-=-=
        # nlen1 = 15
        #print(nlen)

        #self.CheckMImplementations(self.MF1, self.F1, X[:, 0])


        #self.TestMFandMX()
        self.AnalyzeGradients(X, Y)

        """ Training
        print('=-=- Starting Training -=-=')
        for i in range(epochs):
            for j in range(round(Npts/n_batches)):

                ind = BatchCreator(j, n_batches).astype(int)
                XBatch = X_train[:, ind]
                YBatch = Y_train[:, ind]
                self.MF1 = self.MakeMFMatrix(self.F1, nlen)
                self.MF2 = self.MakeMFMatrix(self.F2, nlen1)

                S1, S2, S, P = self.forward(XBatch, self.MF1, self.MF2, self.W)

                gradients= self.backward(S1, S2, S, P, self.W, XBatch, YBatch)

                self.update(*gradients, p.eta)

                    #X, Y, MF1, MF2, W):
            loss = self.ComputeCost(X_train, Y_train, self.MF1, self.MF2, self.W)
            print('loss: ', loss)
            print('Epoch: ', i)
            print('\n')


        print('=-=- Training Completed -=-=')

        S1, S2, S, P = self.forward(X_val, self.MF1, self.MF2, self.W)
        acc = self.ComputeAccuracy(P, y_val)
        print('Accuracy: ', acc)
        #"""

            #print(i)dW, dF2, dF1
#    def backward(self, S1, S2, S, P, W, XBatch, YBatch):

        """
        _,_,_, P = self.forward(MF1, MF2, W, x_input)

        out = np.argmax(P, axis=0).reshape(-1,1)
        print(out)
        print(1 - np.mean(np.where(y==out, 0, 1)))
        #        out = np.argmax(P, axis=0).reshape(-1,1)
        """
