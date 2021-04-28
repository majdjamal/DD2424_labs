
import numpy as np
from utils.utils import softmax, ReLU


class ConvNet:

    def __init__(self):

        self.W = None

    def forward(self):
        pass

    def backward(self):
        pass

    def update(self):
        pass

    def convert2matrix(self, x, height, width):
        x = x.reshape((height, width), order='F')
        return x

    def MakeMFMatrix(self, X, F, NLongest):

        height_x, width_x = X.shape
        depth_f, height_f, width_f = F.shape

        ind_F = np.arange(height_f*width_f)

        FF = F.flatten().reshape(-1, 1)

        MF = np.zeros((
        (width_x - width_f + 1)*(height_x - height_f + 1),
         width_x*height_x))

        vec = MF[0]

        jump = width_x - width_f    # >> 3

        for i in range((width_x - width_f + 1)*(height_x - height_f + 1)):

            curr_F = np.zeros((width_x*height_x))
            curr_jump = 0

            for j in range(ind_F.size):

                if j % width_f == 0 and j != 0:
                    curr_jump += jump

                index = j + curr_jump + i

                curr_F[index] = FF[j]

            MF[i] = curr_F

        return MF

    def MakeMXMatrix(self, X, F):

        hx, wx = X.shape
        df, hf, wf  = F.shape

        ind_rows = np.arange(hf + 1)
        ind_cols = np.arange(wf + 1)

        def MX(x):

            MX = np.zeros(((wx - wf + 1)*(hx - hf + 1), hf*wf))

            for i in range(hx - hf + 1):
                for j in range(wx - wf + 1):

                    vals = x[
                    (ind_rows[0] + i):(ind_rows[-1] + i), #rows
                    (ind_cols[0] + j):(ind_cols[-1] + j)] #columns

                    row = j + i * (wx - wf + 1)
                    MX[i] = vals.flatten()

            return MX

        for k in range(df):

            if k == 0:
                MX_tot = MX(X)
            else:
                current_MX = MX(X)
                MX_tot = np.hstack((MX_tot, current_MX))

        return MX_tot


    def fit(self, data, params):

        X = data.X
        Y = data.Y
        y = data.y

        Ndim, Npts = X.shape
        Nout = Y.shape
        x_input = self.convert2matrix(X[:, 1], data.NUnique, data.NLongest)

        F1_TEST = np.random.random((1, 3, 3))

        F1 = np.random.random((params.NF1, data.NUnique, params.widthF1))
        F2 = np.random.random((params.NF2, params.NF1, params.widthF2))

        W = np.random.random((data.NClasses, Npts))

        MX1 = self.MakeMXMatrix(x_input, F1_TEST)
        MF1 = self.MakeMFMatrix(x_input, F1_TEST, 0)

        #np.savetxt('MX', MX1)
        #np.savetxt('X', x_input)



        print(MX1)
        print(1 in MX1)
        """
        s1 = MX1 @ F1_TEST.flatten().reshape(-1, 1)

        s2 = MF1 @ X[:, 1]

        print(s1, s2)

        """
        #toVec = self.convert2vec(X[:, 1000], data.NUnique, data.NLongest)
        #print(data.names[1000])

        # Complete the convolutional matricies
