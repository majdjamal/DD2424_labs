
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

    def MakeMFMatrix(self, X, F):
        """ Generates the MF matrix for a data point
        :param X: data point, shape = (Nunique, NLongest)
        :param F: filter, shape = (Height, Width)
        :return MF: The MF matrix
        """

        height_x, width_x = X.shape
        df, height_f, width_f = F.shape

        FF = F.flatten().reshape(-1, 1)
        F_entries = height_f* width_f

        MF = np.zeros((
        (width_x - width_f + 1)*(height_x - height_f + 1),

        width_x*height_x))

        shift = 0

        for k in range(height_x - height_f + 1):



            for i in range(width_x - width_f + 1):

                jump = 0
                MF_row = np.zeros(MF.shape[1])

                for j in range(F_entries):

                    if j % width_f == 0 and j != 0:

                        jump += width_x - width_f

                    ind = j + jump + i + shift
                    #print(ind)

                    MF_row[ind] = FF[j]

                #print('\n')
                row = k * (width_x - width_f + 1) + i
                MF[row] = MF_row

            shift += width_x

                    #MF_row[ind] = FF[j]

        return MF


    def MakeMXMatrix(self, X, F):
        """ Generates the MX-matrix
        :param X: data point, shape = (Nunique, NLongest)
        :param F: filter, shape = (Height, Width)
        :return MX: The MX matrix
        """
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

                    row = j + (i)*(wx - wf + 1)
                    MX[row] = vals.flatten()

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


        F1 = np.random.random((params.NF1, data.NUnique, params.widthF1))
        F2 = np.random.random((params.NF2, params.NF1, params.widthF2))
        W = np.random.random((data.NClasses, Npts))

        F = np.random.random((1, 28, 6))
        x_input = self.convert2matrix(X[:, 0], data.NUnique, data.NLongest)

        MX = self.MakeMXMatrix(x_input, F)
        MF = self.MakeMFMatrix(x_input, F)

        print(MF)

        s1 = MX @ F.flatten()
        s2 = MF @ x_input.flatten()
        print(s1, s2)
        print(np.all(s1 == s2))


        #toVec = self.convert2vec(X[:, 1000], data.NUnique, data.NLongest)
        #print(data.names[1000])

        # Complete the convolutional matricies
