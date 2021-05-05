
import numpy as np

class Data:
    """Object to store data
    """
    def __init__(self,
    X_train, Y_train, y_train,
    X_val, Y_val, y_val,
    X, Y, y, names,
    NUnique, NLongest, NClasses):

        self.X_train = X_train
        self.Y_train = Y_train
        self.y_train = y_train

        self.X_val = X_val
        self.Y_val = Y_val
        self.y_val = y_val

        self.X = X
        self.Y = Y
        self.y = y
        self.names = names

        self.NUnique = NUnique
        self.NLongest = NLongest
        self.NClasses = NClasses

def getData():
    """Load and seperate data into
    a training and validation set, and
    return an object of the data.
    """
    X = np.load('data/final/X.npy')
    Y = np.load('data/final/Y.npy')
    y = np.load('data/final/ys.npy')

    names = np.load('data/final/names.npy')

    dims = np.load('data/final/dims.npy')

    all_indices = np.arange(X.shape[1])
    validation_indicies = np.loadtxt('data/dataset/Validation_Inds.txt').astype(int)
    training_indicies = np.delete(all_indices, validation_indicies)

    data = Data(
    X_train = X[:, training_indicies] ,
    Y_train = Y[:, training_indicies],
    y_train = y[training_indicies],
    X_val = X[:, validation_indicies],
    Y_val = Y[:, validation_indicies],
    y_val = y[validation_indicies],
    X = X,
    Y = Y,
    y = y,
    names = names,
    NUnique = dims[0],
    NLongest = dims[1],
    NClasses = dims[2],
)

    return data
