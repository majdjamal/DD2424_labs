
__author__ = 'Majd Jamal'

import numpy as np

class Data:

    def __init__(self, book_data, X, ind_to_char, char_to_ind, NUnique):
        self.book_data = book_data
        self.X = X
        self.ind_to_char = ind_to_char
        self.char_to_ind = char_to_ind
        self.NUnique = NUnique

def getData():

    book_data = np.load('data/processed/book_data.npy')
    X = np.load('data/processed/X.npy')
    ind_to_char = np.load('data/processed/ind_to_char.npy', allow_pickle=True)
    char_to_ind = np.load('data/processed/char_to_ind.npy', allow_pickle=True)
    NUnique = np.load('data/processed/NUnique.npy')

    data = Data(book_data, X, ind_to_char, char_to_ind, NUnique)

    return data

data = getData()
