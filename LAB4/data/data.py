
__author__ = 'Majd Jamal'

import numpy as np
from collections import Counter

book_data = open('goblet_book.txt', 'r').read()
book_data = book_data.lower()
book_data = book_data.replace('\t', 't')
book_data = book_data.replace('\n', '')
book_data = book_data.replace('. . .', '')
unique = Counter(book_data)
chars = unique.items()

# Remove least occuring characters
for char, count in chars:

    if count < 200:
        book_data = book_data.replace(char, '')

# Remove punctuations and excesive spaces
for i in range(1, 5):

    book_data = book_data.replace(' '*i, ' ')
    book_data.replace('.'*i, '.')

unique = Counter(book_data)
chars = list(unique.keys())
NUnique = len(chars)

char_to_ind = {}
ind_to_char = {}

X = np.zeros((NUnique, len(book_data)))

for i in range(len(chars)):
    vec = np.zeros((NUnique, 1))
    char = chars[i]

    vec[i] = 1

    char_to_ind[char] = vec
    ind_to_char[i] = char

for i in range(len(book_data)):
    char = book_data[i]
    vec = char_to_ind[char]

    X[:, i] = vec[:,0]

np.save('processed/book_data.npy', book_data)
np.save('processed/X.npy', X)
np.save('processed/ind_to_char.npy', ind_to_char)
np.save('processed/char_to_ind.npy', char_to_ind)
np.save('processed/NUnique.npy', NUnique)
