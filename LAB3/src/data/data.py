
import time
import numpy as np
import re

file = 'dataset/ascii_names.txt'
start = time.time()


##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
##  Load data and seperate it into
##  names (string) and labels (int)
##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
fid = np.loadtxt(file,delimiter='\t', dtype=str)
Npts = fid.size

names = np.zeros(Npts).astype('str')
ys = np.zeros(Npts)

for i in range(Npts):
    name, label = fid[i].split(' ')

    names[i] = name.lower()
    ys[i] = label


##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
##  Find:
##      number of unique unique characters
##      length of the longest word
##      number of classes
##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
doc = open(file, 'r').read().lower().replace("\n", "").replace(" ", "") #Open document

longest_word = max(names, key = len)    # Get longest string
characters = re.sub(r'\d+', '', doc) # Remove digits
unique_characters = ''.join(set(characters))    # Extract unique characters

NUnique = len(unique_characters) #Number of unique characters
                                 #in the document
NLongest = len(longest_word)     #lenght of longest name
NClasses = np.max(ys).astype(int)  #number of classes


##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
##  Create a lexicon of char2ind, i.e,
##  character to index-notation
##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
char2ind = {}

for i in range(NUnique):
    char2ind[unique_characters[i]] = i


##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
##  Convert names to a matrix representation
##  i.e, the data point matrix X with shape = (Ndim, Npts)
##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
names2matrix = np.zeros((NUnique * NLongest, Npts)) #also known as, X

for i in range(Npts):
    name = names[i]
    name2vec = np.zeros((NUnique, NLongest))

    for j in range(len(name)):

        curr_char = name[j]
        ind = char2ind[curr_char]

        name2vec[ind][j] = 1

    names2matrix[:, i] = name2vec.flatten(order = 'F')


##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
##  Creating one hot vector matrix
##  with shape = (Nclasses, Npts)
##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Y = np.zeros((ys.size, ys.max().astype(int)+1))
Y[np.arange(ys.size), np.reshape(ys, (1,-1)).astype(int)] = 1
Y = Y.T
Y = np.delete(Y, 0, 0)



##=-=-=-=-=-=-=-=-=-=
##  Save files
##=-=-=-=-=-=-=-=-=-=
np.save('final/names.npy', names)
np.save('final/X.npy', names2matrix)
np.save('final/Y.npy', Y)
np.save('final/ys.npy', ys)
np.save('final/dims.npy', np.array([NUnique, NLongest, NClasses]))

end = time.time()
print('running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms
