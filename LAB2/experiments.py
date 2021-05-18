
import numpy as np
import matplotlib.pyplot as plt
from data.data import LoadBatch
from utils.utils import centering, softmax, ReLU, Data, Params, plotter
from model import MLP


#=-=-=-=-=-=-=-
# Experiments
#=-=-=-=-=-=-=-

#=-=-=-=-=-=-=-=-
# 1. Load Data
#=-=-=-=-=-=-=-=-




print('=-=- Loading data -=-= \n')
X_train, Y_train, y_train = (None, None, None)
X_val, Y_val, y_val = (None, None, None)
X_test, Y_test, y_test = LoadBatch('data/test_batch')

files = ['data/data_batch_1', 'data/data_batch_2', 'data/data_batch_3', 'data/data_batch_4', 'data/data_batch_5']
for i in range(len(files)):
    if i == 0:
        X_train, Y_train, y_train = LoadBatch(files[i])
    else:

        X_new_Batch, Y_new_Batch, y_new_Batch = LoadBatch(files[i])

        if i + 1 == len(files):
            _, Npts = X_new_Batch.shape
            Npts = round(Npts*0.9)
            X_train = np.concatenate((X_train, X_new_Batch[:, :Npts]), axis=1)
            Y_train = np.concatenate((Y_train, Y_new_Batch[:, :Npts]), axis=1)
            y_train = np.concatenate((y_train, y_new_Batch[:Npts]))

            X_val = X_new_Batch[:, Npts:]
            Y_val = Y_new_Batch[:, Npts:]
            y_val = y_new_Batch[Npts:]
        else:

            X_train = np.concatenate((X_train, X_new_Batch), axis=1)
            Y_train = np.concatenate((Y_train, Y_new_Batch), axis=1)
            y_train = np.concatenate((y_train, y_new_Batch))

#=-=-=-=-=-=-=-=-=-=-=-=-
# 2. Pre-processing.
#=-=-=-=-=-=-=-=-=-=-=-=-

X_train = centering(X_train)
X_val = centering(X_val)
X_test = centering(X_test)

# >>> X_train shape = (3072, 45000), X_val shape = (3072, 5000)
print('=-=- Data loading is completed! -=-= \n')

"""
_, Npts = X_train.shape
n_batch = 100
n_s = 2 * np.floor(Npts / n_batch)
#n_s = 500
cycles = 2
epochs = 2 * cycles * round(n_s / n_batch)

print('Parameters: n_batch = ', n_batch, 'n_s = ', n_s, '\n cycles = ', cycles, ' epochs = ', epochs, '\n')

lmax, lmin = np.log10(0.0014), np.log10(0.0040)
l = lmin + (lmax - lmin)*np.random.rand(1, 10);
lam = np.power(10, l)

print(lam)


def experiment(lmd):
    mlp = MLP()
    parms = Params(epochs = epochs, n_batch=n_batch,
            n_hidden = 50, lmd = lmd, n_s = n_s)
    data = Data(X_train, Y_train, y_train, X_val, Y_val, y_val)
    mlp.fit(data, parms)
    score = mlp.TestAccuracy(X_val, Y_val, y_val)
    return score

best_score = 0
best_lam = 0
scores = []


for i in lam[0]:
    print(i)
    score = experiment(i)
    scores.append([i, score])
    if score > best_score:
        best_score = score
        best_lam = i

np.save('scores_fine', scores)
print('Best Score: ', best_score)
print('Best lam: ', best_lam)

# >>> Best Score:  0.4407
# >>> Best lam:  0.00375
# Best Score:  0.5115000000000001
# Best lam:  0.0010690617064241779
#Best Score:  0.5362
#Best lam:  0.003897542212738527


"""
"""Parameter tuning
"""

_, Npts = X_train.shape
n_batch = 100
n_s = 2 * np.floor(Npts / n_batch)
#n_s = 500
cycles = 3
epochs = 2 * cycles * round(n_s / n_batch)
lmd = 0.003897542212738527

print('Parameters: n_batch = ', n_batch, 'n_s = ', n_s, '\n cycles = ', cycles, ' epochs = ', epochs, '\n')


mlp = MLP()
parms = Params(epochs = epochs, n_batch=n_batch,
        n_hidden = 50, lmd = lmd, n_s = n_s)

data = Data(X_train, Y_train, y_train, X_val, Y_val, y_val)

mlp.fit(data, parms)
score = mlp.TestAccuracy(X_val, Y_val, y_val)

memory = mlp.getMemory()
print('Validation Score: ', score)

score = mlp.TestAccuracy(X_test, Y_test, y_test)
print('Testing Score: ', score)


plotter(memory['itr'], memory['training'][0], memory['validation'][0], 'cost')
plotter(memory['itr'], memory['training'][1], memory['validation'][1], 'loss')
plotter(memory['itr'], memory['training'][2], memory['validation'][2], 'accuracy')


# Grad Analyze - 5pts: 2.007674884114928e-10 1.0580357916718401e-10 1.5111389718598365e-10 7.176290906267639e-11
