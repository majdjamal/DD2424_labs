
import numpy as np

X = np.arange(6*6) + 1
X = np.reshape(X, (6,6))

X_refined = np.zeros((X.shape))

X_refined[:, 0] = X[:, 0].reshape((2,-1)).flatten(order = 'F')
print(X)
print(X_refined)

x1 = X_refined[:, 0]
x2 = X_refined[:, 0].reshape((2, -2)).flatten(order = 'F')
print(x1)
print(x2)
print(np.all(x1 == x2))
"""
for i in range(Npts):
    X[:, i] = names2matrix[:, i].reshape((NUnique, NLongest)).flatten(order = 'F')

print(X)
print(np.all(X[:, i]))
x1 = X[:, 0]
x2 = X[:, 0].reshape((NUnique, NLongest)).flatten(order = 'F')
print(np.all(x1 == x2))
"""
