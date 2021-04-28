
import numpy as np
import time

start = time.time()

import numpy as np
import time
# 6 x 6
wid = 4
X = np.arange(wid*wid) + 1
X = X.reshape((wid,wid))

dx, nlen = X.shape

d = 4 #height
k = 2 #width
nf = 3 #number of filters
F = np.arange((nf*d*k)) + 1
F = F.reshape((nf,d,k))

def MakeMXMatrix(x_input, d, k, nf):

	x_input = x_input.reshape((dx, nlen))

	"""
	MX = np.zeros((
	(nlen-k+1)*nf,
	(k*nf*d)
	))
	"""

	I_nf = np.identity(nf)

	for i in range(nlen -1):
		step = k - 1
		vec = x_input[:, i:i+k].flatten(order='F')
		vec = np.kron(I_nf, vec)

		if i == 0:
			MX = vec
		else:
			MX = np.vstack((MX, vec))

	print(MX)


start = time.time()

MakeMXMatrix(X.flatten(), d,k,nf)
end = time.time()
print('\n running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms
