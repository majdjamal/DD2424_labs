
import numpy as np
import time

"""
# 6 x 6
wid = 28
hei_x = 19
X = np.arange(wid * hei_x) + 1
X = X.reshape((wid , hei_x))

dx, nlen = X.shape

d = 28 #height
k = 5 #width
nf = 4 #number of filters
F = np.arange((nf*d*k)) + 1
F = F.reshape((nf,d,k))
"""

d = 28 #height
k = 5 #width
nf = 4 #number of filters
F = np.arange((nf*d*k)) + 1
F = F.reshape((nf,d,k))


wid = 28
hei_x = 19
X = np.arange(wid * hei_x) + 1
X = X.reshape((wid, hei_x))

dx, nlen = X.shape

"""
Notes:
https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd
https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
"""

def MakeMFMatrix(F, nlen):

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

def MakeMXMatrix(x_input, d, k, nf):

	x_input = x_input.reshape((dx, nlen))

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

start = time.time()

MF = MakeMFMatrix(F, nlen)
MX = MakeMXMatrix(X.flatten(), d,k,nf)


#print(F)
for filter in range(nf):
	if filter == 0:
		F_flattened = F[filter].flatten(order = 'F')
	else:
		F_flattened = np.hstack((F_flattened, F[filter].flatten(order = 'F')))


S1 = MX @ F_flattened

S2 = MF @ X.flatten(order='F')
print(np.all(S1 == S2))

print(MX.shape)

S = S1.reshape((nf, nlen - 5 + 1))



end = time.time()
print('\n running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms


"""
MX = np.zeros((
(nlen-k+1)*nf,
(k*nf*d)
))
"""
