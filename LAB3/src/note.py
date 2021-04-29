
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

d = 4 #height
k = 2 #width
nf = 2 #number of filters
F = np.arange((nf*d*k)) + 1
F = F.reshape((nf,d,k))


wid = 4
hei_x = 4
X = np.arange(wid * hei_x) + 1
X = X.reshape((wid, hei_x))

dx, nlen = X.shape


def MakeMFMatrix(F, nlen):

	nf,d,k = F.shape

	print('nlen: ', nlen)

	zero = np.zeros((nf, nlen))

	for filter in range(nf):
		if filter == 0:
			VF = F[filter].flatten(order = 'F')
		else:
			VF = np.vstack((VF, F[filter].flatten(order = 'F')))

	print('VF: ', VF.shape)
	print('zero: ', zero.shape)

	for k in range(3):

		for i in range(3):

			if i == k:
				if i == 0:
					MF = VF
				else:

					MF = np.hstack((MF, VF))
			else:
				if i == 0:
					MF = zero
				else:

					MF = np.hstack((MF, zero))

		if k == 0:
			MF_tot = MF
		else:
			MF_tot = np.vstack((MF_tot, MF))

	print('MF \n', MF_tot)
	print('\n shape: ', MF_tot.shape)
	return MF_tot
	#vec_F = np.kron(I_3, vec_F)
	#print(vec_F)


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
print('=-=-=- MF & X -=-=-= ')

print(MF.shape)
print(X.size)
S2 = MF @ X.flatten(order='F')
#print(S1, S2)


end = time.time()
print('\n running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms


"""
MX = np.zeros((
(nlen-k+1)*nf,
(k*nf*d)
))
"""
