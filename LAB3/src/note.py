
import numpy as np
import time

start = time.time()

"""
# 6 x 6
X = np.arange(6*6*4) + 1
X = X.reshape((4, 6,6))


F = np.arange((3*3*4))
F = F.reshape((4, 3,3))
FF = F.flatten().reshape(-1, 1)


d, w, h, = X.shape
Nfilter, f, _, = F.shape

ind = np.arange(f + 1)


MX = np.zeros(((w - f + 1)*(h - f + 1), Nfilter*f**2))

row = 0
for k in range(Nfilter):
    for i in range(w - f + 1):
        for j in range(h - f + 1):
            #vals = X[(ind[0] + i):(ind[-1] + i), (ind[0] + j):(ind[-1] + j)]

            row = j + (i)*(h - f + 1) + k*(w - f + 1) + k*(i)*(h - f + 1)
            print(row)
            #MX[row] = vals.flatten()

print(MX.shape)

#S = MX @ FF

#print(S)

end = time.time()
print('\n running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms



def MXgenerator(X, F):

    dx, w, h = X.shape
    d, f, f  = F.shape
    ind = np.arange(f + 1)

    def MX(x):
        MX = np.zeros(((w - f + 1)*(h - f + 1), f**2))
        for i in range(w - f + 1):
            for j in range(h - f + 1):
                vals = x[(ind[0] + i):(ind[-1] + i), (ind[0] + j):(ind[-1] + j)]
                row = j + (i)*(w - f + 1)
                MX[row] = vals.flatten()

        return MX

    for k in range(dx):

        if k == 0:
            MX_tot = MX(X[k])
        else:
            current_MX = MX(X[k])
            MX_tot = np.hstack((MX_tot, current_MX))

    return MX_tot


"""
# 6 x 6
X = np.arange(1*28*19) + 1
X = X.reshape((1,28,19))

F = np.arange((1*3*3)) + 1
F = F.reshape((1,3,3))



def MXgenerator(X, F):

    dx, w, h = X.shape
    d, f, f  = F.shape
    ind = np.arange(f + 1)

    def MX(x):
        MX = np.zeros(((w - f + 1)*(h - f + 1), f**2))
        for i in range(w - f + 1):
            for j in range(h - f + 1):
                vals = x[(ind[0] + i):(ind[-1] + i), (ind[0] + j):(ind[-1] + j)]
                row = j + (i)*(w - f + 1)
                MX[row] = vals.flatten().reshape(-1, 1)

        return MX

    for k in range(dx):

        if k == 0:
            MX_tot = MX(X[k])
        else:
            current_MX = MX(X[k])
            MX_tot = np.hstack((MX_tot, current_MX))

    return MX_tot



#FF = F.flatten().reshape(-1, 1)

def MFgenerator(X, F, NLongest):

    height_x, width_x = X.shape
    depth_f, height_f, width_f = F.shape

    ind_F = np.arange(height_f*width_f)

    FF = F.flatten().reshape(-1, 1)

    MF = np.zeros((
    (width_x - width_f + 1)*(height_x - height_f + 1),
     width_x*height_x))

    vec = MF[0]

    jump = width_x - width_f    # >> 3

    for i in range((width_x - width_f + 1)*(height_x - height_f + 1)):

        curr_F = np.zeros((width_x*height_x))
        curr_jump = 0

        for j in range(ind_F.size):

            if j % width_f == 0 and j != 0:
                curr_jump += jump

            index = j + curr_jump + i

            curr_F[index] = FF[j]

        MF[i] = curr_F

    print(MF)
    return MF


#S = MX @ FF
MF = MXgenerator(X, F)
print(MF)

end = time.time()
print('\n running time: ', str((end - start)*1000)[:6], 'ms') # >> 160 ms
