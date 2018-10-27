import numpy as np
import matplotlib.pyplot as plt

i=1j
e=np.e
pi=np.pi

def Fourier(y):
	# Implementación propia de la transformada discreta de fourier
    N=len(y)
    F=[]
    for k in range(N):
        s=0
        for n in range(N):
            s+=y[n]*e**(-(2*pi*i*k*n)/N)
        F.append(s/N)
    return np.array(F)

Xdata,Ydata=np.transpose(np.genfromtxt("signal.dat",delimiter=","))

plt.figure()
plt.plot(Xdata,Ydata)
plt.savefig("CarantonWilliam_signal.pdf")
