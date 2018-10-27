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
        F.append(s)
    return np.array(F)

Xdata,Ydata=np.transpose(np.genfromtxt("signal.dat",delimiter=","))

dx=Xdata[1]-Xdata[0]

plt.figure()
plt.plot(Xdata,Ydata)
plt.savefig("CarantonWilliam_signal.pdf")

F=Fourier(Ydata)
freq=np.fft.fftfreq(len(Ydata),dx)

FR=np.sqrt(np.real(F)**2+np.imag(F)**2)

plt.plot(freq,FR)
plt.savefig("CarantonWilliam_TF.pdf")

FR2=FR
freq_max=[]
F_max=[]

for i in range(6):
	index=list(FR).index(max(FR2))
	if freq[index]>0:
		F_max.append(FR[index])
		freq_max.append(freq[index])
	FR2[index]=0
	
print("Las frecuencias donde están los picos son:",freq_max)
	

