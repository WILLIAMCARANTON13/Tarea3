import numpy as np
import matplotlib.pyplot as plt

i=1j
e=np.e
pi=np.pi

def pasa_bajos(trans,frecs,corte):
	filtrado=np.copy(trans)
	for i in range(len(trans)):
		if abs(frecs[i])>corte:
			filtrado[i]=0
	return filtrado
		

def Fourier(y):
	# Implementación propia de la transformada discreta de fourier
    N=len(y)
    F=[]
    for k in range(N):
        s=0+0j
        for n in range(N):
            s+=y[n]*np.exp(-(2j*pi*k*n)/N)
        F.append(s)
    return np.array(F)

Xdata,Ydata=np.transpose(np.genfromtxt("signal.dat",delimiter=","))

dx=Xdata[1]-Xdata[0]

plt.figure()
plt.plot(Xdata,Ydata)
plt.savefig("CarantonWilliam_signal.pdf")
plt.close()

F=Fourier(Ydata)
freq=np.fft.fftfreq(len(Ydata),dx)

FR=np.sqrt(np.real(F)**2+np.imag(F)**2)

plt.plot(freq,FR)
plt.savefig("CarantonWilliam_TF.pdf")
plt.close()

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

F3=pasa_bajos(F,freq,1000)
inversa=np.fft.ifft(F3)

plt.figure()
plt.plot(Xdata,np.real(inversa))
plt.savefig("CarantonWilliam_filtrada.pdf")
plt.close()

X2,Y2=np.transpose(np.genfromtxt("incompletos.dat",delimiter=","))


print("No es posible aplicar la transfomada discreta de Fourier porque no son equidistantes los datos")

from scipy import interpolate

X_continuo=np.linspace(min(X2),max(X2),512)
dx_continuo=X_continuo[1]-X_continuo[0];

cuad=interpolate.interp1d(X2,Y2,"quadratic")
cub=interpolate.interp1d(X2,Y2,"cubic")

Ycuad=cuad(X_continuo)
Ycub=cub(X_continuo)

Fcuad=Fourier(np.array(Ycuad))
FRcuad=np.sqrt(np.real(Fcuad)**2+np.imag(Fcuad)**2)
Fcub=Fourier(np.array(Ycub))
FRcub=np.sqrt(np.real(Fcub)**2+np.imag(Fcuad)**2)
freq_continuo=np.fft.fftfreq(len(Ycuad),dx_continuo)

fig, ax = plt.subplots(3,sharex=True)
ax[0].plot(freq, FR)
ax[0].set_ylabel("signal.dat")
ax[1].plot(freq_continuo,FRcuad)
ax[1].set_ylabel("spline cuadrático")
ax[2].plot(freq_continuo,FRcub)
ax[2].set_ylabel("spline cúbico")

plt.savefig("CarantonWilliam_TF_Interpola.pdf")
plt.close()

F500=pasa_bajos(F,freq,500)
F1K=pasa_bajos(F,freq,1000)
Fcuad500=pasa_bajos(Fcuad,freq_continuo,500)
Fcuad1K=pasa_bajos(Fcuad,freq_continuo,1000)
Fcub500=pasa_bajos(Fcub,freq_continuo,500)
Fcub1K=pasa_bajos(Fcub,freq_continuo,1000)

_500=np.fft.ifft(F500)
_1K=np.fft.ifft(F1K)
cuad500=np.fft.ifft(Fcuad500)
cuad1K=np.fft.ifft(Fcuad1K)
cub500=np.fft.ifft(Fcub500)
cub1K=np.fft.ifft(Fcub1K)

fig, ax = plt.subplots(3,2,sharex=True,sharey=True)

ax[0,0].plot(Xdata,_500.real)
ax[0,1].plot(Xdata,_1K.real)
ax[1,0].plot(X_continuo,cuad500.real)
ax[1,1].plot(X_continuo,cuad1K.real)
ax[2,0].plot(X_continuo,cub500.real)
ax[2,1].plot(X_continuo,cub1K.real)

ax[0,0].set_ylabel("signal.dat")
ax[1,0].set_ylabel("spline cuadrático")
ax[2,0].set_ylabel("spline cúbico")


ax[2,0].set_xlabel("Pasabajos con 500Hz")
ax[2,1].set_xlabel("Pasabajos con 1KHz")

plt.savefig("CarantonWilliam_2Filtros.pdf")




























