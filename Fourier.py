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

F3=F

for i in range(len(F)):
	if abs(freq[i])>1000:
		F3[i]=0



inversa=np.fft.ifft(F3)

plt.figure()
plt.plot(Xdata,np.real(inversa))
plt.savefig("CarantonWilliam_filtrada.pdf")

X2,Y2=np.transpose(np.genfromtxt("incompletos.dat",delimiter=","))

	
print("No es posible aplicar la transfomada discreta de Fourier porque no son equidistantes los datos")

from scipy import interpolate

def Interpolate2(Xd,Yd,x):
	f_cuad=interpolate.interp1d(Xd,Yd,'quadratic')
	return f_cuad(x)

def Interpolate3(Xd,Yd,x):
	f_cubica=interpolate.interp1d(Xd,Yd,'cubic')
	return f_cubica(x)

X_interpolado=np.linspace(min(X2),max(X2),512)
dx_int=X_interpolado[1]-X_interpolado[0];

Y_cuad=[Interpolate2(X2,Y2,i) for i in X_interpolado]
Y_cubico=[Interpolate3(X2,Y2,i) for i in X_interpolado]

F_interpol_1=Fourier(Y_cuad)
F_1_norm=np.sqrt(np.real(F_interpol_1)**2+np.imag(F_interpol_1)**2)
freq_interpol_1=np.fft.fftfreq(len(Y_cuad),dx_int)

F_interpol_2=Fourier(Y_cubico)
F_2_norm=np.sqrt(np.real(F_interpol_2)**2+np.imag(F_interpol_2)**2)
freq_interpol_2=np.fft.fftfreq(len(Y_cubico),dx_int)


f, axarr = plt.subplots(2)
axarr[0].plot(freq, FR)
axarr[1].plot(freq_interpol_1,F_1_norm)

plt.show()





























