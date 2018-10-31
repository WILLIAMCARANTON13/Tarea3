import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data=mpimg.imread("Arboles.png")

F=np.fft.fft2(data-np.mean(data))

FR=np.sqrt(F.real**2+F.imag**2)
plt.figure(figsize=np.array(np.shape(data))/100*6)
plt.imshow(np.log(np.abs(np.fft.fftshift(F)))**2)
#plt.show()

def filtro(i,j):
    if(abs(j-105)<2 and abs(i-118)<2):
        return 0
    elif(abs(j-152)<2 and abs(i-138)<2):
        return 0
    elif(abs(j-65)<2 and abs(i-65)<2):
        return 0
    elif(abs(j-192)<2 and abs(i-192)<2):
        return 0
    else:
        return 1

matriz=np.zeros(np.shape(data))
for i in range(len(data)):
    for j in range(len(data[0])):
        matriz[i,j]+=filtro(i,j)
filtro_real=np.fft.ifftshift(matriz)

IMG=np.fft.ifft2(F*filtro_real)
plt.imshow(-IMG.real,cmap='Greys')
plt.savefig("CarantonWilliam_Imagen_Filtrada.pdf")


