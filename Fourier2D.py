import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data=mpimg.imread("Arboles.png")

F=np.fft.fft2(data-np.mean(data))

FR=np.sqrt(F.real**2+F.imag**2)
plt.figure(figsize=np.array(np.shape(data))/100*6)
plt.imshow(np.log(np.abs(np.fft.fftshift(F)))**2)

plt.show()