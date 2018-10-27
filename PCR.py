import numpy as np
import matplotlib.pyplot as plt

data=open("WDBC.dat").readlines()

data_parsed=[i.split(",") for i in data]

ID=[int(i[0]) for i in data_parsed]

diagnostico=[i[1] for i in data_parsed]

n_data=[]

for i in range(len(data_parsed)):
	n_data.append(data_parsed[i][2:])

valores=np.vectorize(float)(n_data)

def cov_matrix(matriz):
	NX=len(matriz[0]) # número de variables
	N=len(matriz) # numero de datos por varaible
	cov=np.zeros((NX,NX))
	for j in range(NX):
		for k in range(NX):
			s=0
			for i in range(N):
				s+=(matriz[i,j]-np.mean(matriz[:,j]))*(matriz[i,k]-np.mean(matriz[:,k]))
			cov[j,k]+=s
	#if(N%2==1):
	#	return 2*np.array(cov)/(N+1)
	#else:
	#	return 2*np.array(cov)/(N)
	return cov/(N-1)


covarianza=cov_matrix(valores[:50])


a_val,a_vec=np.linalg.eig(covarianza)

#a_vec=np.real(a_vec)
#a_val=np.real(a_val)

for i in range(len(a_val)):
	print("Autovector: ",a_vec[i],"   autovalor correspondiente: ",a_val[i])

a_val=np.ndarray.tolist(a_val)
a_vec=np.ndarray.tolist(a_vec)

indice_mayor_a_val=a_val.index(max(a_val))
PC1=a_vec[indice_mayor_a_val]

a_val2=a_val
a_val[indice_mayor_a_val]=0

indice_mayor_a_val_2=a_val.index(max(a_val2))
PC2=a_vec[indice_mayor_a_val_2]


print("\n Las direcciones a las que los datos muestran mayor tendencia son los autovalores:\n\n",PC1,"\n\n",PC2,"\n\n","elegidos basándose en sus autovalores.")

#print(len(PC1))








