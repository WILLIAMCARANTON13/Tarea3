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


print(len(valores[:100]))

print(np.cov(valores[:100])-cov_matrix(np.transpose(valores[:100])))








