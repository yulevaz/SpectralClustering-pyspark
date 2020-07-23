import numpy as np
from numpy import linalg
from scipy import spatial
import spark_eigen

# Numeric test for spark_eigen comparing 
def test_distance():

	dim = 10
	nsamples = 1000
		  
	data = [[np.random.uniform(0,1) for j in range(0,dim)] for i in range(0,nsamples)]
	D = linalg.distance_matrix(data,data)

	#generating invertible matrix
	while linalg.matrix_rank(D) < dim:
		data = [[np.random.uniform(0,1) for j in range(0,dim)] for i in range(0,nsamples)]
		D = linalg.distance_matrix(data,data)
		
	L,U = spark_eigen.eigen(D)
	eigenval = L.	
