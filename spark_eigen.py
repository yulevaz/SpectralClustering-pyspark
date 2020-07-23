import spark_utils
import numpy as np
from pyspark.mllib.linalg.distributed import RowMatrix

#Define a hessenberg matrix
# @param	X		Considered matrix (numpy.array)
# @return	numpy.array	Hessenberg matrix	
def hessenberg(X):

	n = len(X[:,0])
	H = X
	if n >= 2:

		a = H[1,0]
		H[1:,0] = np.zeros((n-1,))
		if a > 0:
			H[1,0] = a
	
		H[1:,1:] = hessenberg(X[1:,1:])
		return H
	else:
		return X

#Return root mean square residue of two matrix
# @param	uX	pyspark RowMatrix
# @param	uL	pyspark RowMatrix
# @param	sc	spark context because spark QR method does not return two RowMatrix but one RowMatrix and a DenseMatrix
#				so I have to convert it to a RowMatrix with the SparkContext
# @return	float	Residue
def sqr_res(L,sc):
	
	abv = np.abs(L)
	Rt = np.sum(np.tril(abv)-np.diag(np.diag(abv)))/((L.shape[0]**2)/2)
	np.set_printoptions(precision=4,suppress=True)
	print(Rt)
	return Rt 

# QR iteration for calculating eigenpairs
# @param	X		X(t) of qr-eigenvalue algorithm
# @param	Q		prod(Q)_t of qr-eigenvalue algorithm
# @param	sc		spark context because spark QR method does not return two RowMatrix but one RowMatrix and a DenseMatrix
#						so I have to convert it to a RowMatrix with the SparkContext
# @return	RowMatrix,RowMatrix	Candidate eigenvalues and eigenvectors of X
def QR_iter(X,uQ,sc):

	QR = X.tallSkinnyQR(True)	
	Q = QR.Q
	denseQ = spark_utils.to_dense(Q)
	R = RowMatrix(sc.parallelize(QR.R.toArray()))
	L = R.multiply(denseQ)
	U = uQ.multiply(denseQ)

	return L,U

# PySpark eigenvectors and eigenvalues using QR decomposition
# @param	X			numpy.array Square matrix
# @param	sc			spark context because spark QR method does not return two RowMatrix but one RowMatrix and a DenseMatrix
#						so I have to convert it to a RowMatrix with the SparkContext
# @param	tol			tolerance for stop criteria
# @return	RowMatrix,RowMatrix	Eigenvalues and eigenvectors of X
def eigen(X,sc,tol=1e-7):

	dim = X.shape[0]
	H = hessenberg(X) + np.diag([1e-7]*dim)
	H = X
	ann = np.diag([H[dim-1,dim-1]]*dim)
	Hr = RowMatrix(sc.parallelize(H - ann))
	QR = Hr.tallSkinnyQR(True)
	Q = QR.Q
	denseQ = spark_utils.to_dense(Q)
	R = RowMatrix(sc.parallelize(QR.R.toArray()))
	L = R.multiply(denseQ)
	
	H = spark_utils.to_dense(L).toArray() + ann
	
	#stop criterion
	res = sqr_res(H,sc)
	
	while res >= tol:
		ann = np.diag([H[dim-1,dim-1]]*dim)+1e-100
		L = RowMatrix(sc.parallelize(H - ann))
		L,Q = QR_iter(L,Q,sc)
		H = spark_utils.to_dense(L).toArray()
		res = sqr_res(H,sc)
	
	L = RowMatrix(sc.parallelize(H))

	return L,Q
