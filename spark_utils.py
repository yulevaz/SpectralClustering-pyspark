import functools
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import DenseMatrix
import numpy as np
import operator

# Absolute of RowMatrix
# @param	M1	First RowMatrix
# @param	sc	spark context because spark QR method does not return two RowMatrix but one RowMatrix and a DenseMatrix
#				so I have to convert it to a RowMatrix with the SparkContext
# @return	RowMatrix
def spark_abs(M1,sc):
	
	asarr = lambda x : x.toArray().tolist()
	M = M1.rows.collect()
	V1 = list(map(asarr,M))
	L = np.abs(V1).tolist()
	return RowMatrix(sc.parallelize(L))

#Subtract to RowMatrix
# @param	M1	First RowMatrix
# @param	M2	Second RowMatrix
# @param	sc	spark context because spark QR method does not return two RowMatrix but one RowMatrix and a DenseMatrix
#				so I have to convert it to a RowMatrix with the SparkContext
# @return	RowMatrix
def spark_sub(M1,M2,sc):

	V1 = M1.rows.collect()
	V2 = M2.rows.collect()
	lsub = lambda x1,x2 : x1 - x2
	V3 = list(map(lsub,V1,V2))	
	return RowMatrix(sc.parallelize(V3))

#Convert RowMatrix to DenseMatrix
# @param	rowmatrix	RowMatrix to be converted
# @return	DenseMatrix
def to_dense(rowmatrix):

	densev = rowmatrix.rows.collect()
	el = lambda x : [a for a in x]
	M = list(map(el,densev))
	L = functools.reduce(operator.iconcat,M,[])
	return DenseMatrix(rowmatrix.numRows(),rowmatrix.numCols(),L,True)

