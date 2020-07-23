import spark_eigen
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg.distributed import IndexedRow
from pyspark.mllib.linalg import DenseMatrix
from pyspark.ml.pipeline import Estimator, Model, Pipeline
from pyspark.ml.param.shared import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType
from pyspark.sql import Row
from pyspark import keyword_only  
from pyspark.sql.functions import col
import operator
import functools

class HasDistance():

	#Convert distance matrix of (i-index,j-index,distance) format
	#in an array matrix
	# @param	D		Distance matrix in (i-index,j-index,distance) format
	# @return	numpy.array	Distance matrix
	def __dist_array(self,D):
		dim = int(np.sqrt(len(D)))
		Arr = np.empty([dim,dim]) 
		for d in D:
			Arr[d[0],d[1]] = d[2]

		return Arr

	#Calculate distance matrix with IndexedRows and return a numpy.array matrix
	# @param	rddv1		First RDD with dataset
	# @param	rddv2		Second RDD with dataset
	# @param	sc		SparkContext
	# @return	numpy.array	Distance matrix
	def _dist_matrix(self,rddv1,rddv2,sc):
		dlist1 = rddv1.collect()
		dlist2 = rddv2.collect()
		irows1 = [IndexedRow(i,dlist1[i][0].toArray()) for i in range(0,len(dlist1))]
		irows2 = [IndexedRow(i,dlist2[i][0].toArray()) for i in range(0,len(dlist2))]
		IMatrix1 = IndexedRowMatrix(sc.parallelize(irows1))
		IMatrix2 = IndexedRowMatrix(sc.parallelize(irows2))
		cart = IMatrix1.rows.cartesian(IMatrix2.rows)
		A = cart.map(lambda x : (x[0].index,x[1].index, np.sqrt(np.sum(np.power(np.array(x[0].vector) - np.array(x[1].vector),2))))).collect()
		A.sort()	
		Arr = self.__dist_array(A)
		return Arr	

#Torelance parameter to be considered on eigenvectors calculation
class HasTolerance(Params):

    tol = Param(Params._dummy(), "tol", "tol")

    def __init__(self):
        super(HasTolerance, self).__init__()

    def setTolerance(self, value):
        return self._set(K=tol)

    def getTolerance(self):
        return self.getOrDefault(self.tol)

#Number of the rank reduction for the eigenvectors
class HasK(Params):

    K = Param(Params._dummy(), "K", "K")

    def __init__(self):
        super(HasK, self).__init__()

    def setK(self, value):
        return self._set(K=value)

    def getK(self):
        return self.getOrDefault(self.K)

#Projection of the SpectralClustering to transform new data
class HasProjection(Params):

    projection = Param(Params._dummy(), "projection", "projection")

    def __init__(self):
        super(HasProjection, self).__init__()

    def setProjection(self, value):
        return self._set(projection=value)

    def getProjection(self):
        return self.getOrDefault(self.projection)

#Previous data considered for distance calculation between them and new data
class HasPrevdata(Params):

    prevdata = Param(Params._dummy(), "prevdata", "prevdata")

    def __init__(self):
        super(HasPrevdata, self).__init__()

    def setPrevdata(self, value):
        return self._set(prevdata=value)

    def getPrevdata(self):
        return self.getOrDefault(self.prevdata)

#Estimator of spectral clustering for PySpark
class SpectralClustering( Estimator, HasFeaturesCol, HasOutputCol,
			HasPredictionCol, HasK, HasDistance, HasTolerance,
			# Credits https://stackoverflow.com/a/52467470
			# by https://stackoverflow.com/users/234944/benjamin-manns
			DefaultParamsReadable, DefaultParamsWritable):

	@keyword_only
	def __init__(self, featuresCol=None, outputCol=None, K=3, tol=0.005):
		super(SpectralClustering, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	 # Required in Spark >= 3.0
	def setFeaturesCol(self, value):
		"""
		Sets the value of :py:attr:`featuresCol`.
		""" 
		return self._set(featuresCol=value)

	# Required in Spark >= 3.0
	def setPredictionCol(self, value):
		"""
		Sets the value of :py:attr:`predictionCol`.
		"""
		return self._set(predictionCol=value)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, K=None, tol=0.005):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

	def _fit(self, dataset):
		sc = SparkContext.getOrCreate()
	
		x = dataset.select(self.getFeaturesCol())
		rddv = x.rdd.map(list)
		#calculate distance amtrix
		Aarr = self._dist_matrix(rddv,rddv,sc)
		np.fill_diagonal(Aarr,0)
		D = list(map(lambda x : np.sum(x),Aarr))	
		Darr = np.diag(np.sqrt(np.divide(1,D)))
		#Laplacian matrix
		Ln = D - Aarr
		#Normalize
		Ln = np.matmul(np.matmul(Darr,Ln),Darr)
		#Eigenvectors
		V,U = spark_eigen.eigen(Ln,sc,self.getTolerance())
		#K-Rank reduction
		K = self.getK()
		U.rows.count()
		proj = U.rows.map(lambda x: [x[i] for i in range(0,K)])
		densep = DenseMatrix(proj.count(),K,functools.reduce(operator.iconcat,proj.collect(),[]),True)
		return SpectralClusteringModel(featuresCol=self.getFeaturesCol(), predictionCol=self.getPredictionCol(), projection=densep, prevdata=rddv)
		
#Transformer of spectral clustering for pySpark
class SpectralClusteringModel(Model, HasFeaturesCol, HasPredictionCol,
				HasProjection, HasPrevdata, HasDistance,
				DefaultParamsReadable, DefaultParamsWritable):

	@keyword_only
	def __init__(self, featuresCol=None, predictionCol=None, projection=None, prevdata=None):
		super(SpectralClusteringModel, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, projection=None, prevdata=None):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

	def _transform(self, dataset):
		sc = SparkContext.getOrCreate()

		#Get spectral clustering projecction
		P = self.getProjection()
		#Get data
		x = dataset.select(self.getFeaturesCol())
		rdd2 = x.rdd.map(list)
		#Get data adopted to calculate projection
		rdd = self.getPrevdata()
		#Calculate distance between new data and "training one"
		Aarr = self._dist_matrix(rdd,rdd2,sc)
		Arm = RowMatrix(sc.parallelize(Aarr))
		#Transform new data
		result = Arm.multiply(P)
		df = result.rows.map(lambda x : Row(x.toArray().tolist())).toDF()
		return df.withColumnRenamed("_1","projection")
