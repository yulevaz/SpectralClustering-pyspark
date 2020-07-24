# SpectralClustering-pyspark
An implementation of eigenvalues and spectral clustering for pyspark. Note that it is not optimized and requires multiple type conversions given how matrix operations are supported in such framework. Advices on how to circumvent such drawback are welcome.

Also, eigenvalues calculation is sensible with respect to the tolerance of the algorithm stop criterion.
