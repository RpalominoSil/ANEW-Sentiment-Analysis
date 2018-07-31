Experimento ASBCT
----------------
Dataset:
name: asbct.arff
task: multilabel
number of instances: 3000
category or labels: binary
labels: (6)
	unpleasant
	pleasant
	calm
	excited
	outOfControl
	inControl
number of attributes: 1034

Parameters:
----------
Cross-validation iterations: 3 to 10
K-neighbours iterations: 1 to 20
Algorithms: (11)
	GeneralB (baseline)
	mLkNN
	bRkNN
        bRkNN_a
	bRkNN_
	mlmut
	mlmut_a
	mlmut_b
	mlnotmut
	mlnotmut_a
	mlnotmut_b
Distance Functions: (5)
	Euclidean
	Manhattan
	Minkowski
	Chebyshev
	Cosine
