from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMWithSGD
import numpy
import time

col = 29752
n = 1

start = time.time()

def sortPoint(line):
	values = [int(x) for x in line.split("\t")]
	values.sort()
	return values

def parsePoint(input):
	values = input[0]
	i = input[1]
	if i in values:
		label = 1
		values.remove(i)
	else:
		label = 0
	values = [x if x <i else x-1 for x in values] #shift the attributes by one index
	return LabeledPoint(label, SparseVector(col-1, values, numpy.ones(len(values))))

data = sc.textFile("test", 64)
sortedData = data.map(sortPoint)
sortedData.persist()
rows_num = float(sortedData.count())

trainErrors = []
sum = 0.0

for i in range(n):
	parsedData = sortedData.map(lambda line : (line, i)).map(parsePoint)
	parsedData.persist()
	model_start = time.time()	
	#model = NaiveBayes.train(parsedData)
	model = SVMWithSGD.train(parsedData)
	#model = LogisticRegressionWithSGD.train(parsedData)
	#model = LogisticRegressionWithLBFGS.train(parsedData)
	#model = DecisionTree.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	#model = RandomForest.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
	model_end = time.time()
	labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
	error_start = time.time()		
	trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / rows_num
	error_end = time.time()
	sum += trainErr
	trainErrors.append(trainErr)

end = time.time()
print (end - start)
print (model_end - model_start)
print (error_end - error_start)