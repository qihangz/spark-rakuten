from operator import itemgetter
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest
import numpy
import time
import random

x = numpy.random.choice([0, 1], size = 200, p = [0.95, 0.05])
numpy.random.choice([0, 1], size = 200)
[i for i, v in enumerate(x) if v == 1]

""" Select the top_n features"""
def selectTopNFeatures(top_n):
	data = sc.textFile("test", 80)
	columns = data.flatMap(lambda line : line.split("\t")).map(lambda col : (col, 1)).reduceByKey(lambda x, y : x + y)
	sortedFeatures = sorted(columns.collect(), key=itemgetter(1), reverse=True)
	topFeatures = list(int(feature[0]) for feature in sortedFeatures[0 : top_n]) # select & filter out the word count
	return topFeatures

def sortPoint(line):
	values = [int(x) for x in line.split("\t")]
	values.sort()
	return values

def labelData(input):
	values = input[0]
	i = input[1]
	if i in values:
		label = 1
		values.remove(i)
	else:
		label = 0	
	values = [x if x < i else x-1 for x in values] #shift the attributes by one index
	return LabeledPoint(label, SparseVector(columnNum*10-1, values, numpy.ones(len(values))))

def addFeatures(values):
	randomValues = numpy.random.choice([0, 1], size = columnNum * 9, p = [0.9995, 0.0005])
	values.extend([i+columnNum for i, v in enumerate(randomValues) if v == 1])
	return values

#select_n = 1
#topFeatures = selectTopNFeatures(select_n)

data = sc.textFile("test", 800)
columnNum = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = data.map(sortPoint)
addedSortedData = sortedData.map(addFeatures)
addedSortedData.cache()

precisions = []
recalls = []
alex hongli yin Carnegie mellon
model_start = time.time()

for i in [711]:
	parsedData = addedSortedData.map(lambda line : (line, i)).map(labelData)
	splits = parsedData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	test_set = splits[1]
	#count.append(test_set.filter(lambda lp : lp.label == 1).count())
	time1 = time.time()
	model = NaiveBayes.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)
	"""time2 = time.time()
	model = SVMWithSGD.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time3 = time.time()
	model = LogisticRegressionWithSGD.train(train_set)	
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time4 = time.time()
	model = LogisticRegressionWithLBFGS.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)"""	
	"""time5 = time.time()
	model = DecisionTree.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time6 = time.time()
	model = RandomForest.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)"""	
	time7 = time.time()

end = time.time()

print (end - model_start) / 60
numpy.mean(precisions)
numpy.mean(recalls)
print (end - start) / 60

for i in range(4):
	print i
	print numpy.mean([precisions[x*4+i] for x in range(30)])
	print numpy.mean([recalls[x*4+i] for x in range(30)])

print (time7 - time6) / 60 * 30