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

def parseData(line):
	values = [int(x) for x in line.split("\t")]
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
	return LabeledPoint(label, SparseVector(col-1, values, numpy.ones(len(values))))

select_n = 1
topFeatures = selectTopNFeatures(select_n)

data = sc.textFile("test", 80)
parsedData = data.map(parseData)
filteredData = parsedData.filter(lambda line : len(line) >= 5)
col = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = filteredData.map(sortPoint)
sortedData.cache()

precisions = []
recalls = []
model_start = time.time()

accum = sc.accumulator(0)

def accumulate(line):
	accum.add(line[1]) 
	sum += line[1]
	return (line[0], sum)

def findCoveragePercent(ranked_actual_value, recall):
	sum = ranked_actual_value.sum()
	accum = 0
	for i in range(ranked_actual_value.size):
		accum += ranked_actual_value[i]
		if accum >= recall*sum:
			return i / float(ranked_actual_value.size)


for i in topFeatures:
	labeledData = sortedData.map(lambda line : (line, i)).map(labelData)
	splits = labeledData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	test_set = splits[1]
	#count.append(test_set.filter(lambda lp : lp.label == 1).count())
	model = NaiveBayes.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (model.predict(lp.features), lp.label))#.zip(predictions)
	ranked_actual_value = numpy.array(labelsAndPredictions.collect())
	findCoveragePercent(ranked_actual_value, 0.4)
	labelsAndPredictions.cache()
	
	correctPredict = float(labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count())
	precision = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): p == 1).count())
	recall = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	model = SVMWithSGD.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	correctPredict = float(labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count())
	precision = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): p == 1).count())
	recall = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)
	model = LogisticRegressionWithSGD.train(train_set)	
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	correctPredict = float(labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count())
	precision = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): p == 1).count())
	recall = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)
	model = LogisticRegressionWithLBFGS.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	correctPredict = float(labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count())
	precision = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): p == 1).count())
	recall = correctPredict / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)
	"""
	model = DecisionTree.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	model = RandomForest.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: (lp.label, model.predict(lp.features)))#.zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)"""	

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