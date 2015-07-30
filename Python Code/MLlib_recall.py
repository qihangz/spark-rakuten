from operator import itemgetter
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import SVMWithSGD, SVMModel
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

def findCoveragePercent(scoredLabel, recall):
	if recall > 1:
		return 1.0
	sum = scoredLabel.sum()
	accum = 0
	for i in range(scoredLabel.size):
		accum += scoredLabel[i]
		if accum >= recall*sum:
			return i / float(scoredLabel.size)

select_n = 1
topFeatures = selectTopNFeatures(select_n)

data = sc.textFile("test", 80)
col = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = data.map(sortPoint)
sortedData.cache()

NB_percent = []
LRSGD_percent = []
LRLBFGS_percent = []

for i in topFeatures:
	parsedData = sortedData.map(lambda line : (line, i)).map(labelData)
	splits = parsedData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	train_set.cache()
	test_set = splits[1]
	test_set.cache()
	#NBmodel = NaiveBayes.train(train_set)
	#NB_socredLabel = numpy.array(test_set.map(lambda lp: (NBmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	#findCoveragePercent(NB_socredLabel, 0.4)
	SVMSGDmodel = SVMWithSGD.train(train_set)
	SVMSGDmodel.clearThreshold()
	SVM_scoredLabel = numpy.array(test_set.map(lambda lp: (SVMSGDmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	
	SVM_percent.append(findCoveragePercent(SVM_scoredLabel, 0.4))
	SVM_percent.append(findCoveragePercent(SVM_scoredLabel, 0.8))
	SVM_percent.append(findCoveragePercent(SVM_scoredLabel, 1.0))
	LRSGDmodel = LogisticRegressionWithSGD.train(train_set)	
	LRSGDmodel.clearThreshold()
	LRSGD_scoedLabel = numpy.array(test_set.map(lambda lp: (LRSGDmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRSGD_percent.append(findCoveragePercent(LRSGD_scoedLabel, 0.4))
	LRSGD_percent.append(findCoveragePercent(LRSGD_scoedLabel, 0.8))
	LRSGD_percent.append(findCoveragePercent(LRSGD_scoedLabel, 1.0))
	LRLBFGSmodel = LogisticRegressionWithLBFGS.train(train_set)
	LRLBFGSmodel.clearThreshold()
	LRLBFGS_scoredLabel = numpy.array(test_set.map(lambda lp: (LRLBFGSmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRLBFGS_percent.append(findCoveragePercent(LRLBFGS_scoredLabel, 0.4))
	LRLBFGS_percent.append(findCoveragePercent(LRLBFGS_scoredLabel, 0.8))
	LRLBFGS_percent.append(findCoveragePercent(LRLBFGS_scoredLabel, 1.0))

def getAccumulatedPercentage(socredLabel):
	result = []
	total = socredLabel.sum()
	accum = 0
	for i in range(socredLabel.size):
		accum += socredLabel[i]
		result.append(accum/total)
	return result
SVM_accum = getAccumulatedPercentage(SVM_socredLabel)
LRSGDmodel = SVMWithSGD.train(train_set)
LRSGDmodel = LogisticRegressionWithSGD.train(train_set)	
LRLBFGSmodel = LogisticRegressionWithLBFGS.train(train_set)
labelsAndPredictions = test_set.map(lambda lp: (LRLBFGSmodel.predict(lp.features), lp.label)).zipWithIndex().filter(lambda line: line[0][0] == 1 and line[0][1] == 0).map(lambda line: line[1]).collect()


x = numpy.arange(len(SVM_accum))
import matplotlib.pyplot as plt
plt.plot(x, SVM_accum, 'r', label="SVM")
plt.plot(x, t)
plt.show()
y = numpy.array(labelsAndPredictions)

t= [SVM_accum[i] if i in labelsAndPredictions else 0 for i in x]

for i in x:
	if i in labelsAndPredictions:
		t.append(SVM_accum[i])
	else:
		t.append(0)		