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

def getAccumulatedPercentage(scoredLabel):
	result = []
	total = scoredLabel.sum()
	accum = 0
	for i in range(scoredLabel.size):
		accum += scoredLabel[i]
		result.append(accum/total)
	return result

#select_n = 1
#topFeatures = selectTopNFeatures(select_n)
topFeatures = [711]
data = sc.textFile("test", 80)
col = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = data.map(sortPoint)
sortedData.cache()

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
	SVM_socredLabel = numpy.array(test_set.map(lambda lp: (SVMSGDmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	SVM_accum = getAccumulatedPercentage(SVM_socredLabel)
	LRSGDmodel = LogisticRegressionWithSGD.train(train_set)	
	LRSGDmodel.clearThreshold()
	LRSGD_socredLabel = numpy.array(test_set.map(lambda lp: (LRSGDmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRSGD_accum = getAccumulatedPercentage(LRSGD_socredLabel)
	LRLBFGSmodel = LogisticRegressionWithLBFGS.train(train_set)
	LRLBFGSmodel.clearThreshold()
	LRLBFGS_socredLabel = numpy.array(test_set.map(lambda lp: (LRLBFGSmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRLBFGS_accum = getAccumulatedPercentage(LRLBFGS_socredLabel)

x = (numpy.arange(len(SVM_accum)) +1) / float(len(SVM_accum))
import matplotlib.pyplot as plt
plt.plot(x, SVM_accum, 'r', label="SVM")
plt.plot(x, LRSGD_accum, 'b', label="LR_SGD")
plt.plot(x, LRLBFGS_accum, 'g', label="LR_LBFGS")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

LRSGDweights = LRSGDmodel.weights
SVMweights = SVMSGDmodel.weights

plt.plot(x, SVMweights, 'r', label="SVM")
plt.plot(x, LRSGDweights, 'b', label="LR_SGD")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

LRLBFGSweights = LRLBFGSmodel.weights
plt.plot(x, LRLBFGSweights, 'g', label="LR_LBFGS")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()