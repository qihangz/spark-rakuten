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

def findCoveragePercent(socredLabel, recall):
	if recall > 1:
		return 1.0
	sum = socredLabel.sum()
	accum = 0
	for i in range(socredLabel.size):
		accum += socredLabel[i]
		if accum >= recall*sum:
			return i / float(socredLabel.size)

select_n = 30
topFeatures = selectTopNFeatures(select_n)
data = sc.textFile("test", 80)
col = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = data.map(sortPoint)
sortedData.cache()

SVM_percent_5 = []
LRSGD_percent_5 = []
LRLBFGS_percent_5 = []
SVM_percent_10 = []
LRSGD_percent_10 = []
LRLBFGS_percent_10 = []

for i in topFeatures:
	parsedData = sortedData.map(lambda line : (line, i)).map(labelData)
	splits = parsedData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	train_set_5 = train_set.filter(lambda lp : lp.features.indices.size >= 5)
	train_set_5.cache()
	train_set_10 = train_set.filter(lambda lp : lp.features.indices.size >= 20)
	train_set_10.cache()
	test_set = splits[1].filter(lambda lp : lp.features.indices.size >= 5)
	test_set.cache()
	#NBmodel = NaiveBayes.train(train_set)
	#NB_socredLabel = numpy.array(test_set.map(lambda lp: (NBmodel.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	#findCoveragePercent(NB_socredLabel, 0.4)
	SVMSGDmodel_5 = SVMWithSGD.train(train_set_5)
	SVMSGDmodel_5.clearThreshold()
	SVM_socredLabel_5 = numpy.array(test_set.map(lambda lp: (SVMSGDmodel_5.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	SVM_percent_5.append(findCoveragePercent(SVM_socredLabel_5, 0.4))
	SVM_percent_5.append(findCoveragePercent(SVM_socredLabel_5, 0.8))
	SVM_percent_5.append(findCoveragePercent(SVM_socredLabel_5, 1.0))
	##########
	SVMSGDmodel_10 = SVMWithSGD.train(train_set_10)
	SVMSGDmodel_10.clearThreshold()
	SVM_socredLabel_10 = numpy.array(test_set.map(lambda lp: (SVMSGDmodel_10.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	SVM_percent_10.append(findCoveragePercent(SVM_socredLabel_10, 0.4))
	SVM_percent_10.append(findCoveragePercent(SVM_socredLabel_10, 0.8))
	SVM_percent_10.append(findCoveragePercent(SVM_socredLabel_10, 1.0))
	##########
	LRSGDmodel_5 = LogisticRegressionWithSGD.train(train_set_5)	
	LRSGDmodel_5.clearThreshold()
	LRSGD_socredLabel_5 = numpy.array(test_set.map(lambda lp: (LRSGDmodel_5.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRSGD_percent_5.append(findCoveragePercent(LRSGD_socredLabel_5, 0.4))
	LRSGD_percent_5.append(findCoveragePercent(LRSGD_socredLabel_5, 0.8))
	LRSGD_percent_5.append(findCoveragePercent(LRSGD_socredLabel_5, 1.0))
	##########
	LRSGDmodel_10 = LogisticRegressionWithSGD.train(train_set_10)	
	LRSGDmodel_10.clearThreshold()
	LRSGD_socredLabel_10 = numpy.array(test_set.map(lambda lp: (LRSGDmodel_10.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRSGD_percent_10.append(findCoveragePercent(LRSGD_socredLabel_10, 0.4))
	LRSGD_percent_10.append(findCoveragePercent(LRSGD_socredLabel_10, 0.8))
	LRSGD_percent_10.append(findCoveragePercent(LRSGD_socredLabel_10, 1.0))
	##########
	LRLBFGSmodel_5 = LogisticRegressionWithLBFGS.train(train_set_5)
	LRLBFGSmodel_5.clearThreshold()
	LRLBFGS_socredLabel_5 = numpy.array(test_set.map(lambda lp: (LRLBFGSmodel_5.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRLBFGS_percent_5.append(findCoveragePercent(LRLBFGS_socredLabel_5, 0.4))
	LRLBFGS_percent_5.append(findCoveragePercent(LRLBFGS_socredLabel_5, 0.8))
	LRLBFGS_percent_5.append(findCoveragePercent(LRLBFGS_socredLabel_5, 1.0))
	##########
	LRLBFGSmodel_10 = LogisticRegressionWithLBFGS.train(train_set_10)
	LRLBFGSmodel_10.clearThreshold()
	LRLBFGS_socredLabel_10 = numpy.array(test_set.map(lambda lp: (LRLBFGSmodel_10.predict(lp.features), lp.label)).sortByKey(ascending=False).map(lambda (k,v): v).collect())
	LRLBFGS_percent_10.append(findCoveragePercent(LRLBFGS_socredLabel_10, 0.4))
	LRLBFGS_percent_10.append(findCoveragePercent(LRLBFGS_socredLabel_10, 0.8))
	LRLBFGS_percent_10.append(findCoveragePercent(LRLBFGS_socredLabel_10, 1.0))

def printAverage(n):
	for i in range(n):
		print i
		print numpy.mean([SVM_percent_5[x*n+i] for x in range(30)])
		print numpy.mean([SVM_percent_10[x*n+i] for x in range(30)])
		print numpy.mean([LRSGD_percent_5[x*n+i] for x in range(30)])
		print numpy.mean([LRSGD_percent_10[x*n+i] for x in range(30)])
		print numpy.mean([LRLBFGS_percent_5[x*n+i] for x in range(30)])
		print numpy.mean([LRLBFGS_percent_10[x*n+i] for x in range(30)])


def getAccumulatedPercentage(socredLabel):
	result = []
	total = socredLabel.sum()
	accum = 0
	for i in range(socredLabel.size):
		accum += socredLabel[i]
		result.append(accum/total)
	return result

SVM_accum_5 = getAccumulatedPercentage(SVM_socredLabel_5)
LRSGD_accum_5 = getAccumulatedPercentage(LRSGD_socredLabel_5)
LRLBFGS_accum_5 = getAccumulatedPercentage(LRLBFGS_socredLabel_5)

SVM_accum_10 = getAccumulatedPercentage(SVM_socredLabel_10)
LRSGD_accum_10 = getAccumulatedPercentage(LRSGD_socredLabel_10)
LRLBFGS_accum_10 = getAccumulatedPercentage(LRLBFGS_socredLabel_10)

x = (numpy.arange(len(SVM_accum_5)) +1) / float(len(SVM_accum_5))
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
plt.plot(x, SVM_accum_5, 'r', label="SVM")
plt.plot(x, LRSGD_accum_5, 'b', label="LR_SGD")
plt.plot(x, LRLBFGS_accum_5, 'g', label="LR_LBFGS")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()