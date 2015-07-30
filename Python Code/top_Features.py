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
	topFeatures = list(feature[0] for feature in sortedFeatures[0 : top_n]) # select & filter out the word count
	return topFeatures

def selectFrequentFeatures(freq_threshold):
	data = sc.textFile("test", 80)
	columns = data.flatMap(lambda line : line.split("\t")).map(lambda col : (col, 1)).reduceByKey(lambda x, y : x + y)
	sortedFeatures = sorted(columns.collect(), key=itemgetter(1), reverse=True)
	frequentFeatures = list(feature[0] for feature in sortedFeatures if feature[1] >= freq_threshold) # select & filter out the word count
	return frequentFeatures

def sortAndLabel(line):
	features = [topFeatures.index(x) for x in line.split("\t") if x in topFeatures]
	features.sort()
	labels = [int(x) for x in line.split("\t") if int(x) in random_labels]
	return (features, labels)

def labelData(line):
	features = line[0][0]
	labels = line[0][1]
	label = line[1]
	if label in labels:
		label_val = 1
	else:
		label_val = 0
	return LabeledPoint(label_val, SparseVector(top_n, features, numpy.ones(len(features))))

start = time.time()
col = 297523
top_n = 50

random_n = 10
freq_threshold = 10775
topFeatures = selectTopNFeatures(top_n)
frequentFeatures = selectFrequentFeatures(freq_threshold)

""" Select random_n ramdom non-top Features"""
random_labels = random.sample([feature for feature in frequentFeatures if feature not in topFeatures], random_n)

data = sc.textFile("test", 80)
sorted_labelled = data.map(sortAndLabel)
sorted_labelled.cache()
rows_num = float(sorted_labelled.count())


precisions = []
recalls = []
recallNum = []
sum = 0.0
model_start = time.time()

for label in random_labels:
	parsedData = sorted_labelled.map(lambda line : (line, label)).map(labelData)
	splits = parsedData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	test_set = splits[1]
	test_set.cache()
	model = SVMWithSGD.train(train_set)
	#model = LogisticRegressionWithSGD.train(train_set)
	#model = LogisticRegressionWithLBFGS.train(train_set)
	#model = DecisionTree.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	#model = RandomForest.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, numTrees=5, featureSubsetStrategy="auto", impurity='gini', maxDepth=3, maxBins=32)
	#labelsAndPreds = test_set.map(lambda p: (p.label, model.predict(p.features)))
	#testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test_set.count())
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	if labelsAndPredictions.filter(lambda (v, p): v == 1).count() != 0:
		recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
		recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	else:
		recall = 1.0
	precisions.append(precision)
	recalls.append(recall)

end = time.time()

print (end - model_start) / 60
print (end - start) / 60

print("Average testErr = " + str(sum/random_n))

for item in testErrors:
	print item

