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

def sortAndLabel(line):
	features = [topFeatures.index(x) for x in line.split("\t") if x in topFeatures]
	features.sort()
	labels = [x for x in line.split("\t") if x in frequentFeatures]
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
select_n = 800
top_n = 1000
random_n = 30
topFeatures = selectTopNFeatures(select_n)
frequentFeatures = selectTopNFeatures(top_n)

""" Select random_n ramdom non-top Features"""
random_labels = random.sample([feature for feature in frequentFeatures if feature not in topFeatures], random_n)

data = sc.textFile("test", 80)
sorted_labelled = data.map(sortAndLabel)
sorted_labelled.cache()
#rows_num = float(sorted_labelled.count())

precisions = []
recalls = []
#recallNum = []
#count = []
model_start = time.time()

for label in random_labels: #[feature for feature in frequentFeatures if feature not in topFeatures]:
	parsedData = sorted_labelled.map(lambda line : (line, label)).map(labelData)
	splits = parsedData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	train_set.cache()
	test_set = splits[1]
	test_set.cache()
	#count.append(test_set.filter(lambda lp : lp.label == 1).count())
	time1 = time.time()
	model = NaiveBayes.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time2 = time.time()
	model = SVMWithSGD.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time3 = time.time()
	model = LogisticRegressionWithSGD.train(train_set)	
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time4 = time.time()
	model = LogisticRegressionWithLBFGS.train(train_set)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	"""time5 = time.time()
	model = DecisionTree.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)"""	
	time6 = time.time()
	model = RandomForest.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test_set.count())
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	#recallNum.append(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)	
	time7 = time.time()

	#labelsAndPreds = test_set.map(lambda p: (p.label, model.predict(p.features)))
	#testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test_set.count())	

end = time.time()

print (end - model_start) / 60
numpy.mean(precisions)
numpy.mean(recalls)
print (end - start) / 60

for i in range(5):
	print i
	print numpy.mean([precisions[x*5+i] for x in range(30)])
	print numpy.mean([recalls[x*5+i] for x in range(30)])

print (time7 - time6) / 60 * 30