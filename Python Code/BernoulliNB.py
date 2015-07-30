import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)
print(clf.predict(X[2]))




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
col = 29752
top_n = 200
random_n = 30

topFeatures = selectTopNFeatures(top_n)

""" Select random_n ramdom non-top Features"""
random_labels = random.sample([feature for feature in range(col) if feature not in topFeatures], random_n)

data = sc.textFile("test", 80)
sorted_labelled = data.map(sortAndLabel)
sorted_labelled.cache()
rows_num = float(sorted_labelled.count())

precisions = []
recalls = []
sum = 0.0
model_start = time.time()

for label in random_labels:
	parsedData = sorted_labelled.map(lambda line : (line, label)).map(labelData)
	splits = parsedData.randomSplit((0.9, 0.1))
	train_set = splits[0]
	test_set = splits[1]
	train_set.map()
	#model = SVMWithSGD.train(train_set)
	#model = LogisticRegressionWithSGD.train(train_set)
	#model = LogisticRegressionWithLBFGS.train(train_set)
	#model = DecisionTree.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	#model = RandomForest.trainClassifier(train_set, numClasses=2, categoricalFeaturesInfo={}, numTrees=5, featureSubsetStrategy="auto", impurity='gini', maxDepth=3, maxBins=32)
	#labelsAndPreds = test_set.map(lambda p: (p.label, model.predict(p.features)))
	#testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test_set.count())
	predictions = model.predict(test_set.map(lambda x: x.features))
	labelsAndPredictions = test_set.map(lambda lp: lp.label).zip(predictions)
	labelsAndPredictions.cache()
	precision = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(rows_num * 0.1)
	recall = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count() / float(labelsAndPredictions.filter(lambda (v, p): v == 1).count())
	precisions.append(precision)
	recalls.append(recall)

end = time.time()

print (end - model_start) / 60
print (end - start) / 60

print("Average testErr = " + str(sum/random_n))

for item in testErrors:
	print item


row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
model = BernoulliNB()
model.fit(csr_matrix((data, (row, col)), shape=(3, 3)),[0, 1, 0])