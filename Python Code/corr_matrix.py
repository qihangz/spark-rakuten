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

select_n = 30
topFeatures = selectTopNFeatures(select_n)

data = sc.textFile("test", 80)
col = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = data.map(sortPoint)
sortedData.cache()

result = []

for i in topFeatures:
	parsedData = sortedData.map(lambda line : (line, i)).map(labelData)
	SVMSGDmodel = SVMWithSGD.train(parsedData)
	result.append([i, numpy.insert(SVMSGDmodel.weights, i, None)[topFeatures]])


x = numpy.array([[i, j] for i, j in enumerate(SVMSGDmodel.weights) if j < 0])
y = x[x[:,1].argsort()]

f = open("matrix.csv", "w")

title = "column\t"
for item in result:
	title = title + str(item[0]) + "\t"

f.write(title+"\n")

for item in result:
	string = ""
	for element in item[1]:
		string = string + str(element) + "\t"
	f.write(str(item[0]) + "\t" + string+"\n")

f.close()