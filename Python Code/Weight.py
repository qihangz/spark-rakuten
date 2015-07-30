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
colCount = data.flatMap(lambda line : line.split("\t")).distinct().count() #col = 297523
sortedData = data.map(sortPoint).filter(lambda list : len(list) >= 10)
rowCount = float(sortedData.count())
sortedData.cache()

matrix = numpy.zeros(select_n * select_n).reshape(select_n, select_n)
count = numpy.zeros(select_n)

for i in range(select_n):
	#positiveCount = sortedData.filter(lambda line : topFeatures[i] in line).count()
	#negativeCount = colCount - positiveCount
	for j in range(select_n):
		matrix[i][j] = sortedData.filter(lambda line : topFeatures[i] in line and topFeatures[j] in line).count() / rowCount

result = matrix + matrix.T

for i in range(select_n):
	result[:, i] /= count[i]

for i in range(select_n):
	result[i][i] = 1

x = numpy.array([[i, j] for i, j in enumerate(SVMSGDmodel.weights) if j < 0])
y = x[x[:,1].argsort()]

f = open("output.csv", "w")

title = "column\t"
for item in topFeatures:
	title = title + str(item) + "\t"

f.write(title+"\n")

for i, item in enumerate(output):
	string = ""
	for element in item:
		string = string + str(element) + "\t"
	f.write(str(topFeatures[i]) + "\t" + string + "\n")

f.close()
