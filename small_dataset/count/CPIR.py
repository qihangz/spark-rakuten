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

def createIndex():
	data = sc.textFile("test_data", 80)
	index = [int(x) for x in data.flatMap(lambda line : line.split("\t")).distinct().collect()]
	index.sort()
	index_dict = dict()
	for new, old in enumerate(index):
		index_dict[old] = new
	return index_dict

""" Select the top_n features"""
def selectTopNFeatures(top_n):
	data = sc.textFile("test_data", 80)
	columns = data.flatMap(lambda line : line.split("\t")).map(lambda col : (index_dict[int(col)], 1)).reduceByKey(lambda x, y : x + y)
	sortedFeatures = sorted(columns.collect(), key=itemgetter(1), reverse=True)
	topFeatures = list(feature[0] for feature in sortedFeatures[0 : top_n]) # select & filter out the word count
	return topFeatures

def sortPoint(line):
	values = [index_dict[int(x)] for x in line.split("\t")]
	values = list(set(values))
	values.sort()
	return values

index_dict = createIndex()
data = sc.textFile("test_data", 80)
colCount = data.flatMap(lambda line : line.split("\t")).distinct().count()

select_n = colCount
topFeatures = selectTopNFeatures(select_n)

sortedData = data.map(sortPoint)#.filter(lambda list : len(list) >= 10)
rowCount = float(sortedData.count())
sortedData.cache()

CPIR = numpy.zeros(select_n * select_n).reshape(select_n, select_n)
prob = numpy.zeros(select_n)


"""
X => Y
p(Y|X) = p(Y U X) / p(X)
CPIR(Y|X) = (p(Y|X) - p(Y)) / (1 - p(Y)) if p(Y|X) >= P(Y), p(Y) != 1
CPIR(Y|X) = (p(Y|X) - p(Y)) / (p(Y)) if P(Y) > p(Y|X), p(Y) != 0
"""
for i in range(select_n):
	prob[i] = sortedData.filter(lambda line : topFeatures[i] in line).count() / rowCount

for i in range(select_n):
	#positiveCount = sortedData.filter(lambda line : topFeatures[i] in line).count()
	#negativeCount = colCount - positiveCount
	for j in range(select_n):
		pY_X = sortedData.filter(lambda line : topFeatures[j] in line and topFeatures[i] in line).count() / rowCount / prob[i]
		if pY_X >= prob[j]:
			CPIR[i][j] = (pY_X - prob[j]) / (1 - prob[j])
		else:
			CPIR[i][j] = (pY_X - prob[j]) / (prob[j])


f = open("CPIR_output.csv", "w")

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

for i in range(len(output)):
	for j in range(len(output)):
		if output[i][j] < -3:
			print str(topFeatures[i]) + "=>" + str(topFeatures[j])
			print CPIR[i][j]
