#DENSE VECTOR
from operator import itemgetter
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest
import numpy
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy

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

# Load and parse the data
# Select Y index before parsePoint!
def parsePoint(input):
	line = input[0]
	i = input[1]
	values = [index_dict[int(x)] for x in line.split("\t")]
	if len(values) == 1 and i == values[0]:
		return None
	else:
		array = numpy.zeros(col, dtype=int)
		for value in values:
			array[value] = 1
		label = array[i]
		numpy.delete(array, i)
		return LabeledPoint(label, array)

index_dict = createIndex()

translate_index_dict = {}

for k in index_dict:
	translate_index_dict[index_dict[k]] = k

data = sc.textFile("test_data", 80)
col = data.flatMap(lambda line : line.split("\t")).distinct().count()


select_n = 10
topFeatures = selectTopNFeatures(select_n)

result = []
for i in topFeatures:
	parsedData = data.map(lambda line : (line, i)).map(parsePoint).filter(lambda data : data != None)
	LBFGS_model = LogisticRegressionWithLBFGS.train(parsedData)
	result.append([i, numpy.insert(LBFGS_model.weights, i, None)[topFeatures]])


f = open("dense_matrix.csv", "w")

title = "column\t"
for item in result:
	title = title + str(translate_index_dict[item[0]]) + "\t"

f.write(title+"\n")

for item in result:
	string = ""
	for element in item[1]:
		string = string + str(element) + "\t"
	f.write(str(translate_index_dict[item[0]]) + "\t" + string+"\n")

f.close()