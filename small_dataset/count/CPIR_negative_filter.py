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

def check_occurrence(line):
	ans = []
	for i in line:
		for j in line:
			key = str(i) + '|' + str(j)
			ans.append((key, 1))
	return ans

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

for i in range(select_n):
	prob[i] = sortedData.filter(lambda line : topFeatures[i] in line).count() / rowCount

count_result = numpy.array(sortedData.flatMap(check_occurrence).reduceByKey(lambda x, y : x + y).collect())

"""
X => Y
p(Y|X) = p(Y U X) / p(X)
CPIR(Y|X) = (p(Y|X) - p(Y)) / (1 - p(Y)) if p(Y|X) >= P(Y), p(Y) != 1
CPIR(Y|X) = (p(Y|X) - p(Y)) / (p(Y)) if P(Y) > p(Y|X), p(Y) != 0
"""

for i in range(select_n):
	for j in range(select_n):
		key = str(topFeatures[i]) + "|" + str(topFeatures[j])
		if key in count_result[:,0]:
			index = numpy.where(count_result[:,0]==key)[0][0]
			pY_X = int(count_result[index,1]) / rowCount / prob[i]
		else:
			pY_X = 0
		if pY_X >= prob[j]:
			CPIR[i][j] = (pY_X - prob[j]) / (1 - prob[j])
		else:
			CPIR[i][j] = (pY_X - prob[j]) / (prob[j])


cooccurrence = numpy.zeros(select_n * select_n).reshape(select_n, select_n)
for i in range(select_n):
	for j in range(select_n):
		key = str(topFeatures[i]) + "|" + str(topFeatures[j])
		if key in count_result[:,0]:
			index = numpy.where(count_result[:,0]==key)[0][0]
			cooccurrence[i][j] = int(count_result[index,1]) / rowCount
		else:
			cooccurrence[i][j] = 0

import pickle
f = open("translate_index_dict.pkl")
inverse_index_dict = pickle.load(f)
f.close()

for i in range(len(topFeatures)):
	topFeatures[i] = inverse_index_dict[topFeatures[i]]

genres = numpy.array(numpy.loadtxt("test_data_genre_meta", delimiter="\t", dtype="string"))

genre_dict = {}

for genre in genres:
	genre_dict[int(genre[0])] = genre[1] + '\t' + genre[2]


def get_negative(frequency_threshold, independence_threshold):
	ans = []
	for i in range(select_n):
		for j in range(select_n):
			if CPIR[i][j] < 0:
				if prob[i] > frequency_threshold:
					if abs(prob[i]*prob[j]-cooccurrence[i][j]) > independence_threshold:
						first = genre_dict[topFeatures[i]].split('\t')
						second = genre_dict[topFeatures[j]].split('\t')
						result = ( '(' + first[0] + ' -> ' + second[0] +')', 
							'(' + first[1] + ' -> ' + second[1] +')', 
							CPIR[i][j])
						ans.append(result)
	ans = sorted(ans, key=itemgetter(2))
	return ans

frequency_threshold = 0.00001
independence_threshold = 0.00001
result = get_negative(frequency_threshold, independence_threshold)
len(result)

f = open("negative_correlation_0.00005", "w")

for line in result:
	f.write(line[0] + "\t" + line[1] + '\t' + str(line[2]) + "\n")

f.close()
