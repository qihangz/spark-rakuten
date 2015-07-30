from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark import SparkConf, SparkContext
import json
import numpy

#retain items etype=="pv", chkout==50 and total_price<=100000
def filterPoint(line):
	value = json.loads(line[1])
	etype = value.get("etype")
	chkout = int(value.get("chkout"))
	prices = [int(i) for i in value.get("price")]
	num = [int(i) for i in value.get("ni")]
	total_price = sum([a*b for a, b in zip(prices, num)])
	if etype == "pv" and chkout == 50 and total_price <= 100000:
		return True
	else:
		return False

def parsePoint(line):
	value = json.loads(line[1])
	genres = [int(i.split("/")[0]) for i in value.get("items")]
	return (value.get("uid"), genres)

def sortPoint(line):
	values = [column_id.index(i) for i in line]
	values.sort()
	return values

def labelData(input):
	values = input[0]
	genre = input[1]
	if genre in values:
		label = 1
		values.remove(genre)
	else:
		label = 0	
	values = [x if x < genre else x-1 for x in values] #shift the attributes by one index
	ones = []
	ones = [1] * len(values)
	return LabeledPoint(label, SparseVector(column_num-1, values, ones))

#set master
#conf = SparkConf().setMaster("local[*]").setAppName("LogisticRegressionWithSGD")
#sc = SparkContext(conf = conf)

#set hdfs path
data = sc.sequenceFile("hdfs://localhost:9000/test/*")

parsedData = data.filter(filterPoint).map(parsePoint).reduceByKey(lambda x, y : list(set(x + y))).map(lambda (k, v) : v)
parsedData.cache()

#Calculate total number of columns in the dataset
column_num = parsedData.flatMap(lambda _ : _ ).distinct().count()
column_id = parsedData.flatMap(lambda _ : _ ).distinct().collect()
column_id.sort()

#choose a genre to test, default is 100th column as target variable
genre = 100

sortedData = parsedData.map(sortPoint)

labeledData = sortedData.map(lambda line : (line, genre)).map(labelData)

LRSGDmodel = LogisticRegressionWithSGD.train(labeledData)	

print LRSGDmodel.weights

