from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("LogisticRegressionWithSGD").setExecutorEnv(["PYTHON_EGG_CACHE","/tmp/geap"),("SPARK_LIBRARY_PATH", "$SPARK_LIBRARY_PATH:$HADOOP_HOME/lib/native")])
sc = SparkContext(conf = conf)
sc.addPyFile("hdfs://nameservice1/tmp/geap/numpy.egg")

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

import numpy
import json

# retain items etype=="pv", chkout==50 and total_price<=100000
# cks, ckp, acc, aid, chkout, ua, res, ip, genre, igenre, itemid, ni, price, ts
def filterPoint(line):
	value = json.loads(line[1])
	etype = value.get("etype")
	chkout = value.get("chkout")
	if chkout == "50":
		prices = [int(i) for i in value.get("price")]
		num = [int(i) for i in value.get("ni")]
		if len(prices) == len(num):
			total_price = sum([a*b for a, b in zip(prices, num)])
		else:
			return False
		if total_price <= 100000:
			return True
	return False

def parsePoint(line):
	value = json.loads(line[1])
	genres = [i.split("/")[0] for i in value.get("items")]
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


#set hdfs path
data = sc.sequenceFile("hdfs://nameservice1/user/geap/warehouse/camus/etl/rat/hourly/2015/06/01/00/*")
data = sc.sequenceFile("hdfs://localhost:9000/test/*")

parsedData = data.filter(filterPoint).map(parsePoint).reduceByKey(lambda x, y : x + y).map(lambda (k, v) : list(set(v)))
parsedData.cache()

#Calculate total number of columns in the dataset
column_num = parsedData.flatMap(lambda _ : _ ).distinct().count()
column_id = parsedData.flatMap(lambda _ : _ ).distinct().collect()
column_id.sort()

#choose a genre to test, default is 100th column as target variable
genre = 1

sortedData = parsedData.map(sortPoint)

labeledData = sortedData.map(lambda line : (line, genre)).map(labelData)

LRSGDmodel = LogisticRegressionWithSGD.train(labeledData)	

print LRSGDmodel.weights

