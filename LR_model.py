from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("LogisticRegressionWithSGD").setExecutorEnv("PYTHON_EGG_CACHE","/tmp/geap")
sc = SparkContext(conf = conf)
sc.addPyFile("hdfs://nameservice1/user/geap/warehouse/lib/numpy.egg")

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

import numpy
import logging

# retain items etype=="pv", chkout==50 and total_price<=100000
# cks, ckp, acc, aid, chkout, ua, res, ip, genre, igenre, itemid, ni, price, ts
def filterPoint(line):
	try:
		value = line.split("\t")
	except Exception, e:
		logging.exception(e)
	#etype = value[1]
	chkout = ""
	try:
		chkout = value[4]
	except Exception, e:
		logging.exception(e)
	if chkout == "50":
		try:
			prices = [int(i) for i in eval(value[12])]
			num = [int(i) for i in eval(value[11])]
			if len(prices) == len(num):
				total_price = sum([a*b for a, b in zip(prices, num)])
			else:
				return False
			if total_price <= 100000:
				return True
		except Exception, e:
			logging.exception(e)
	return False

def parsePoint(line):
	try:
		value = line.split("\t")
		genres = [i.split("/")[0] for i in eval(value[10])]
		return (value[1], genres)
	except Exception, e:
		logging.exception(e)
		return None

def sortPoint(line):
	try:
		values = [column_id.index(i) for i in line]
		values.sort()
		return values
	except Exception, e:
		logging.exception(e)
		return None

def labelData(input):
	try:
		values = input[0]
		genre = input[1]
		if genre in values:
			label = 1
			values.remove(genre)
		else:
			label = 0	
		values = [x if x < genre else x-1 for x in values] #shift the attributes by one index
		ones = [1] * len(values)
		return LabeledPoint(label, SparseVector(column_num-1, values, ones))
	except Exception, e:
		logging.exception(e)
		return None


#set hdfs path
#data = sc.sequenceFile("hdfs://nameservice1/user/geap/warehouse/camus/etl/rat/hourly/2015/06/01/00/*")
data = sc.textFile("hdfs://nameservice1/user/geap/warehouse/geap.db/user_hist_plain/year=2015/*/*/*/*")

parsedData = data.filter(filterPoint).map(parsePoint).filter(lambda kv : kv != None).reduceByKey(lambda x, y : x + y).map(lambda (k, v) : list(set(v)))
parsedData.cache()

#Calculate total number of columns in the dataset
column_num = parsedData.flatMap(lambda _ : _ ).distinct().count()
column_id = parsedData.flatMap(lambda _ : _ ).distinct().collect()
column_id.sort()

#choose a genre to test, default is 100th column as target variable
genre = 1

sortedData = parsedData.map(sortPoint).filter(lambda p : p != None)

labeledData = sortedData.map(lambda line : (line, genre)).map(labelData).filter(lambda p : p != None)

LRSGDmodel = LogisticRegressionWithSGD.train(labeledData)	

print LRSGDmodel.weights

