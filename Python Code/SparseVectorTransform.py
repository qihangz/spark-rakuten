from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy

col = 29752
n = 1

start = time.time()

def sortPoint(line):
	values = [int(x) for x in line.split("\t")]
	values.sort()
	return values

def parsePoint(input):
	values = input[0]
	i = input[1]
	if i in values:
		label = 1
		values.remove(i)
	else:
		label = 0	
	values = [x if x <i else x-1 for x in values] #shift the attributes by one index
	return LabeledPoint(label, SparseVector(col-1, values, numpy.ones(len(values))))

data = sc.textFile("test", 80)
sortedData = data.map(sortPoint)
sortedData.persist()
rows_num = float(sortedData.count())

trainErrors = []
sum = 0.0

for i in range(n):
	parsedData = sortedData.map(lambda line : (line, i)).map(parsePoint)	
	# Build the model
	model = LogisticRegressionWithSGD.train(parsedData)
	#Evaluating the model on training data
	labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
	trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / rows_num
	sum += trainErr
	trainErrors.append(trainErr)

end = time.time()

print (end - start) / 60

print("Average trainErr = " + str(sum/n))
for item in trainErrors:
	print item