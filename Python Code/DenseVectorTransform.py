#DENSE VECTOR
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy

col = 29752

# Load and parse the data
# Select Y index before parsePoint!
def parsePoint(input):
	line = input[0]
	i = input[1]
	array = numpy.zeros(col, dtype=int)
	values = [int(x) for x in line.split("\t")]
	for value in values:
		array[value] = 1
	label = array[i]
	numpy.delete(array, i)
	return LabeledPoint(label, array)

data = sc.textFile("trial")
parsedData = data.map(lambda line : (line, 0)).map(parsePoint)

# Build the model
model1 = LogisticRegressionWithLBFGS.train(parsedData)

#Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))