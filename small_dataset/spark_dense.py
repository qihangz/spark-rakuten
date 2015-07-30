import numpy

data = sc.textFile("test_data", 80)
index = [int(x) for x in data.flatMap(lambda line : line.split("\t")).distinct().collect()]
index.sort()
index_dict = dict()
for new, old in enumerate(index):
	index_dict[old] = new

col = data.flatMap(lambda line : line.split("\t")).distinct().count()

def convertToDense(line):
	array = [0] * col
	values = [index_dict[int(x)] for x in line.split("\t")]
	for value in values:
		array[value] = 1
	return array

data = sc.textFile("test_data", 80)
denseData = data.map(convertToDense)
splits = denseData.randomSplit((0.7, 0.2, 0.1))
train_data = splits[0].collect()
validate_data = splits[1].collect()
test_data = splits[2].collect()

numpy.save("test_data", test_data)

writeFile(train_data, "train_data.csv")
writeFile(validate_data, "validate_data.csv")
writeFile(test_data, "test_data.csv")