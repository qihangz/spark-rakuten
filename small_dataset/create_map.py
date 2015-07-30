data = sc.textFile("test_data", 80)
index = [int(x) for x in data.flatMap(lambda line : line.split("\t")).distinct().collect()]
index.sort()
index_dict = dict()
for new, old in enumerate(index):
	index_dict[old] = new

import numpy
numpy.save("index_dict", index_dict)