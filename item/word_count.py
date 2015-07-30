import TitlePreprocessing as ttp
import numpy

def splitWord(line):
	vran = line[0]
	title = line[1]
	words = ttp.clean_title(title).split(" ")
	words = filter(None, words)
	for i in range(len(words)):
		words[i] = vran + "-" + words[i]
	return words

def joinWord(line):
	key = line[0]
	value = line[1]
	parts = key.split("-")
	return parts[1]+"\t"+parts[0]+"\t"+str(value)

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
data = sc.parallelize(input_data, 80)
result = data.map(splitWord).flatMap(lambda words: words).map(lambda word : (word, 1)).reduceByKey(lambda x, y : x + y)
result.map(joinWord).saveAsTextFile("item_count")