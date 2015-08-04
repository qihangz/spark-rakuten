import numpy

def join(line):
	alphabet = line[0]
	katakana = line[1]
	return alphabet + '|' + katakana

def disjoin(line):
	parts = line.split('|')
	alphabet = parts[0]
	katakana = parts[1]
	return (alphabet, katakana)

threshold = 1

input_data = numpy.array(numpy.loadtxt("alphabet_kata.csv", delimiter=",", dtype="string"))
input_data = numpy.array([data for data in input_data if float(data[2]) < threshold])
input_data = input_data[:,:2]
data = sc.parallelize(input_data, 80)
result = data.map(join).distinct().map(disjoin).reduceByKey(lambda x, y : x + '\t' + y).map(lambda (k, v): k + ':' + v).collect()

f = open("dictionary_threshold=1", "w")

for line in result:
	f.write(line+"\n")

f.close()