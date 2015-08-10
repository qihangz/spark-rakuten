from operator import itemgetter
import numpy

path = "test_data.out"

def create_index_dict(inverse=False):
	data = sc.textFile(path, 80)
	index = [int(x) for x in data.flatMap(lambda line : line.split("\t")).distinct().collect()]
	index.sort()
	index_dict = dict()
	if inverse:
		for new, old in enumerate(index):
			index_dict[new] = old		
	else:
		for new, old in enumerate(index):
			index_dict[old] = new
	return index_dict

""" Select the top_n features"""
def selectTopNFeatures(top_n):
	data = sc.textFile(path, 80)
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
		ans.append((i, 1))
	return ans

def check_cooccurrence(line):
	ans = []
	for i in line:
		for j in line:
			key = str(i) + '|' + str(j)
			ans.append((key, 1))
	return ans

index_dict = create_index_dict()
inverse_index_dict = create_index_dict(inverse=True)

data = sc.textFile(path, 80)
colCount = data.flatMap(lambda line : line.split("\t")).distinct().count()

select_n = colCount
topFeatures = selectTopNFeatures(select_n)

sortedData = data.map(sortPoint)#.filter(lambda list : len(list) >= 10)
rowCount = float(sortedData.count())
sortedData.cache()

prob = numpy.zeros(select_n)

count = sortedData.flatMap(check_occurrence).reduceByKey(lambda x, y : x + y).collect()
for item in count:
	prob[topFeatures.index(item[0])] = item[1] / rowCount
"""
X => Y
p(Y|X) = p(Y U X) / p(X)
CPIR(Y|X) = (p(Y|X) - p(Y)) / (1 - p(Y)) if p(Y|X) >= P(Y), p(Y) != 1
CPIR(Y|X) = (p(Y|X) - p(Y)) / (p(Y)) if P(Y) > p(Y|X), p(Y) != 0
"""
CPIR = numpy.empty(select_n * select_n).reshape(select_n, select_n)
CPIR.fill(-1)

cooccurrence = numpy.zeros(select_n * select_n).reshape(select_n, select_n)

def get_CPIR(line):
	parts = line[0].split('|')
	count = line[1]
	i = topFeatures.index(int(parts[0]))
	j = topFeatures.index(int(parts[1]))
	cooccurrence = count / rowCount
	pY_X = count / rowCount / prob[i]
	if pY_X >= prob[j]:
		CPIR = (pY_X - prob[j]) / (1 - prob[j])
	else:
		CPIR = (pY_X - prob[j]) / (prob[j])
	return ((i, j), CPIR, cooccurrence)

fills = sortedData.flatMap(check_cooccurrence).reduceByKey(lambda x, y : x + y).map(get_CPIR).collect()

for fill in fills:
	CPIR[fill[0]] = fill[1]

for fill in fills:
	cooccurrence[fill[0]] = fill[2]

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

def get_positive(frequency_threshold, independence_threshold):
	ans = []
	for i in range(select_n):
		for j in range(select_n):
			if CPIR[i][j] > 0 and i != j:
				if prob[i] > frequency_threshold:
					if abs(prob[i]*prob[j]-cooccurrence[i][j]) > independence_threshold:
						first = genre_dict[topFeatures[i]].split('\t')
						second = genre_dict[topFeatures[j]].split('\t')
						result = ( '(' + first[0] + ' -> ' + second[0] +')', 
							'(' + first[1] + ' -> ' + second[1] +')', 
							CPIR[i][j])
						ans.append(result)
	ans = sorted(ans, key=itemgetter(2), reverse=True)
	return ans

frequency_threshold = 0.00001
independence_threshold = 0.00001
result = get_negative(frequency_threshold, independence_threshold)
len(result)

f = open("negative_correlation_0.00001.csv", "w")

for line in result:
	f.write(line[0] + "\t" + line[1] + '\t' + str(line[2]) + "\n")

f.close()

frequency_threshold = 0.00001
independence_threshold = 0.00001
result = get_positive(frequency_threshold, independence_threshold)
len(result)

f = open("positive_correlation_0.00001.csv", "w")

for line in result:
	f.write(line[0] + "\t" + line[1] + '\t' + str(line[2]) + "\n")

f.close()