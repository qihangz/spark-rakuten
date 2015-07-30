from operator import itemgetter

def parse(line):
	parts = line.split("\t")
	return (parts[0], 1)

data = sc.textFile("item_count.txt", 80)
result = data.map(parse).reduceByKey(lambda x, y : x + y)

count = sorted(result.collect(), key=itemgetter(1), reverse=True)

