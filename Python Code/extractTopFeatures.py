from operator import itemgetter
import numpy
def selectTopNFeatures(top_n):
	data = sc.textFile("test", 80)
	columns = data.flatMap(lambda line : line.split("\t")).map(lambda col : (col, 1)).reduceByKey(lambda x, y : x + y)
	sortedFeatures = sorted(columns.collect(), key=itemgetter(1), reverse=True)
	topFeatures = list(feature[0] for feature in sortedFeatures[0 : top_n]) # select & filter out the word count
	return topFeatures

def sortAndLabel(line):
	features = [topFeatures.index(x) for x in line.split("\t") if x in topFeatures]
	features.sort()
	if features != []:
		return features

select_n = 20
topFeatures = selectTopNFeatures(select_n)
data = sc.textFile("test", 1)
sorted_labelled = data.map(sortAndLabel).filter(lambda line : line is not None)
sorted_labelled.saveAsTextFile(top)