import TitlePreprocessing as ttp
import romkan
import numpy
import re
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import copy

KATAKANA = ur'[\u30A0-\u30FF]'

def parseTitle(line):
	vran = line[0]
	title = line[1]
	words = ttp.clean_title(title).split(" ")
	words = [word for word in words if is_katakana(word)]
	for i in range(len(words)):
		words[i] = vran + "-" + words[i]
	return words

def splitKV(line):
	parts = line.split('-')
	vran = parts[0]
	word = parts[1]
	return (vran, [word])

def joinWord(line):
	key = line[0]
	value = line[1]
	parts = key.split("-")
	return parts[1]+"\t"+parts[0]+"\t"+str(value)

def is_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall(KATAKANA, x) == [] for x in token])

def vectorize(token, length):
	token = re.sub(ur'[^\u30A0-\u30F9]', '', token)
	array = numpy.zeros((90, length))
	for pos, char in enumerate(token):
		array[ord(char)-ord(u"\u30A0"), pos:] += 1
	return normalize(array)

def findMatch(line):
	grouped_data = line[1]
	size = len(grouped_data)
	dim = (size, size)
	similarity = numpy.empty(dim)
	similarity.fill(10)
	for i in range(size):
		for j in range(i+1, size):
			max_len = max(len(grouped_data[i]), len(grouped_data[j]))
			similarity[i][j] = numpy.linalg.norm(vectorize(grouped_data[i], max_len)-vectorize(grouped_data[j], max_len))
			#similarity[i][j] = distance.euclidean(vectorize(grouped_data[i], max_len),vectorize(grouped_data[j], max_len))
	ans = []
	grouped_data_rows = copy.copy(grouped_data)
	grouped_data_cols = copy.copy(grouped_data)
	for i in range(min(dim[0], dim[1])):
		row_index = similarity.argmin() / similarity.shape[1]
		col_index = similarity.argmin() % similarity.shape[1]
		ans.append((grouped_data_rows[row_index],grouped_data_cols[col_index], similarity[row_index,col_index]))
		del grouped_data_rows[row_index]
		del grouped_data_cols[col_index]
		similarity = numpy.delete(similarity, row_index, 0)
		similarity = numpy.delete(similarity, col_index, 1)
	return ans

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
data = sc.parallelize(input_data, 80)
grouped_data = data.flatMap(parseTitle).distinct().map(splitKV).reduceByKey(lambda x, y : x + y).filter(lambda (k, v): k == "149017772091073097").cache()
matches = grouped_data.map(findMatch).flatMap(lambda _ : _ ).distinct().collect()


f = open("match_test.csv", "w")

for match in matches:
	string = ''
	for element in match:
		if type(element) == unicode:
			ele = element.encode('utf8')
		else:
			ele = str(element)
		string = string + ele + "\t"
	f.write(string+"\n")

f.close()