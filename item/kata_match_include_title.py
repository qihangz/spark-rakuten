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
	title = unicode(line[1], 'utf-8')
	words = ttp.clean_title(title).split(" ")
	words = [word for word in words if word and is_katakana(word)]
	for i in range(len(words)):
		words[i] = words[i] + "|" + title
	return words

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

def findMatch(grouped_data):
	size = len(grouped_data)
	dim = (size, size)
	similarity = numpy.empty(dim)
	similarity.fill(10)
	for i in range(size):
		for j in range(i+1, size):
			max_len = max(len(grouped_data[i][0]), len(grouped_data[j][0]))
			similarity[i][j] = numpy.linalg.norm(vectorize(grouped_data[i][0], max_len)-vectorize(grouped_data[j][0], max_len))
			#similarity[i][j] = distance.euclidean(vectorize(grouped_data[i], max_len),vectorize(grouped_data[j], max_len))
	ans = []
	for i in range(size):
		for j in range(i+1, size):
			ans.append((i, j, similarity[i,j]))
	return ans

def splitTitle(line):
	parts = line.split('|')
	word = parts[0]
	title = parts[1]
	return (word, title)

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
data = sc.parallelize(input_data, 80)
grouped_data = data.filter(lambda (k, v): k == "149017772091073097").flatMap(parseTitle).map(splitTitle).reduceByKey(lambda x, y : x + ' | ' + y).collect()

matches = findMatch(grouped_data)
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

import time
start = time.time()
matches = findMatch(grouped_data)
end = time.time()
(end - start)

f = open("match_test_reduced_2csv", "w")

for match in matches:
	if match[2] < 2:
		i = match[0]
		j = match[1]
		sim = str(match[2])
		word_1 = grouped_data[i][0].encode('utf8')
		word_2 = grouped_data[j][0].encode('utf8')
		titles_1 = grouped_data[i][1].encode('utf8')
		titles_2 = grouped_data[j][1].encode('utf8')
		string = word_1 + '\t' + word_2 + '\t' + sim + '\t' + titles_1 + '\t' + titles_2
		f.write(string+"\n")

f.close()