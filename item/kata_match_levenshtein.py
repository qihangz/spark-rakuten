import TitlePreprocessing as ttp
import romkan
import numpy
import re
import Levenshtein

KATAKANA = ur'[\u30A0-\u30FF]'

def parseTitle(line):
	vran = line[0]
	title = line[1]
	words = ttp.clean_title(title).split(" ")
	words = [word for word in words if is_katakana(word) and word]
	for i in range(len(words)):
		words[i] = vran + "-" + words[i]
	return words

def splitKV(line):
	parts = line.split('-')
	vran = parts[0]
	word = parts[1]
	return (vran, [word])

def is_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall(KATAKANA, x) == [] for x in token])

def clean_token(token):
	token = re.sub(ur'[^\u30A0-\u30F9]', '', token)
	return token

def findDistance(line):
	vran = line[0]
	grouped_data = line[1]
	ans = []
	size = len(grouped_data)
	dim = (size, size)
	for i in range(size):
		word_1 = clean_token(grouped_data[i])
		for j in range(i+1, size):
			word_2 = clean_token(grouped_data[j])
			distance = Levenshtein.distance(word_1,word_2) / (float(len(word_1) + len(word_2)) / 2)
			if distance < 0.5:
				ans.append( (vran, grouped_data[i], grouped_data[j], distance) )	
	return ans

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
data = sc.parallelize(input_data, 80)
grouped_data = data.flatMap(parseTitle).distinct().map(splitKV).reduceByKey(lambda x, y : x + y).cache()
matches = grouped_data.flatMap(findDistance).distinct().collect()

f = open("kata_match_levenshtein.csv", "w")

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

def save_to_file(matches, key):
	f = open('/'.join(('katakana_pairs', key + '.csv')), 'w')
	for match in matches:
		if match[2] < 100:
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