import TitlePreprocessing as ttp
import romkan
import numpy
import re
from scipy.spatial import distance
from sklearn.preprocessing import normalize

def containsAlphabets(line):
	title = ttp.clean_title(line)
	if re.search('[a-zA-Z]' ,title):
		return title
	else:
		return 0

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
titles = input_data[:, 1]
data = sc.parallelize(titles, 80)
matches = data.map(containsAlphabets).filter(lambda _ : _).map(findMatch).flatMap(lambda _ : _ ).distinct().collect()
words = data.map(containsAlphabets).filter(lambda _ : _).map(countAlphabet).flatMap(lambda _ : _ ).distinct().collect()

def countAlphabet(line):
	alphabets = []
	words = filter(lambda word : word != '' and not re.search(r'\d', word),line.split(' '))
	for word in words:
		if re.search('[a-zA-Z]' ,word) and len(word) > 1:
			alphabets.append(word)
	return alphabets


def findMatch(line):
	alphabets = []
	nonalphabets = []
	romanized = []
	words = filter(lambda word : word != '' and not re.search(r'\d', word),line.split(' '))
	for word in words:
		if re.search('[a-zA-Z]' ,word) and len(word) > 1:
			alphabets.append(word)
		elif has_katakana(word):
			nonalphabets.append(word)
			romanized.append(romkan.to_roma(word))
	dim = (len(alphabets), len(romanized))
	similarity = numpy.zeros(dim)
	for i in range(len(alphabets)):
		for j in range(len(romanized)):
			similarity[i][j] = distance.euclidean(vectorize(alphabets[i]),vectorize(romanized[j]))
	ans = []
	if dim[1] > 0:
		for i in range(dim[0]):
			if min(similarity[i,:]) < 0.5:
				j = numpy.argmin(similarity[i,:])
				ans.append((alphabets[i],nonalphabets[j]))
	return ans

def vectorize(token):
	replace_dict = {# '': ['a', 'e', 'i', 'o', 'u'],
					'b': ['v'],
					'p': ['f'],
					'c': ['k'],
					'l': ['r'],
					's': ['z'],
					'g': ['j']
					}
	token = re.sub('[^a-z]', '', token.lower())
	for newtokens, tokens in replace_dict.iteritems():
		for oldtoken in tokens:
			token = token.replace(oldtoken, newtokens)
	array = numpy.zeros(26)
	for pos, char in enumerate(token):
		array[ord(char)-ord('a')] += (pos+1) 
	return normalize(array)

def has_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return any([not re.findall(ur'[\u30a0-\u30f0]', x) == [] for x in token])

def remove_double_spaces(title):
    ''' convert many spaces to single space'''
    title = re.sub(u' +', ' ', title)
    return re.sub(u'^ ', '', title)

def remove_bracket_contant(title):
	return remove_double_spaces(re.sub(r'\[[^]]*\]', '', title))

f = open("sim.csv", "w")

for row in similarity:
	string = ''
	for element in row:
		string = string + str(element) + "\t"
	f.write(string+"\n")

f.close()

f = open("match_with_vowel.csv", "w")

for match in matches:
	string = ''
	for element in match:
		string = string + element.encode('utf8') + "\t"
	f.write(string+"\n")

f.close()