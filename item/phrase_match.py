import TitlePreprocessing as ttp
import romkan
import numpy
import re
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import phrases

def containsAlphabets(line):
	title = ttp.clean_title(line)
	if re.search('[a-zA-Z]' ,title):
		return title
	else:
		return 0

def findMatch(line):
	words = phrases.splitWord(line)
	alphabets = phrases.get_english_phrase(words)
	nonalphabets = phrases.generate_katakana_phrase(words)
	romanized = []
	for nonalphabet in nonalphabets:
		romanized.append(romkan.to_roma(nonalphabet))
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
				ans.append((alphabets[i],nonalphabets[j], line))
	return ans

def vectorize(token):
	replace_dict = {'': ['a', 'e', 'i', 'o', 'u'],
					'b': ['v'],
					'p': ['f', 'h'],
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

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
titles = input_data[:, 1]
data = sc.parallelize(titles, 80)
matches = data.map(containsAlphabets).filter(lambda _ : _).map(findMatch).flatMap(lambda _ : _ ).distinct().collect()

f = open("phrase_match.csv", "w")

for match in matches:
	string = ''
	for element in match:
		string = string + element.encode('utf8') + "\t"
	f.write(string+"\n")

f.close()