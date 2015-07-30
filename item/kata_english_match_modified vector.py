import TitlePreprocessing as ttp
import romkan
import numpy
import re
from scipy.spatial import distance
from sklearn.preprocessing import normalize

KATAKANA = ur'[\u30A0-\u30FF]'

def containsAlphabets(line):
	title = ttp.clean_title(line)
	if re.search('[a-zA-Z]' ,title):
		return title
	else:
		return 0

def findMatch(line):
	alphabets = []
	nonalphabets = []
	romanized = []
	words = filter(lambda word : word != '' and not re.search(r'\d', word),line.split(' '))
	for word in words:
		if re.search('[a-zA-Z]' ,word) and len(word) > 1:
			alphabets.append(word)
		elif is_katakana(word):
			nonalphabets.append(word)
			romanized.append(romkan.to_roma(word))
	dim = (len(alphabets), len(romanized))
	similarity = numpy.zeros(dim)		
	for i in range(len(alphabets)):
		for j in range(len(romanized)):
			alphabet_len = len(alphabets[i])
			romanized_len = len(romanized[j])
			max_len = max(alphabet_len, romanized_len)
			similarity[i][j] = numpy.linalg.norm(vectorize(alphabets[i], max_len)-vectorize(romanized[j], max_len))
			# similarity[i][j] = distance.euclidean(vectorize(alphabets[i], max_len),vectorize(romanized[j], max_len))
	ans = []
	for i in range(min(dim[0], dim[1])):
		row_index = similarity.argmin() / similarity.shape[1]
		col_index = similarity.argmin() % similarity.shape[1]
		ans.append((alphabets[row_index],nonalphabets[col_index], similarity[row_index,col_index],line))
		del alphabets[row_index]
		del nonalphabets[col_index]
		similarity = numpy.delete(similarity, row_index, 0)
		similarity = numpy.delete(similarity, col_index, 1)
	return ans

def vectorize(token, length):
	replace_dict = {#'': ['a', 'e', 'i', 'o', 'u'],
					'': ['u'],
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
	array = numpy.zeros((26, length))
	for pos, char in enumerate(token):
		array[ord(char)-ord('a'), pos:] += 1
	return normalize(array)

def is_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall(KATAKANA, x) == [] for x in token])


input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
titles = input_data[:, 1]
data = sc.parallelize(titles, 80)
matches = data.map(containsAlphabets).filter(lambda _ : _).map(findMatch).flatMap(lambda _ : _ ).distinct().collect()

f = open("match_modified_vector.csv", "w")

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
