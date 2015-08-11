import TitlePreprocessing as ttp
import romkan
import numpy
import re
import Levenshtein

def containsAlphabets(line):
	title = ttp.clean_title(line)
	if re.search('[a-zA-Z]' ,title):
		return title
	else:
		return 0

def findMatch(line):
	words = filter(lambda word : word != '' and not re.search(r'\d', word),line.split(' '))
	alphabets = []
	nonalphabets = []
	romanized = []
	for word in words:
		if re.search('[a-zA-Z]' ,word) and len(word) > 1:
			alphabets.append(word)
		elif is_katakana(word):
			nonalphabets.append(word)
			romanized.append(to_romaji(word))
	dim = (len(alphabets), len(romanized))
	similarity = numpy.zeros(dim)
	for i in range(len(alphabets)):
		for j in range(len(romanized)):
			similarity[i][j] = Levenshtein.distance(alphabets[i],romanized[j]) / (float(len(alphabets[i])+len(romanized[j]))/2)
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

def to_romaji(token):
	replace_dict = {'': ['a', 'e', 'i', 'o', 'u'],
					'b': ['v'],
					'p': ['f', 'h'],
					'c': ['k'],
					'l': ['r'],
					's': ['z'],
					'g': ['j']
					}
	token = re.sub('[^a-z]', '', romkan.to_roma(token).lower())
	for newtokens, tokens in replace_dict.iteritems():
		for oldtoken in tokens:
			token = token.replace(oldtoken, newtokens)
	return token

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, [1, -1]]
titles = input_data[:, 1]
data = sc.parallelize(titles, 80)
matches = data.map(containsAlphabets).filter(lambda _ : _).map(findMatch).flatMap(lambda _ : _ ).distinct().collect()

f = open("kata_alphabet_levenshtein.csv", "w")

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