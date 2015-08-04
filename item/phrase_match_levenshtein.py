import TitlePreprocessing as ttp
import romkan
import numpy
import re
import phrases
import Levenshtein

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
		romanized.append(to_romaji(nonalphabet))
	dim = (len(alphabets), len(romanized))
	similarity = numpy.zeros(dim)
	for i in range(len(alphabets)):
		for j in range(len(romanized)):
			similarity[i][j] = Levenshtein.distance(alphabets[i],romanized[j]) / (float(min(len(alphabets[i]), len(romanized[j])))+1)
	ans = []
	# if dim[1] > 0:
	# 	for i in range(dim[0]):
	# 		#if min(similarity[i,:]) < 0.5:
	# 		j = numpy.argmin(similarity[i,:])
	# 		ans.append((alphabets[i],nonalphabets[j], similarity[i][j], line))
	# return ans
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
	replace_dict = {#'': ['a', 'e', 'i', 'o', 'u'],
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

f = open("phrase_match_Levenshtein.csv", "w")

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