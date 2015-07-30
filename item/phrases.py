import TitlePreprocessing as ttp
import numpy
import re

KATAKANA = ur'[\u30A0-\u30FF]'

def splitWord(title):
	words = ttp.clean_title(title).split(' ')
	words = filter(None, words)
	return words

def is_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall(KATAKANA, x) == [] for x in token])

def is_english(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall('[a-zA-Z]', x) == [] for x in token])

def get_katakana_phrase_pos(tokens):
	indicators = [is_katakana(token) for token in tokens]
	pos = []
	result = []
	for i in range(len(indicators)):
		if indicators[i] and pos == []:
			pos.append(i)
		if i == len(indicators) - 1:
			if indicators[i] and pos != []:
				result.append((pos.pop(), i))
		else:
			if indicators[i] and not indicators[i+1]:
				result.append((pos.pop(), i))
	return result

def get_english_phrase_pos(tokens):
	indicators = [is_english(token) for token in tokens]
	pos = []
	result = []
	for i in range(len(indicators)):
		if indicators[i] and pos == []:
			pos.append(i)
		if i == len(indicators) - 1:
			if indicators[i] and pos != []:
				result.append((pos.pop(), i))
		else:
			if indicators[i] and not indicators[i+1]:
				result.append((pos.pop(), i))
	return result

def get_english_phrase(tokens):
	english_phrase = []
	pos = get_english_phrase_pos(tokens)
	for i in range(len(pos)):
		english_phrase.append(" ".join(tokens[pos[i][0]:pos[i][1]+1]))
	return english_phrase

def generate_katakana_phrase(tokens):
	katakana_phrase = []
	pos = get_katakana_phrase_pos(tokens)
	for index in range(len(pos)):
		for i in range(pos[index][0], pos[index][1]):
			for j in range(i-1, pos[index][1]):
				katakana_phrase.append(" ".join(tokens[i:(j+1)]))
	return katakana_phrase