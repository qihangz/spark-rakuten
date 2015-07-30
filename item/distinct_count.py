import TitlePreprocessing as ttp
import numpy
import re

HIRAGANA = ur'[\u3041-\u309F]'
KATAKANA = ur'[\u30A0-\u30FF]'
KANJI = ur'[\u4E00-\u9FAF]'

def splitWord(title):
	words = ttp.clean_title(title).split(' ')
	words = [word for word in words if not word in ['', '[', ']']]
	return words

input_data = numpy.array(numpy.loadtxt("res0.csv", delimiter=",", dtype="string"))
input_data = input_data[1:, -1]
data = sc.parallelize(input_data, 80)
words = data.map(splitWord).flatMap(lambda words: words).distinct().cache()
total_count = words.count()
katakana_count = words.filter(lambda word : is_katakana(word)).count()
hiragana_kanji_count = words.filter(lambda word : is_hiragana_or_kanji(word)).count()
english_count = words.filter(lambda word : is_english(word)).count()
number_count = words.filter(lambda word : has_number(word)).count()
others_count = words.filter(lambda word : not (is_katakana(word) or is_hiragana_or_kanji(word) or is_english(word) or has_number(word))).count()

def is_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall(KATAKANA, x) == [] for x in token])

def is_hiragana_or_kanji(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not (re.findall(HIRAGANA, x) == [] and re.findall(KANJI, x) == []) for x in token])

def is_english(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall('[a-zA-Z]', x) == [] for x in token])

def has_number(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return any([not re.findall('[0-9]', x) == [] for x in token])