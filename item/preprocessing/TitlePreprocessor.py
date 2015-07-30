#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
    This module contains methods for cleaning and normalizing Ichiba item titles.
"""
import MeCab
import re
import unicodedata as ud
__author__ = 'hsperr'

QUANTITIES = u"cmlgk時週万円日倍本枚個回粒組点茶黒赤色灯種袋入円号缶ケ%吋段巻幅台役個"

NUMERIC = u'0123456789'
ALPHANUM = (u'abcdefghijklmnopqrstuvwxyz'
            u'ABCDEFGHIJKLMNOPQRSTUVWXYZ')

SYMBOLS = u'!\"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'
ROMAJI = set(NUMERIC + ALPHANUM + SYMBOLS)

HIRAGANA = ur'[\u3041-\u309F]'
KATAKANA = ur'[\u30A0-\u30F9]'
KANJI = ur'[\u4e00-\u9faf]'
JAP_RANGE = ur'[\u3000-\u9faf]'
ASCII = ur'[\u0021-\u007E]'
FULL_WIDTH_HALF_WIDTH = ur'[\uff00-\uffef]'
NONJAP = ur'[\u0021-\u2FFF]'

class TitlePreprocessor(object):
    def __init__(self, dictionary_path=None):
        if dictionary_path is None:
            self.tokenizer = MeCab.Tagger('-Owakati')
        else:
            self.tokenizer = MeCab.Tagger('-Owakati --dicdir='+dictionary_path)

    def replace_symbols(self, title, replace_dict=None):
        """
        Replaces occurrances of replaceDict.
        The key is the token things in the value will be replaced with!
        """
        if not replace_dict:
            replace_dict = {'[': [u'【', u'〔', u'［', u'｛', u'｟', u'《', u'〈', u'(', u'<'],
                            ']': [u'】', u'〕', u'］', u'｝', u'｠', u'》', u'〉', u')', u'>'],
                            u'-': [u'−'],
                            u'x': [u'＊', u'×', u'*'],
                            u' ': [u'"', u'・', u'･', u'#'],
                            u'': [u'°'],
                            u'~': [u'〜']
                            }
        for newtokens, tokens in replace_dict.iteritems():
            for oldtoken in tokens:
                title = title.replace(oldtoken, newtokens)
        return title

    def keep_string_if_between_digits(self, title, string):
        if string in title:
            stitle = title.split(string)
            stitle = [x for x in stitle if len(x)]
            if not stitle:
                return title
            title = stitle[0]
            for part in stitle[1:]:
                if not len(part):
                    continue
                t_d = title[-1].isdigit()
                p_d = part[0].isdigit()
                if t_d and p_d:
                    title = string.join([title, part])
                elif t_d or p_d:
                    title = ''.join([title, part])
                else:
                    title = ' '.join([title, part])
        return title

    def delete_symbols(self, title, delete_list=None):
        '''
        Do not delete any of these chars if they appear between digits
        Otherwise it will be hard to extract quantities and numbers
        If it is between letter an digit it is fine though, eg in serial numbers
        '''
        if not delete_list:
            delimiter = {u'.', u'−', u'-', u'/'}
            delete_tokens = {'[ br ]'}

        for token in delete_tokens:
            title = title.replace(token, ' ')

        title = title.replace('\'', '')
        SYMBOLS = u'!\"#&()/*,;<=>?@\\^`{|}~'
        for symbol in SYMBOLS:
            title = title.replace(symbol, '  ')

        title = title.replace('_', '')
        title = title.replace('-', '')

        title = self.keep_string_if_between_digits(title, ':')
        title = self.keep_string_if_between_digits(title, '.')

        title = re.sub(ur'[\u0180-\u303f]', '  ', title)
        title = re.sub(ur'[\ua000-\uffff]', '  ', title)
        title = re.sub(ur'\[ [a-zA-Z0-9] \]', '', title)

        return title


    def normalize_unicode(self, title):
        '''Normalizes a unicode string  using NFKC'''
        title = ud.normalize('NFKC', title)
        return title


    def lowercase(self, title):
        '''lowercases the title'''
        return title.lower()


    def remove_spaces_between_digits(self, title):
        ''' sometimes mecab will split digits, this method combines digits '''
        # needs to be done twice because with a number like
        # 1 2 3 4  re only matches to 12 34 the first go
        title = re.sub(ur'(\d)[^\S\r\n]+(\d)', ur'\g<1>\g<2>', title)
        return re.sub(ur'(\d)[^\S\r\n]+(\d)', ur'\g<1>\g<2>', title)


    def remove_spaces_between_romaji(self, title):
        ''' this method removes spaces between romaji characters as they are likely a serial number'''
        # needs to be done twice because with a number like
        # a b c d  re only matches to ab cd the first go
        title = re.sub(ur'([a-z])[^\S\r\n]+([a-z]) ', ur'\g<1>\g<2> ', title)
        return re.sub(ur'([a-z])[^\S\r\n]+([a-z]) ', ur'\g<1>\g<2> ', title)


    def space_digits_and_string(self, title):
        ''' this method disconnects digits from strings'''
        title = re.sub(ur' (\d+)(\D)', ur' \g<1> \g<2>', title)
        title = re.sub(ur'^(\d+)(\D)', ur'\g<1> \g<2>', title)

        title = re.sub(ur'(\D)(\d+) ', ur'\g<1> \g<2> ', title)
        title = re.sub(ur'(\D)(\d+)$', ur'\g<1> \g<2>', title)

        return re.sub(ur'(\D)(\d+)(\D)', ur'\g<1> \g<2> \g<3>', title)


    def connect_dangling_numbers(self, title):
        ''' connect numbers to quantities like 2 g to 2g'''
        title = re.sub(ur' (\d+)[^\S\r\n]+([' + QUANTITIES + '])', ur' \g<1>\g<2>', title)
        title = re.sub(ur' (\d+) *x *(\d+)', ur' \g<1>x\g<2>', title)
        title = re.sub(ur'(\d+.x\d+) (.) ', ur'\g<1>\g<2> ', title)
        title = re.sub(ur'(\d+..x\d+) (.) ', ur'\g<1>\g<2> ', title)
        title = re.sub(ur'(\d+.x\d+) (.)$', ur'\g<1>\g<2>', title)
        title = re.sub(ur'(\d+..x\d+) (.)$', ur'\g<1>\g<2>', title)

        ''' connect the rest '''

        #title = re.sub(ur'([^ \[\]\d]*) (\d+) ([^ \[\]\d]*)', ur'\g<1> \g<1>\g<2> \g<2>\g<3> \g<3>', title)
        #title = re.sub(ur'([^ \[\]\d]*) (\d+)$', ur'\g<1>\g<2> \g<2>', title)
        #title = re.sub(ur'^(\d+) ([^ \[\]\d]*)', ur'\g<1>\g<2> \g<2>', title)

        # if token ends with number insert concat token
        #title = re.sub(ur'([^ ]*\d+) ([^\[\]]+)', ur'\g<1> \g<1>\g<2> \g<2>', title)
        return title


    def connect_rational_numbers(self, title):
        ''' try to keep rational numbers connected '''
        # convert 1 . 5 to 1.5
        return re.sub(ur'(\d+)[^\S\r\n]*?\.[^\S\r\n]*?(\d+)', ur'\g<1>.\g<2>', title)

    def remove_double_spaces(self, title):
        ''' convert many spaces to single space'''
        title = re.sub(u' +', ' ', title)
        return re.sub(u'^ ', '', title)

    def space_brackets(self, title):
        ''' put spaces around brackets '''
        title = title.replace('[', ' [ ')
        title = title.replace(']', ' ] ')
        return title

    def concatenate_single_letters(self, title):
        ''' connect dangling ascii characters '''
        title = re.sub(r" ([a-z]) ([a-z])", r"\g<1>\g<2>", title)
        return re.sub(r"([a-z]) ([a-z]) ", r"\g<1>\g<2>", title)

    def is_ascii(self, word):
        ''' return whether the input string is ascii or not '''
        try:
            word.decode('ascii')
            return True
        except:
            return False

    def repair_connectors(self, title):
        '''
        Repairs the - between japanese characters, some merchants use -
        instead of ー so テープ becomes　テ-プ
        '''
        title = re.sub(ur'([a-zA-Z0-9])ー([a-zA-Z0-9])',u'\g<1>-\g<2>', title)
        title = re.sub(ur'([\u3040-\u30F0])-([\u3040-\u30F0])',u'\g<1>ー\g<2>', title)

        return title

    def is_katakana(self, token):
        ''' return whether a token is only katakana '''
        if not type(token) == type(u''):
            token = unicode(token, 'utf-8')
        return all([not re.findall(ur'[\u30a0-\u30f0]', x) == [] for x in token])

    def space_romaji_and_kana(self, title):
        ''' insert a space between romaji and kana '''

        title = re.sub(ur'('+NONJAP+')('+JAP_RANGE+')',u'\g<1> \g<2>', title)
        title = re.sub(ur'('+JAP_RANGE+')('+NONJAP+')',u'\g<1> \g<2>', title)

        title = re.sub(ur'('+KATAKANA+')('+HIRAGANA+')',u'\g<1> \g<2>', title)
        title = re.sub(ur'('+HIRAGANA+')('+KATAKANA+')',u'\g<1> \g<2>', title)

        title = re.sub(ur'('+KATAKANA+')('+KANJI+')',u'\g<1> \g<2>', title)
        title = re.sub(ur'('+KANJI+')('+KATAKANA+')',u'\g<1> \g<2>', title)

        return title

    def insert_space_after_percent(self, title):
        ''' insert a space after percent signs '''
        return title.replace("%", "% ")

    def tokenize(self, title):
        ''' tokenize title using mecab '''
        newdata = []
        for sentence in title.split('\n'):
            newsentence = []
            for token in sentence.split(' '):
                if self.is_ascii(token):
                    newsentence.append(token)
                else:
                    newsentence.append(
                        unicode(self.tokenizer.parse(token.encode('utf-8')).rstrip(), 'utf-8'))
            newdata.append(" ".join(newsentence))
        return "\n".join(newdata)

    def clean_title(self, title):
        ''' clean a title using several methods of normalizing, tokenizing and cleaning '''
        if not isinstance(title, unicode):
            title = unicode(title, 'utf-8')

        original_data = title
        title = self.normalize_unicode(title)
        title = self.lowercase(title)
        title = self.repair_connectors(title)
        title = self.replace_symbols(title)
        title = self.space_brackets(title)
        title = self.delete_symbols(title)
        # if the title contained only a - for example it will be empty so we just
        # return the orginal
        if len(title) < 2:
            return original_data
       # title = concatenate_single_letters(title)

        title = self.space_romaji_and_kana(title)
        title = self.remove_double_spaces(title)

        title = self.tokenize(title)

        title = self.insert_space_after_percent(title)

        title = self.connect_dangling_numbers(title)
        title = self.connect_rational_numbers(title)

        return title


    def clean_titles(self, titles):
        ''' cleans a list of titles using several methods of normalizing, tokenizing and cleaning '''
        for i, title in enumerate(titles):
            titles[i] = self.clean_title(title)
        return titles


if __name__ == '__main__':
    tokenizer = TitlePreprocessor()
    for token in tokenizer.clean_title('ビーレーザー[男性用]forM LA-1A<br>　[家庭用レーザー脱毛器・美顔器]<br>　【送料・代引き手数料無料】').split(' '):
        print token
