#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
    This module contains methods for cleaning and normalizing Ichiba item titles.
"""
import re
import sys
import unicodedata as ud

import MeCab

QUANTITIES = u"cmlgk時週万円日倍本枚個回粒組点茶黒赤色灯種袋入円号缶ケ%吋段巻幅台役個"

NUMERIC = u'0123456789'
ALPHANUM = u'abcdefghijklmnopqrstuvwxyz' + \
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SYMBOLS = u'!\"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'
ROMAJI = set(NUMERIC + ALPHANUM + SYMBOLS)

HIRAGANA = ur'[\u3041-\u309F]'
KATAKANA = ur'[\u30A0-\u30F9]'
KANJI = ur'[\u4e00-\u9faf]'
JAP_RANGE = ur'[\u3000-\u9faf]'
ASCII = ur'[\u0021-\u007E]'
FULL_WIDTH_HALF_WIDTH = ur'[\uff00-\uffef]'
NONJAP = ur'[\u0021-\u2FFF]'


try:
    MECAB = MeCab.Tagger('-Owakati --dicdir=/a/ins-bpiop101/gv0/pkmed/pcatuser/mecab_dict/')
    print 'Using mecab with RIT dictionary'
except:
    try:
        MECAB =MeCab.Tagger('-Owakati')
        print 'Using mecab with IPADIC dictionary'
    except:
        print 'Error: mecab not found,seems to be not installed'


def replace_symbols(title, replace_dict=None):
    '''
        Replaces occurrances of replaceDict.
        The key is the token things in the value will be replaced with!
    '''
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


def keep_string_if_between_digits(title, string):
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


def delete_symbols(title, delete_list=None):
    '''
        Do not delete any of these chars if they appear between digits
        Otherwise it will be hard to extract quantities and numbers
        If it is between letter an digit it is fine though, eg in serial numbers
    '''
    if not delete_list:
        delimiter = set([u'.', u'−', u'-', u'/'])
        delete_tokens = set(['[ br ]'])

    for token in delete_tokens:
        title = title.replace(token, ' ')

    title = title.replace('\'', '')
    SYMBOLS = u'!\"#&()/*,;<=>?@\\^`{|}~'
    for symbol in SYMBOLS:
        title = title.replace(symbol, '  ')

    title = title.replace('_', '')
    title = title.replace('-', '')

    # if ':' or '.' in title:
    #    title_tokens = []
    #    for cnt, token in enumerate(title):
    #        if token in [':', '.']:
    #            if cnt>0 and cnt+1<len(title) and title[cnt-1].isdigit() and title[cnt+1].isdigit():
    #                    title_tokens.append(token)
    #            else:
    #                title_tokens.append(' ')
    #        else:
    #            title_tokens.append(token)
    #    title = ''.join(title_tokens)
    title = keep_string_if_between_digits(title, ':')
    title = keep_string_if_between_digits(title, '.')

    title = re.sub(ur'[\u0180-\u303f]', '  ', title)
    title = re.sub(ur'[\ua000-\uffff]', '  ', title)
    title = re.sub(ur'\[ [a-zA-Z0-9] \]', '', title)

    return title


def normalize_unicode(title):
    '''Normalizes a unicode string  using NFKC'''
    title = ud.normalize('NFKC', title)
    return title


def lowercase(title):
    '''lowercases the title'''
    return title.lower()


def remove_spaces_between_digits(title):
    ''' sometimes mecab will split digits, this method combines digits '''
    # needs to be done twice because with a number like
    # 1 2 3 4  re only matches to 12 34 the first go
    title = re.sub(ur'(\d)[^\S\r\n]+(\d)', ur'\g<1>\g<2>', title)
    return re.sub(ur'(\d)[^\S\r\n]+(\d)', ur'\g<1>\g<2>', title)


def remove_spaces_between_romaji(title):
    ''' this method removes spaces between romaji characters as they are likely a serial number'''
    # needs to be done twice because with a number like
    # a b c d  re only matches to ab cd the first go
    title = re.sub(ur'([a-z])[^\S\r\n]+([a-z]) ', ur'\g<1>\g<2> ', title)
    return re.sub(ur'([a-z])[^\S\r\n]+([a-z]) ', ur'\g<1>\g<2> ', title)


def space_digits_and_string(title):
    ''' this method disconnects digits from strings'''
    title = re.sub(ur' (\d+)(\D)', ur' \g<1> \g<2>', title)
    title = re.sub(ur'^(\d+)(\D)', ur'\g<1> \g<2>', title)

    title = re.sub(ur'(\D)(\d+) ', ur'\g<1> \g<2> ', title)
    title = re.sub(ur'(\D)(\d+)$', ur'\g<1> \g<2>', title)

    return re.sub(ur'(\D)(\d+)(\D)', ur'\g<1> \g<2> \g<3>', title)


def connect_dangling_numbers(title):
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


def connect_rational_numbers(title):
    ''' try to keep rational numbers connected '''
    # convert 1 . 5 to 1.5
    return re.sub(ur'(\d+)[^\S\r\n]*?\.[^\S\r\n]*?(\d+)', ur'\g<1>.\g<2>', title)


def remove_double_spaces(title):
    ''' convert many spaces to single space'''
    title = re.sub(u' +', ' ', title)
    return re.sub(u'^ ', '', title)


def space_brackets(title):
    ''' put spaces around brackets '''
    title = title.replace('[', ' [ ')
    title = title.replace(']', ' ] ')
    return title


def concatenate_single_letters(title):
    ''' connect dangling ascii characters '''
    title = re.sub(r" ([a-z]) ([a-z])", r"\g<1>\g<2>", title)
    return re.sub(r"([a-z]) ([a-z]) ", r"\g<1>\g<2>", title)

# speedup cache query
ISASCII_CACHE = {}


def is_ascii(word):
    ''' return whether the input string is ascii or not '''
    if word in ISASCII_CACHE:
        return ISASCII_CACHE[word]
    try:
        word.encode('ascii')
        ISASCII_CACHE[word] = True
        return True
    except:
        ISASCII_CACHE[word] = False
        return False


def repair_connectors(title):
    '''
        Repairs the - between japanese characters, some merchants use -
        instead of ー so テープ becomes　テ-プ
    '''
    title = re.sub(ur'([a-zA-Z0-9])ー([a-zA-Z0-9])',u'\g<1>-\g<2>', title)
    title = re.sub(ur'([\u3040-\u30F0])-([\u3040-\u30F0])',u'\g<1>ー\g<2>', title)

    return title


def is_katakana(token):
    ''' return whether a token is only katakana '''
    if not type(token) == type(u''):
        token = unicode(token, 'utf-8')
    return all([not re.findall(ur'[\u30a0-\u30f0]', x) == [] for x in token])

def space_romaji_and_kana(title):
    ''' insert a space between romaji and kana '''

    title = re.sub(ur'('+NONJAP+')('+JAP_RANGE+')',u'\g<1> \g<2>', title)
    title = re.sub(ur'('+JAP_RANGE+')('+NONJAP+')',u'\g<1> \g<2>', title)

    title = re.sub(ur'('+KATAKANA+')('+HIRAGANA+')',u'\g<1> \g<2>', title)
    title = re.sub(ur'('+HIRAGANA+')('+KATAKANA+')',u'\g<1> \g<2>', title)

    title = re.sub(ur'('+KATAKANA+')('+KANJI+')',u'\g<1> \g<2>', title)
    title = re.sub(ur'('+KANJI+')('+KATAKANA+')',u'\g<1> \g<2>', title)

    return title

def insert_space_after_percent(title):
    ''' insert a space after percent signs '''
    return title.replace("%", "% ")

def tokenize(title, mecab=MECAB):
    ''' tokenize title using mecab '''
    newdata = []
    for sentence in title.split('\n'):
        newsentence = []
        for token in sentence.split(' '):
            if is_ascii(token):
                newsentence.append(token)
            else:
                newsentence.append(
                    unicode(mecab.parse(token.encode('utf-8')).rstrip(), 'utf-8'))
        newdata.append(" ".join(newsentence))
    return "\n".join(newdata)

def clean_title(title, mecab=MECAB):
    ''' clean a title using several methods of normalizing, tokenizing and cleaning '''
    if not isinstance(title, unicode):
        title = unicode(title, 'utf-8')

    original_data = title
    title = normalize_unicode(title)
    title = lowercase(title)
    title = repair_connectors(title)
    title = replace_symbols(title)
    title = space_brackets(title)
    title = delete_symbols(title)
    # if the title contained only a - for example it will be empty so we just
    # return the orginal
    if len(title) < 2:
        return original_data
   # title = concatenate_single_letters(title)

    title = space_romaji_and_kana(title)
    title = remove_double_spaces(title)

    title = tokenize(title, mecab)

    title = insert_space_after_percent(title)

    title = connect_dangling_numbers(title)
    title = connect_rational_numbers(title)

    return title


def clean_titles(titles, mecab=MECAB):
    ''' cleans a list of titles using several methods of normalizing, tokenizing and cleaning '''
    for i, title in enumerate(titles):
        titles[i] = clean_title(title, mecab)
    return titles


def benchmark(titles, method):
    import time
    n_iter = 3
    unique_tokens = set()
    print 'Benchmarking'+str(len(titles))
    
    rqmc = MeCab.Tagger('-Owakati --dicdir=/a/ins-bpiop101/gv0/pkmed/pcatuser/mecab_dict/')
    ormc = MeCab.Tagger('-Owakati')

    for i in xrange(n_iter):
        stime = time.time()

        diff = 0
        tokens = []
        rqtokens=[]

        for title in titles:
            
            ctitle = method(title, ormc)
            print 
            print title
            print ctitle
            
            #method(title).split(' ')
        etime = time.time()
        dt = etime - stime
        print ' '.join([str(x) for x in [i, len(titles), diff, 100.0*diff/len(titles), dt, 1.0 * len(titles) / dt]])


def main():
    ''' cleans titles read from a file'''
    #if len(sys.argv) != 2:
    #    print "Usage: filename"
    #    sys.exit()

    #print clean_title('エアークリーナーテ-プ快適空房ＫＫＡー４０')
    #print clean_title('プラソニエ　エピルーズEX 2')
    #print clean_title('2 EX プラソニエ　エピルーズEX 2')
    #print clean_title('arobo/アロボ 空気洗浄機 CLV-800 WH')
    for token in clean_title('ビーレーザー[男性用]forM LA-1A<br>　[家庭用レーザー脱毛器・美顔器]<br>　【送料・代引き手数料無料】').split(' '):
        print token


    #print clean_title('エアー 2134 クリーナーテ-プ asd快適空房ＫＫＡー４０')
    #print clean_title('★【送料無料】ヤーマンレーザー脱毛器 フォトレーザーシグマA-5【K】【TC】【ギフト/贈り物】【楽ギフ_包装】【RCP】【取寄品】【マラソン201312_送料無料】10P13Dec13')
    sys.exit()


    from database import CassConn as CC
    cc = CC.CassConn(env='PROD')
    iids = [x[0] for x in cc.cfProductMaster.xget('G_212571')]
    titles = []
    for iid, data in cc.cfProductMaster.multiget(iids, columns=['title']).iteritems():
        title = data['title']#data['V1'].split('\t')[9]
        titles.append(title)

    benchmark(titles, clean_title)


        
    assert False

    import gzip
    titles = []
    with gzip.open(sys.argv[1], 'r') as file_handle:
        for cnt, line in enumerate(file_handle):
            titles.append(line.strip().split('\t')[-1])
            if cnt > 20000:
                break
    # for cnt, sentence in enumerate(titles.split('\n')[1:]):
    #    title = sentence.split('\t')[-1]
    #    cleaned_title = sr.remove_spam(clean_title(title))
        # print title
        # print 'new',cleaned_title
        # print
        # if cnt>20:
#            break
    # print titles[1]
    # print tokenize_remove_pos(titles[1])
    benchmark(titles, clean_title)

if __name__ == "__main__":
    main()
