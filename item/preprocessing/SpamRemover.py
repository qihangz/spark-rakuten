#-*- coding: utf-8 -*-

''' this module import spam removal for Ichiba titles '''
import re
import os
import pickle
import sys
import logging
import traceback
from collections import defaultdict
from itertools import izip

import redis


def init_logging(name, log_level):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(process)d ' + name + ' : %(message)s')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def returnset():
    return defaultdict(set)

class NLPSpamRemover(object):

    def __init__(self, log_level='WARNING'):
        self.logger = init_logging('NLPSpamRemover', log_level)
        self.logger.info('NLPSpamRemover init...')

        self.redis_pwd = 'abcd$efghijkl\/mnopqr$stuvwxyz$NOW_iT_Is_bIT_triCky_2_BrEak_tHiS'
        self.redis_host = 'bgeapcore101z.prod.jp.local'
        self.redis_port = 6381
        self.redis_db = 11

        self.spam_map = None

    def get_spam_map(self):
        """
          loads the spam map for the spam remover
        """
        if os.path.exists('spam_map.pickle'):
            return pickle.loads(open('spam_map.pickle').read())
            
        spam_map = defaultdict(returnset)
        try:
            r = redis.StrictRedis(
                host=self.redis_host, port=self.redis_port, password=self.redis_pwd, db=self.redis_db)
            keys = [int(k) for k in r.keys()]
            for k in keys:
                try:
                    for tup_str in r.smembers(k):
                        # eval converts string to original type(tuple)
                        tup = eval(tup_str)
                        leng = tup[0]
                        h_remn = tup[1]
                        spam_map[k][leng].add(h_remn)
                except Exception as e:
                    self.logger.error('failed key: ' + k + ' ERROR: ' + traceback.format_exc())
        except Exception as e:
            self.logger.error(str(e))

        open('spam_map.pickle', 'w').write(pickle.dumps(spam_map))
        self.logger.info('Spam Map Size ' + str(len(spam_map)))
        return spam_map

    def remove_spam2(self, tokens, spam_map=None):
        """
        new as of 20150106
        """
        if spam_map == None:
            if not self.spam_map:
                self.spam_map = self.get_spam_map()
            spam_map = self.spam_map

        masked_tokens = [re.subn('[0-9]+','##',tok)[0] for tok in tokens]
        unmasked_tokens = [tok for tok in tokens]
        spam_flag = [0 for tkn in masked_tokens]
        for indx in range(len(masked_tokens)):
          tkn = masked_tokens[indx]
          if tkn in ['[',']','<','>']:
            spam_flag[indx] = 1
          h_tkn = hash(tkn)
          if h_tkn in spam_map:
            matched = False
            phrase_lengths = spam_map[h_tkn].keys()
            phrase_lengths.sort(reverse=True)
            for phrase_length in phrase_lengths:
              if phrase_length == 1:
                spam_flag[indx] = 1
              else:
                h_phrase = hash(tuple(masked_tokens[indx+1:indx+phrase_length]))
                if h_phrase in spam_map[h_tkn][phrase_length]:
                  spam_flag[indx:indx+phrase_length] = [1]*phrase_length
                  break
        clean_tokens =[unmasked_tokens[i] for i in range(len(spam_flag)) if spam_flag[i] == 0]
        return clean_tokens



    def remove_spam(self, tokens, spam_map=None):
        """
        tokens: list of unigram
        """
        if not spam_map:
            spam_map = self.spam_map

        # UPDATED version on 2015.01.06
        ret = []
        mtokens = [re.subn('[0-9]+', '##', tok)[0] for tok in tokens]
        t_length = len(mtokens)
        i = -1
        end = -1
        while i + 1 < t_length:
            i += 1
            h_tok = hash(mtokens[i])

            if mtokens[i] in ['[', ']', '<', '>']:
                continue

            if h_tok in spam_map:
                mat = False
                # check longest first
                for leng in sorted(spam_map[h_tok].keys(), reverse=True):
                    h_remains = spam_map[h_tok][leng]
                    if leng == 1:
                        h_phrase = hash(tuple([]))
                        #spam = mtokens[i].encode('utf8')
                    else:
                        h_phrase = hash(tuple(mtokens[i + 1:i + leng]))
                        #spam = ' '.join(mtokens[i:i+leng]).encode('utf8')
                    if h_phrase in h_remains:
                        mat = True
                        if i + leng - 1 > end:
                            end = i + leng - 1  # TODO DEBUG
                        break
                if not mat:
                    if end >= i:  # TODO DEBUG
                        i = end
                        end = i
                    else:
                        ret.append(tokens[i])
            else:
                if i < t_length:
                    ret.append(tokens[i])  # TODO

        return ret

class EntropySpamRemover(object):

    def __init__(self, log_level='WARNING'):
        self.logger = init_logging('EntropySpam', log_level)
        self.logger.info('Entropy SpamRemover init... ')

        self.redis_pwd = 'abcd$efghijkl\/mnopqr$stuvwxyz$NOW_iT_Is_bIT_triCky_2_BrEak_tHiS'
        self.redis_host = 'bgeapcore101z.prod.jp.local'
        self.redis_port = 6381
        self.redis_db = 10

        self.spam_map = self.get_spam_map()
        self.logger.info('Spam Map Size ' + str(len(self.spam_map)))

    def get_spam_map(self):
        """
            loads and returns the spam map for the entropy remover from redis
        """
        r = redis.StrictRedis(
            host=self.redis_host, port=self.redis_port, password=self.redis_pwd, db=self.redis_db)
        keys = [k for k in r.keys()]
        spam_map = dict()
        for k in keys:
            spam_map[k] = r.smembers(k)
        return spam_map

    def remove_spam(self, tokens, spam_map=None):
        """
            a list of tokens(order is important) and returns a list of tokens with removed spam.
        """
        if not spam_map:
            spam_map = self.spam_map
        spams = set()
        tokens = [t for t in tokens if not t in ('[', ']')]
        token_list = list(tokens)
        masked_tokens = [re.sub("\d+", "##", token) for token in token_list]
        marked_tokens = [0 for token in token_list]
        n_grams = 3

        for indx in range(len(token_list)):
            masked_token = str(hash(masked_tokens[indx]))

            if not masked_token in spam_map:
                continue

            spam_endings = spam_map[masked_token]
            if indx + 2 < len(masked_tokens) and str(hash((masked_tokens[indx + 1], masked_tokens[indx + 2]))) in spam_endings:
                marked_tokens[indx] = 1
                marked_tokens[indx + 1] = 1
                marked_tokens[indx + 2] = 1
            elif indx + 1 < len(masked_tokens) and str(hash(masked_tokens[indx + 1])) in spam_endings:
                marked_tokens[indx] = 1
                marked_tokens[indx + 1] = 1
            elif '' in spam_endings:
                marked_tokens[indx] = 1

        tokens = [t for m, t in izip(marked_tokens, tokens) if not m == 1]
        return tokens


def removeSpam_benchmark(spam_remover, genreID, logger):
    import CassConn.CassConn as CC
    import TitlePreprocessing as ttp
    stime = time.time()

    cc = CC.CassConn(env='INS')

    batchSize = 500
    rowKey = "G_" + str(genreID)
    items = [x[0] for x in cc.cfProductMaster.xget(rowKey)]

    logger.debug(
        ' '.join(['genre ', genreID, ' received ', str(len(items)), ' from G_ index']))
    if len(items) > 60000:
        print 'skip big genre'
        return

    for indx in range(0, len(items), batchSize):
        pdata = cc.cfProductMaster.multiget(
            items[indx:indx + batchSize], columns=['V1'])
        for prodKey, pvals in pdata.iteritems():
            try:
                parts = pvals['V1'].split('\t')
                title = parts[8]

                # tokenize and remove spam
                tkns = ttp.clean_title(title).split(' ')  # .decode('utf8'))

                ptitle1 = ' '.join(tkns)
                ret = spam_remover.remove_spam(tkns)
                ptitle2 = ' '.join(ret)

                #logger.debug(' '.join([prodKey,'in :',ptitle1.encode('utf8')]))
                #logger.debug(' '.join([prodKey,'out:',ptitle2.encode('utf8')]))
                #logger.debug(' -------------------------- ')
            except Exception, e:
                print traceback.format_exc()
                sys.exit()
                continue

    logger.debug('time 4 remove spam:' + str(int(time.time() - stime)))


def main(spam_remover, logger):
    import TitlePreprocessing as ttp
    import time

    logger.debug('#keys:' + str(len(spam_remover.spam_map)))
    logger.debug('key1:' + str(type(spam_remover.spam_map.keys()
                                    [0])) + ' ' + str(spam_remover.spam_map.keys()[0]))
    logger.debug(
        'val1:' + str(spam_remover.spam_map[spam_remover.spam_map.keys()[0]]))
    # TODO move these outputs ito each spam removers class and change it according to datatype
    # nlp_spam map is hash->dict and entropy_spam is hash->set
    try:
        for k, vals in spam_remover.spam_map[spam_remover.spam_map.keys()[0]].iteritems():
            logger.debug('elem k: ' + str(type(k)) + ' ' + str(k))
            for v in vals:
                logger.debug('elem v:' + str(type(v)) + ' ' + str(v))
    except:
        pass
    stime = time.time()

    # test1
    logger.debug('')
    logger.debug('test1 --------------------------------')
    title = '【エントリーでポイント10倍】セクシーワンピ/ワンピース/レディースファッション/海外人気モデル【10500円以上で送料無料】'
    logger.debug('title : ' + str(type(title)) + ' ' + title)
    tkns = ttp.clean_title(title).split(' ')
    ptitle1 = ' '.join(tkns)

    ret = spam_remover.remove_spam(tkns)
    ptitle2 = ' '.join(ret)
    if ptitle1 != ptitle2:
        logger.debug('in :' + ptitle1.encode('utf8'))
        logger.debug('out:' + ptitle2.encode('utf8'))
    #"""

    #"""
    # test2
    if len(sys.argv) > 1:
        logger.debug('')
        logger.debug('test2 --------------------------------')
        # removeSpam_test('507745')
        removeSpam_benchmark(spam_remover, sys.argv[1], logger)
    #"""

if __name__ == '__main__':
    import time
    logger = init_logging('SpamRemoverMainLogger', 'DEBUG')
    stime = time.time()
    spam_remover = NLPSpamRemover(log_level='DEBUG')
    logger.debug('time 4 load spam: ' + str(round(time.time() - stime, 2)))
    main(spam_remover, logger)

    stime = time.time()
    spam_remover = EntropySpamRemover(log_level='DEBUG')
    logger.debug('time 4 load spam: ' + str(round(time.time() - stime, 2)))
    main(spam_remover, logger)
