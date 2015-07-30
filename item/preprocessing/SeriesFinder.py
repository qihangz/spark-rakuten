# -*- coding:utf8 -*-
import redis
import re
from collections import defaultdict
from preprocessing import TitlePreprocessing as ttp
import time


def get_series_redis_connection():
    """
    """
    pwd = 'abcd$efghijkl\/mnopqr$stuvwxyz$NOW_iT_Is_bIT_triCky_2_BrEak_tHiS'
    r = redis.StrictRedis(
        host='bgeapcore101z.prod.jp.local', port=6381, password=pwd, db=8)
    return r


def get_series(tokens, r=get_series_redis_connection()):
    maker2series = {}
    hash_tokens = [hash(token) for token in tokens]
    htkn2tkn = dict((hash(token), token) for token in tokens)
    for htkn in hash_tokens:
        print htkn2tkn[htkn], r.scard(htkn)
        if r.scard(htkn) == 0:
            continue
        maker2series[htkn2tkn[htkn]] = []
        for series in r.smembers(htkn):
            if series in hash_tokens:
                maker2series[htkn2tkn[htkn]].append(htkn2tkn[htkn])
    return maker2series

if __name__ == '__main__':
    title = 'クリスタルガイザー CRYSTAL GEYSER ミネラルウォーター ナチュラルミネラルウォーター 水 オランチャ・シャスタ産 PET 24本入り 500ml 軟水 <BR><BR>送料無料'
    tkns = ttp.clean_title(title).split(' ')
    maker2series = get_series(tkns)
    print title
    for maker, series in maker2series.iteritems():
        for s in series:
            print maker, s
