# -*- coding:utf8 -*-
"""
    This module will do everything related to aliases, where a alias might be a romaji katakana pair of words
"""
import redis
from collections import defaultdict
from preprocessing import TitlePreprocessing as ttp

ALIAS_MAP = defaultdict(set)


def get_alias_redis_connection():
    """
        returns the default redis connection
    """
    pwd = r'abcd$efghijkl\/mnopqr$stuvwxyz$NOW_iT_Is_bIT_triCky_2_BrEak_tHiS'
    r = redis.StrictRedis(
        host='bgeapcore101z.prod.jp.local', port=6381, password=pwd, db=7)
    return r


def get_missing_aliases(tokens, alias_map=ALIAS_MAP, redis_con=get_alias_redis_connection()):
    """
        insert all tokens that are not in alias map if they appear in redis into alias map
    """
    found_aliases = defaultdict(list)
    for token in set(tokens):
        if token in alias_map or redis_con.scard(token) == 0:
            continue

        for alias in redis_con.smembers(token):
            found_aliases[token].append(alias.decode('utf-8'))
    return found_aliases


def insert_aliases(tokens, alias_map=ALIAS_MAP, redis_con=get_alias_redis_connection()):
    """Inserts alias tokens to the current token list.

    Parameters
    ----------
    tokens: list of unigram
    alias_map: token -> iterable of aliasses
    redis_con: redis db to get the aliasses from
    """
    found_aliases = get_missing_aliases(tokens, alias_map, redis_con)
    for token in set(tokens):
        if token in found_aliases:
            for alias in found_aliases[token]:
                tokens.extend(alias.split(' '))
            continue
    return tokens


def insert_and_return_aliases(tokens, alias_map=ALIAS_MAP, redis_con=get_alias_redis_connection()):
    """Inserts alias tokens to the current token list
    and return a dictionary of token-> alias-list

    Parameters
    ----------
    tokens: list of unigram
    alias_map: token -> iterable of aliasses
    redis_con: redis db to get the aliasses from
    """
    found_aliases = get_missing_aliases(tokens, alias_map, redis_con)

    for token in set(tokens):
        if token in found_aliases:
            for alias in found_aliases[token]:
                found_aliases[token].append(alias.decode('utf-8'))
                tokens.extend(alias.decode('utf-8').split(' '))
            continue
    return tokens, found_aliases


def replace_aliases(tokens, alias_map=ALIAS_MAP, redis_con=get_alias_redis_connection()):
    """ replaces a token with all its aliases

    Parameters
    ----------
    tokens: list of unigram
    alias_map: token -> iterable of aliasses
    redis_con: redis db to get the aliasses from
    """
    found_aliases = get_missing_aliases(tokens, alias_map, redis_con)
    for token in set(tokens):
        if token in alias_map:
            for alias in found_aliases[token]:
                tokens.extend(alias.decode('utf-8').split(' '))
            tokens.remove(token)
            continue
    return tokens

if __name__ == '__main__':
    title = 'クリスタルガイザー CRYSTAL GEYSER ミネラルウォーター ナチュラルミネラルウォーター 水 オランチャ・シャスタ産 PET 24本入り 500ml 軟水 <BR><BR>送料無料'
    tkns = ttp.clean_title(title).split(' ')
    tkns = insert_aliases(tkns)
    print 'insert_aliases'
    print title
    print ' '.join(tkns)
    tkns = ttp.clean_title(title).split(' ')
    tkns = replace_aliases(tkns)
    print 'replace_aliases'
    print title
    print ' '.join(tkns)
    tkns = ttp.clean_title(title).split(' ')
    tkns, aliases = insert_and_return_aliases(tkns)
    print 'insert_and_return_aliases'
    print title
    print ' '.join(tkns)
    print aliases
