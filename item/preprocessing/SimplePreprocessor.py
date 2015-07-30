# -*- coding: utf-8 -*-
import TitlePreprocessing as ttp
from SpamRemover import NLPSpamRemover
import AliasInserter as ais
import SeriesFinder as sf

def preprocess(title, srm=NLPSpamRemover()):
    tokenized_title = ttp.clean_title(title)
    
    tokens = []
    for t in tokenized_title.split(' '):
        adj = t.replace(' ', '').replace(u'\u3000', '')
        if len(adj) < 4 and not adj.isalnum():
            continue
        if u'\u5186' in t:  #'円'
            continue
        #print len(adj), t, len(t), adj.isalnum()
        tokens.append(adj)

    clean_tokens = srm.remove_spam2(tokens)

    return tokenized_title, clean_tokens


