# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:25:51 2021

@author: yoonseok

1. n-gram extraction https://lovit.github.io/nlp/2018/10/23/ngram/
2. defaultdict(int) https://itholic.github.io/python-defaultdict/
"""

from konlpy.tag import Komoran
import os
import pandas as pd
import re
from collections import defaultdict

def doPreprocess(documentColumn):
    """
    Dataframe에서 문서열을 받아와서, 특수문자, 숫자, 알파벳, 불필요한 공백을 제거

    Parameters
    ----------
    documentColumn : pandas.core.series.Series
        Dataframe의 문서열

    Returns
    -------
    container : TYPE
        DESCRIPTION.

    """
    container = []
    for string in documentColumn:
        string = re.sub('[-=+,#/\?:^$.@*\"“”※~&%ⅰⅱⅲ○●ㆍ°’『!』」\\‘|\(\)\[\]\<\>`\'…》]', '', string)  # 특수문자 제거
        string = re.sub('\w*[0-9]\w*', '', string)  # 숫자 제거
        string = re.sub('\w*[a-zA-Z]\w*', '', string)  # 알파벳 제거
        string = string.strip()  # 문서 앞뒤 공백 제거
        string = ' '.join(string.split())  # Replace Multiple whitespace into one
        
        container.append(string)
    return container

def to_ngrams(words, n):
    ngrams = []
    for b in range(0, len(words) - n + 1):
        ngrams.append(tuple(words[b:b+n]))
    return ngrams

def get_ngram_counter(docs, min_count=10, n_range=(1,3)):

    def to_ngrams(words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
        return ngrams

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:
        words = komoran.pos(doc, join=True)
        for n in range(n_begin, n_end + 1):
            for ngram in to_ngrams(words, n):
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram:count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter


# data preprocessing

os.chdir(r"C:\analytics")

df = pd.read_csv("auditingStandard.csv", encoding="cp949", names=('documents', 'B'))
df = df.loc[:, ~df.columns.str.contains('B')]  # 제거
df = df.dropna()  # Nan 제거

df["documents"] = doPreprocess(df["documents"])

df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 3]  # 입력내용이 3을 초과하는 입력값

df2 = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")
df2 = df2.dropna()  # Nan 제거

df2["documents"] = doPreprocess(df2["documents"])

df2["documents"].str.strip()  # Whitespace 제거
df2["count"] = df2["documents"].str.len()  # string 수 
df2 = df2[df2["count"] > 3]  # 입력내용이 3을 초과하는 입력값

df = df.append(df2)

# N-gram 적용 토큰화

komoran = Komoran()
ngram_counter = get_ngram_counter(df["documents"], n_range=(1,5))  # 5 단어까지 추출

ngramList = []

for key in ngram_counter.items():
    if "/N" in key[0][0] and "/N" in key[0][-1]:
        key = "".join(key[0])
        key = re.sub('[\/A-Z]', '-', key)
        ngramList.append(key)

exportNgramDictionary = pd.DataFrame(ngramList)
exportNgramDictionary.to_excel('Ngramdictionary.xlsx')
