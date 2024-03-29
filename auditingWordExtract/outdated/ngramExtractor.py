# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:25:51 2021

@author: yoonseok

1. n-gram extraction https://lovit.github.io/nlp/2018/10/23/ngram/
2. defaultdict(int) https://itholic.github.io/python-defaultdict/
3. Komoran PoS Table https://docs.komoran.kr/firststep/postypes.html#pos-table
"""

from konlpy.tag import Komoran
import os
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm

def doPreprocess(documentColumn, model, stopWords, desc = None):
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
    for string in tqdm(documentColumn, desc=desc):
        string = re.sub('[-=+,#/\?:^$.@*\"“”※~&%ⅰⅱⅲ○●ㆍ°’『!』」\\‘|\(\)\[\]\<\>`\'…》]', '', string)  # 특수문자 제거
        # string = re.sub('\w*[0-9]\w*', '', string)  # 숫자 제거
        string = re.sub('\w*[a-zA-Z]\w*', '', string)  # 알파벳 제거
        string = string.strip()  # 문서 앞뒤 공백 제거
        string = ' '.join(string.split())  # Replace Multiple whitespace into one
        
        tokenList = model(string)  # 형태소분석기로 문서 형태소 추출
        for i in tokenList:  # 추출된 형태소 중 불용어 제거
            if i in stopwords:
                tokenList.remove(i)
        string = ' '.join(tokenList)
        container.append(string)
    return container

def get_ngram_counter(docs, model, min_count=10, n_range=(1,3)):

    def to_ngrams(words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
        return ngrams

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:  # https://calmcode.io/tqdm/nested-loops.html
        words = model(doc)  # pos(doc, join=True)
        for n in range(n_begin, n_end + 1):
            for ngram in to_ngrams(words, n):
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram:count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter

komoran = Komoran()

# data preprocessing

os.chdir(r"C:\analytics")

stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
stopwords = stopWord.stopword.to_list()

df = pd.read_csv("auditingStandard.csv", encoding="cp949", names=('documents', 'B'))
df = df.loc[:, ~df.columns.str.contains('B')]  # 제거
df = df.dropna()  # Nan 제거

df["documents"] = doPreprocess(df["documents"], 
                                komoran.nouns, 
                                stopwords,
                                desc="Audit Standard")

df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 3]  # 입력내용이 3을 초과하는 입력값

df2 = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")
df2 = df2.dropna()  # Nan 제거

df2["documents"] = doPreprocess(df2["documents"], 
                                komoran.nouns, 
                                stopwords,
                                desc="감사업무수행내역")

df2["documents"].str.strip()  # Whitespace 제거
df2["count"] = df2["documents"].str.len()  # string 수 
df2 = df2[df2["count"] > 3]  # 입력내용이 3을 초과하는 입력값

df = df.append(df2)

# N-gram 적용 토큰화
ngram_counter = get_ngram_counter(df["documents"], 
                                  komoran.nouns, 
                                  min_count=10, 
                                  n_range=(2,5))  # 2-5 단어까지 추출

ngramList = []

for key in tqdm(ngram_counter.items(), desc="ListOut"):       
    keyText = "-".join(key[0])
    keyText2 = keyText.replace("-", "")
    keyText3 = keyText.replace("-", " ")
    ngramList.append([keyText, keyText2, keyText3, key[1]])

NgramDictionary = pd.DataFrame(ngramList).sort_values(by=[3])

NgramDictionary["tag"] = "NNP"
NgramDictionary = NgramDictionary[[2, "tag"]]
NgramDictionary.to_csv('Ngramdictionary.txt',    # 코모란 사용자 사전 
                        sep='\t', 
                        header=False,
                        index=False)