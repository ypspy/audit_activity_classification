# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 23:32:14 2021

@author: yoonseok
"""

from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.tokenizer import MaxScoreTokenizer
from konlpy.tag import Komoran
from collections import defaultdict
import os
import pandas as pd
import re
from tqdm import tqdm


def doPreprocess(documentColumn, spacing = True, desc = None):

    container = []
    for string in tqdm(documentColumn, desc=desc):
        string = re.sub('[-=+,#/\?:^$.@*\"“”※~&%ⅰⅱⅲ○●ㆍ°’『!』」\\‘|\(\)\[\]\<\>`\'…》]', '', string)  # 특수문자 제거
        string = re.sub('\w*[0-9]\w*', '', string)  # 숫자 제거
        string = re.sub('\w*[a-zA-Z]\w*', '', string)  # 알파벳 제거
        string = string.strip()  # 문서 앞뒤 공백 제거
        if spacing:
            string = ' '.join(string.split())  # Replace Multiple whitespace into one
        else:
            string = ''.join(string.split())
        
        container.append(string)
    return container

def get_ngram_counter(dataframe, min_count=10, n_range=(1,3)):

    def to_ngrams(words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
        return ngrams

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in dataframe:  # https://calmcode.io/tqdm/nested-loops.html
        
        for n in range(n_begin, n_end + 1):
            for ngram in to_ngrams(doc, n):
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

df["documents"] = doPreprocess(df["documents"], desc="Audit Standard")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 3]  # 입력내용이 3을 초과하는 입력값

df2 = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")
df2 = df2.dropna()  # Nan 제거
df2["documents"] = doPreprocess(df2["documents"], desc="감사업무수행내역")
df2["documents"].str.strip()  # Whitespace 제거
df2["count"] = df2["documents"].str.len()  # string 수 
df2 = df2[df2["count"] > 3]  # 입력내용이 3을 초과하는 입력값

df = df.append(df2)

noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
noun_extractor.train(df["documents"])
nouns = noun_extractor.extract()

noun_scores = {noun: (score.score if score.score <= 1 else 1) for noun, score in nouns.items()}

word_extractor = WordExtractor()
word_extractor.train(df["documents"])
word_score_table = word_extractor.extract()

cohesion_scores = {word:score.cohesion_forward for word, score in word_score_table.items()}

combined_scores = {noun:score + cohesion_scores.get(noun, 0)
    for noun, score in noun_scores.items()}
combined_scores.update(
    {subword:cohesion for subword, cohesion in cohesion_scores.items()
    if not (subword in combined_scores)}
)

stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
stopwords = stopWord.stopword.to_list()

l_tokenizer = LTokenizer(scores=combined_scores)
max_tokenizer = MaxScoreTokenizer(scores=combined_scores)

container_l, container_m = [], []
for docs in tqdm(df["documents"], desc="Ltokenizer"):
    
    l_tokenList = l_tokenizer.tokenize(docs)
    for i in l_tokenList:  # 추출된 형태소 중 불용어 제거 https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
            if i in stopwords:
                l_tokenList = list(filter(lambda a: a != i, l_tokenList))
    container_l.append(l_tokenList)
    
    m_tokenList = max_tokenizer.tokenize(docs)
    for i in m_tokenList:
            if i in stopwords:
                m_tokenList = list(filter(lambda a: a != i, m_tokenList))
    container_m.append(m_tokenList)
    
df["Ltoken"] = container_l
df["Mtoken"] = container_m

# N-gram 적용 토큰화
ngram_counter = get_ngram_counter(df["Mtoken"], 
                                  min_count=100, 
                                  n_range=(2,5))  # 2-3 단어까지 추출

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