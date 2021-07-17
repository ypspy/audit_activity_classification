# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 00:21:53 2021
@author: yoonseok
Soynlp를 이용하여 형태소분석기를 학습시키고, 학습된 모델(개체)를 pkl형식으로 저장
"""

import os
import pandas as pd
import re
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import MaxScoreTokenizer
import joblib

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

os.chdir(r"C:\analytics")

# df = pd.read_csv("auditingStandard.csv", encoding="cp949", names=('documents', 'B'))
df = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")
df = df.dropna()  # Nan 제거

df["documents"] = doPreprocess(df["documents"])

df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 3]  # 입력내용이 3을 초과하는 입력값

word_extractor = WordExtractor(min_frequency=100,
                                min_cohesion_forward=0.05,
                                min_right_branching_entropy=0.0)

word_extractor.train(df["documents"])
words = word_extractor.extract()

cohesion_score = {word:score.cohesion_forward for word, score in words.items()}

noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(df["documents"])

noun_scores = {noun:score.score for noun, score in nouns.items()}

combined_scores = {noun:score + cohesion_score.get(noun, 0) for noun, score in noun_scores.items()}
combined_scores.update({subword:cohesion for subword, cohesion in cohesion_score.items() if not (subword in combined_scores)})

L_Tokenizer = LTokenizer(scores=cohesion_score)
M_Tokenizer = MaxScoreTokenizer(scores=cohesion_score)

joblib.dump(L_Tokenizer, './L_tokenizer_auditing.pkl')
joblib.dump(M_Tokenizer, './M_tokenizer_auditing.pkl')
