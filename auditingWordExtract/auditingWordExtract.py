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
import joblib

def doPreprocess(documentColumn):
    """
    Dataframe에서 문서열을 받아와서, 특수문자, 숫자, 알파벳, 불필요한 공백을 제거

    Parameters
    ----------
    documentColumn : pandas.core.series.Series
        Dataframe의 문서열을 입력

    Returns
    -------
    container : list
        처리된 문서를 리스트에 담아서 반환

    """
    container = []
    for string in documentColumn:
        string = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', string)  # 특수문자 제거
        string = re.sub('\w*[0-9]\w*', '', string)  # 숫자 제거
        string = re.sub('\w*[a-zA-Z]\w*', '', string)  # 알파벳 제거
        string = string.strip()  # 문서 앞뒤 공백 제거
        string = ' '.join(string.split())  # Replace Multiple whitespace into one
        
        container.append(string)
    return container

os.chdir(r"C:\analytics")

df = pd.read_csv("auditingStandard.csv", encoding="cp949", names=('documents', 'B'))
df = df.loc[:, ~df.columns.str.contains('B')]
df = df.dropna()
df["documents"] = doPreprocess(df["documents"])

word_extractor = WordExtractor(min_frequency=100,
                               min_cohesion_forward=0.05,
                               min_right_branching_entropy=0.0)

word_extractor.train(df["documents"])

joblib.dump(word_extractor, './word_extractor_auditing.pkl')
