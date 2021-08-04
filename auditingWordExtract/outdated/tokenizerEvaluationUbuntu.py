# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:19:20 2021

@author: yoonseok
"""

from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from konlpy.tag import Hannanum, Kkma, Okt
from PyKomoran import Komoran
from ckonlpy.tag import Twitter
import math

import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score


def returnCVScore(X, y, model):
    scores = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    cvScores = cross_validate(model, X, y, scoring=scores, cv=5)
    
    accuracy = np.round(np.mean(cvScores["test_accuracy"]), 4)
    precision = np.round(np.mean(cvScores["test_precision_macro"]), 4)
    recall = np.round(np.mean(cvScores["test_recall_macro"]), 4)
    f1 = np.round(np.mean(cvScores["test_f1_macro"]), 4)    
    
    print("Accuracy:{:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))
    
    return None

def returnScore(y_test, y_predict, model, average="macro"):

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average = average)
    recall = recall_score(y_test, y_predict, average = average)
    f1 = f1_score(y_test, y_predict, average = average)
    
    print("Accuracy:{:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict))

    return None

def evaluateVectorizer(vector, label, model, test_size=0.3, random_state=11):
    X_train, X_test, y_train, y_test = train_test_split(vector,
                                                        label,
                                                        test_size=test_size,
                                                        random_state=random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    returnScore(y_test, pred, model, average="macro")
    return None

def doPreprocess(documentColumn, spacing = True, desc = None):

    container = []
    for string in tqdm(documentColumn, desc=desc):
        string = re.sub('[-=+,#/\?:^$.@*\"“”※~&%ⅰⅱⅲ○●ㆍ°’『!』」\\‘|\(\)\[\]\<\>`\'…》]', '', string)  # 특수문자 제거
        string = re.sub('\w*[0-9]\w*', '', string)  # 숫자 제거
        # string = re.sub('\w*[a-zA-Z]\w*', '', string)  # 알파벳 제거
        string = string.strip()  # 문서 앞뒤 공백 제거
        if spacing:
            string = ' '.join(string.split())  # Replace Multiple whitespace into one
        else:
            string = ''.join(string.split())
        
        container.append(string)
    return container


# Data preprocessing

# os.chdir(r"/home/yoonseokseong/downloads/idea-IC-212.4746.92/bin")
os.chdir(r"C:\analytics")
df = pd.read_csv("preprocessed.txt")

df2 = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data", engine="openpyxl")
df2 = df2.loc[:, ~df2.columns.str.contains('Unnamed')]  # 제거
df2 = df2.dropna()  # Nan 제거
df2["documents"] = doPreprocess(df2["documents"], desc="Preprocess")
df2["documents"].str.strip()  # Whitespace 제거
df2["count"] = df2["documents"].str.len()  # string 수 
df2 = df2[df2["count"] > 1]  # 입력내용이 3을 초과하는 입력값   

# Tokenization
okt = Twitter()
df["documents"] = [str(x).replace("-", " ") for x in tqdm(df["documentsEval"], desc="Replace-")]
df["Okt_noun"] = [okt.nouns(x) for x in tqdm(df["documents"], desc="Okt_noun")]
df["Label"] = df2["Label"]

df2["Okt_noun"] = [' '.join(okt.nouns(x)) for x in tqdm(df2["documents"], desc="Okt_noun")]

container = []

for i in tqdm(df2["Okt_noun"]):
    mod, count = '', 0
    
    for j in i.split():
        if count:
            mod = mod + j
        else:
            mod = j
        count += 1
    container.append(mod)

df2["documentsEval"] = container

df2["documentsEval"].to_csv('preprocessed2.txt',
                            sep='\t',
                            header=True,
                            index=False)

# Vectorizer and Classifier

countVec = CountVectorizer()
randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

# Vectorize

oktNounVec = countVec.fit_transform(df["documents"].values.astype("U"))
oktNounVec2 = countVec.fit_transform(df2["Okt_noun"].values.astype("U"))

# Evaluate

vectors = {
           "oktNounVec": oktNounVec,
           "oktNounVec2": oktNounVec2}

models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df["Label"], model)
        print("---")
        returnCVScore(value, df["Label"], model)
