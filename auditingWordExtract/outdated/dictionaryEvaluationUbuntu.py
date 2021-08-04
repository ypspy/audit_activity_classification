# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:19:20 2021

@author: yoonseok
"""

from ckonlpy.tag import Twitter
from ckonlpy.utils import load_ngram
from ckonlpy.tag import Postprocessor

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

def removeStopword(dataframe):
    stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
    stopword = stopWord.stopword.to_list()
    container = []
    for packages in tqdm(dataframe, desc="stop words"):
        container2 = []
        for words in packages:
            if words not in stopword:
                container2.append(words)
        container.append(container2)
    return container


# Data preprocessing

# os.chdir(r"/home/yoonseokseong/downloads/idea-IC-212.4746.92/bin")
os.chdir(r"C:\analytics")

stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
stopwords = stopWord.stopword.to_list()

df = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data", engine="openpyxl")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 3을 초과하는 입력값   

# Tokenization
twitter = Twitter()

df["documents"] = [' '.join(x) for x in removeStopword([twitter.nouns(x) for x in tqdm(df["documents"], desc="Twitter_noun")])]

ngrams = load_ngram('./4gramdict.txt')
postprocessor = Postprocessor(twitter, ngrams = ngrams)

df["Twitter_noun"] = [postprocessor.pos(x) for x in tqdm(df["documents"], desc="Twitter_noun")]

container = []
for tupleOut in df["Twitter_noun"]:
    wordSet = ''
    for words in tupleOut:
        wordSet = wordSet + " " + words[0].replace(" - ", "")
    container.append(wordSet)
df["Twitter_noun_dict"] = container


# Vectorizer and Classifier

countVec = CountVectorizer()
randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

# Vectorize

twitterNounVec = countVec.fit_transform(df["Twitter_noun_dict"].values.astype("U"))

# Train and Evaluate

vectors = {
           "twitterNounVec": twitterNounVec,}

models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df["Label"], model)
        print("---")
        returnCVScore(value, df["Label"], model)
