# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:19:20 2021

@author: yoonseok
"""

from konlpy.tag import Mecab

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

os.chdir(r"/mnt/c/analytics")

df = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 3을 초과하는 입력값

# Tokenization

mecab = Mecab()
df["Mecab_morph"] = [' '.join(mecab.morphs(x)) for x in tqdm(df["documents"], desc="Mecab_morph")]
df["Mecab_noun"] = [' '.join(mecab.nouns(x)) for x in tqdm(df["documents"], desc="Mecab_noun")]

# komoran + 사용자 사전

# komoran_2gram = Komoran("EXP")
# komoran_2gram.set_user_dic("2gramdict.txt")
# df["Komoran_noun_2gram"] = [' '.join(komoran_2gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_2gram")]

# komoran_3gram = Komoran("EXP")
# komoran_3gram.set_user_dic("3gramdict.txt")
# df["Komoran_noun_3gram"] = [' '.join(komoran_3gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_3gram")]

# komoran_4gram = Komoran("EXP")
# komoran_4gram.set_user_dic("4gramdict.txt")
# df["Komoran_noun_4gram"] = [' '.join(komoran_4gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_4gram")]

# komoran_5gram = Komoran("EXP")
# komoran_5gram.set_user_dic("5gramdict.txt")
# df["Komoran_noun_5gram"] = [' '.join(komoran_5gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_5gram")]

# komoran_soynlp = Komoran("EXP")
# komoran_soynlp.set_user_dic("soynlpdict.txt")
# df["Komoran_noun_soynlp"] = [' '.join(komoran_soynlp.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_soynlp")]

# Vectorizer and Classifier

countVec = CountVectorizer()
randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

# Vectorize

mecabMorphVec = countVec.fit_transform(df["Mecab_morph"])
mecabNounVec = countVec.fit_transform(df["Mecab_noun"])

# Evaluate

vectors = {
           "mecabMorphVec": mecabMorphVec,
           "mecabNounVec": mecabNounVec,}

models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df["Label"], model)
        print("---")
        returnCVScore(value, df["Label"], model)
