# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:19:20 2021

@author: yoonseok
"""

from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from konlpy.tag import Hannanum, Kkma, Okt, Mecab
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

def removeStopword(dataframe):
    stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
    stopword = stopWord.stopword.to_list()
    
    container = []
    for packages in dataframe:
        container2 = []
        for words in packages:
            if words not in stopword:
                container2.append(words)
        container.append(container2)
    return container


# Data preprocessing

os.chdir("/home/yoonseokseong/downloads/idea-IC-212.4746.92/bin")

stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
stopword = stopWord.stopword.to_list()

df = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data", engine="openpyxl")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 3을 초과하는 입력값   

# Tokenization

df["split"] = [' '.join(x.split()) for x in tqdm(df["documents"], desc="split")]

komoran = Komoran("EXP")
df["Komoran_morph"] = [' '.join(x) for x in removeStopword([komoran.get_morphes_by_tags(x) for x in tqdm(df["documents"], desc="Komoran_morph")])]
df["Komoran_noun"] = [' '.join(x) for x in removeStopword([komoran.get_nouns(x) for x in tqdm(df["documents"], desc="Komoran_noun")])]

hannanum = Hannanum()
df["Hannanum_morph"] = [' '.join(x) for x in removeStopword([hannanum.morphs(x) for x in tqdm(df["documents"], desc="Hannanum_morph")])]
df["Hannanum_noun"] = [' '.join(x) for x in removeStopword([hannanum.nouns(x) for x in tqdm(df["documents"], desc="Hannanum_noun")])]

kkma = Kkma()
df["Kkma_morph"] = [' '.join(x) for x in removeStopword([kkma.morphs(x) for x in tqdm(df["documents"], desc="Kkma_morph")])]
df["Kkma_noun"] = [' '.join(x) for x in removeStopword([kkma.nouns(x) for x in tqdm(df["documents"], desc="Kkma_noun")])]

okt = Okt()
df["Okt_morph"] = [' '.join(x) for x in removeStopword([okt.morphs(x) for x in tqdm(df["documents"], desc="Okt_morph")])]
df["Okt_noun"] = [' '.join(x) for x in removeStopword([okt.nouns(x) for x in tqdm(df["documents"], desc="Okt_noun")])]

mecab = Mecab()
df["Mecab_morph"] = [' '.join(x) for x in removeStopword([mecab.morphs(x) for x in tqdm(df["documents"], desc="Mecab_morph")])]
df["Mecab_noun"] = [' '.join(x) for x in removeStopword([mecab.nouns(x) for x in tqdm(df["documents"], desc="Mecab_noun")])]

twitter = Twitter()
df["Twitter_morph"] = [' '.join(x) for x in removeStopword([twitter.morphs(x) for x in tqdm(df["documents"], desc="Twitter_morph")])]
df["Twitter_noun"] = [' '.join(x) for x in removeStopword([twitter.nouns(x) for x in tqdm(df["documents"], desc="Twitter_noun")])]

df.to_csv("dataframe.txt")  # 결과 모니터링

# Soynlp

df2 = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data", engine="openpyxl")
df2 = df2.dropna()  # Nan 제거
df2["documents"] = doPreprocess(df2["documents"], desc="감사업무수행내역")
df2["documents"].str.strip()  # Whitespace 제거
df2["count"] = df2["documents"].str.len()  # string 수 
df2 = df2[df2["count"] > 1]  # 입력내용이 3을 초과하는 입력값

noun_extractor = LRNounExtractor_v2(verbose=False, extract_compound=True)
nouns = noun_extractor.train_extract(df2["documents"])
noun_scores = {noun: (score.score if score.score <= 1 else 1) for noun, score in nouns.items()}

word_extractor = WordExtractor()
word_extractor.train(df2["documents"])
word_score_table = word_extractor.extract()

scores = {word:score.cohesion_forward * math.exp(score.right_branching_entropy) for word, score in word_score_table.items()}
# https://github.com/lovit/soynlp/blob/master/tutorials/wordextractor_lecture.ipynb

cohesion_scores = {word:score.cohesion_forward for word, score in word_score_table.items()}

combined_scores = {noun:score + cohesion_scores.get(noun, 0)
    for noun, score in noun_scores.items()}
combined_scores.update(
    {subword:cohesion for subword, cohesion in cohesion_scores.items()
    if not (subword in combined_scores)}
)

l_tokenizer = LTokenizer(scores=combined_scores)
df["Soynlp_morph"] = [' '.join(x) for x in removeStopword([l_tokenizer.tokenize(x) for x in tqdm(df["documents"], desc="Soynlp_Morph")])]

# Vectorizer and Classifier

countVec = CountVectorizer()
randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

# Vectorize

splitVec = countVec.fit_transform(df["split"].values.astype("U"))

komoranMorphVec = countVec.fit_transform(df["Komoran_morph"].values.astype("U"))
hannanumMorphVec = countVec.fit_transform(df["Hannanum_morph"].values.astype("U"))
kkmaMorphVec = countVec.fit_transform(df["Kkma_morph"].values.astype("U"))
oktMorphVec = countVec.fit_transform(df["Okt_morph"].values.astype("U"))
mecabMorphVec = countVec.fit_transform(df["Mecab_morph"].values.astype('U'))
twitterMorphVec = countVec.fit_transform(df["Twitter_morph"].values.astype('U'))
soynlpMorphVec = countVec.fit_transform(df["Soynlp_morph"].values.astype("U"))

komoranNounVec = countVec.fit_transform(df["Komoran_noun"].values.astype("U"))
hannanumNounVec = countVec.fit_transform(df["Hannanum_noun"].values.astype("U"))
kkmaNounVec = countVec.fit_transform(df["Kkma_noun"].values.astype("U"))
oktNounVec = countVec.fit_transform(df["Okt_noun"].values.astype("U"))
mecabNounVec = countVec.fit_transform(df["Mecab_noun"].values.astype('U'))
twitterNounVec = countVec.fit_transform(df["Twitter_noun"].values.astype("U"))

print("Twitter Shape: {}".format(twitterNounVec.shape))

# Evaluate

vectors = {"splitVec": splitVec, 
           "komoranMorphVec": komoranMorphVec,
           "hannanumMorphVec": hannanumMorphVec,
           "kkmaMorphVec": kkmaMorphVec,
           "oktMorphVec": oktMorphVec,
           "mecabMorphVec": mecabMorphVec,
           "twitterMorphVec": twitterMorphVec,
           "soynlpMorphVec": soynlpMorphVec,
           "komoranNounVec": komoranNounVec,
           "hannanumNounVec": hannanumNounVec,
           "kkmaNounVec": kkmaNounVec,
           "oktNounVec": oktNounVec,
           "mecabNounVec": mecabNounVec,
           "twitterNounVec": twitterNounVec,}

models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df["Label"], model)
        print("---")
        returnCVScore(value, df["Label"], model)

