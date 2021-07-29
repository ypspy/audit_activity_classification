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
from collections import defaultdict
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

# Data preprocessing

os.chdir(r"C:\analytics")

df = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 3을 초과하는 입력값

# Tokenization

df["split"] = [' '.join(x.split()) for x in tqdm(df["documents"], desc="split")]

komoran = Komoran("EXP")
df["Komoran_morph"] = [' '.join(komoran.get_morphes_by_tags(x)) for x in tqdm(df["documents"], desc="Komoran_morph")]
df["Komoran_noun"] = [' '.join(komoran.get_nouns(x)) for x in tqdm(df["documents"], desc="Komoran_noun")]

hannanum = Hannanum()
df["Hannanum_morph"] = [' '.join(hannanum.morphs(x)) for x in tqdm(df["documents"], desc="Hannanum_morph")]
df["Hannanum_noun"] = [' '.join(hannanum.nouns(x)) for x in tqdm(df["documents"], desc="Hannanum_noun")]

kkma = Kkma()
df["Kkma_morph"] = [' '.join(kkma.morphs(x)) for x in tqdm(df["documents"], desc="Kkma_morph")]
df["Kkma_noun"] = [' '.join(kkma.nouns(x)) for x in tqdm(df["documents"], desc="Kkma_noun")]

okt = Okt()
df["Okt_morph"] = [' '.join(okt.morphs(x)) for x in tqdm(df["documents"], desc="Okt_morph")]
df["Okt_noun"] = [' '.join(okt.nouns(x)) for x in tqdm(df["documents"], desc="Okt_noun")]

# komoran + 사용자 사전

komoran_2gram = Komoran("EXP")
komoran_2gram.set_user_dic("2gramdict.txt")
df["Komoran_noun_2gram"] = [' '.join(komoran_2gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_2gram")]

komoran_3gram = Komoran("EXP")
komoran_3gram.set_user_dic("3gramdict.txt")
df["Komoran_noun_3gram"] = [' '.join(komoran_3gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_3gram")]

komoran_4gram = Komoran("EXP")
komoran_4gram.set_user_dic("4gramdict.txt")
df["Komoran_noun_4gram"] = [' '.join(komoran_4gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_4gram")]

komoran_5gram = Komoran("EXP")
komoran_5gram.set_user_dic("5gramdict.txt")
df["Komoran_noun_5gram"] = [' '.join(komoran_5gram.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_5gram")]

komoran_soynlp = Komoran("EXP")
komoran_soynlp.set_user_dic("soynlpdict.txt")
df["Komoran_noun_soynlp"] = [' '.join(komoran_soynlp.get_nouns(x)) for x in tqdm(df["Komoran_noun"], desc="Komoran_noun_soynlp")]

# Soynlp

df2 = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")
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
df["Soynlp_morph"] = [' '.join(l_tokenizer.tokenize(x)) for x in tqdm(df["documents"], desc="Soynlp_Morph")]

# Vectorizer and Classifier

countVec = CountVectorizer()
randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

# Vectorize

splitVec = countVec.fit_transform(df["split"])

komoranMorphVec = countVec.fit_transform(df["Komoran_morph"])
hannanumMorphVec = countVec.fit_transform(df["Hannanum_morph"])
kkmaMorphVec = countVec.fit_transform(df["Kkma_morph"])
oktMorphVec = countVec.fit_transform(df["Okt_morph"])
soynlpMorphVec = countVec.fit_transform(df["Soynlp_morph"])

komoranNounVec = countVec.fit_transform(df["Komoran_noun"])
hannanumNounVec = countVec.fit_transform(df["Hannanum_noun"])
kkmaNounVec = countVec.fit_transform(df["Kkma_noun"])
oktNounVec = countVec.fit_transform(df["Okt_noun"])

komoranNounVec2gram = countVec.fit_transform(df["Komoran_noun_2gram"])
komoranNounVec3gram = countVec.fit_transform(df["Komoran_noun_3gram"])
komoranNounVec4gram = countVec.fit_transform(df["Komoran_noun_4gram"])
komoranNounVec5gram = countVec.fit_transform(df["Komoran_noun_5gram"])
komoranNounVecSoynlp = countVec.fit_transform(df["Komoran_noun_soynlp"])

# Evaluate

vectors = {"splitVec": splitVec, 
           "komoranMorphVec": komoranMorphVec,
           "hannanumMorphVec": hannanumMorphVec,
           "kkmaMorphVec": kkmaMorphVec,
           "oktMorphVec": oktMorphVec,
           "soynlpMorphVec": soynlpMorphVec,
           "komoranNounVec": komoranNounVec,
           "hannanumNounVec": hannanumNounVec,
           "kkmaNounVec": kkmaNounVec,
           "oktNounVec": oktNounVec,
           "komoranNounVec2gram": komoranNounVec2gram,
           "komoranNounVec3gram": komoranNounVec3gram,
           "komoranNounVec4gram": komoranNounVec4gram,
           "komoranNounVec5gram": komoranNounVec5gram,
           "komoranNounVecSoynlp": komoranNounVecSoynlp}
models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df["Label"], model)
        print("---")
        returnCVScore(value, df["Label"], model)
