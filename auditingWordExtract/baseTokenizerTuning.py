# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 23:32:14 2021

@author: yoonseok
"""

from ckonlpy.tag import Twitter
from collections import defaultdict

import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score

from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer


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

### n-gram 토크나이저 lovit/soynlp 참조 https://lovit.github.io/nlp/2018/10/23/ngram/

class NgramTokenizer:
 
    def __init__(self, ngrams, base_tokenizer, n_range=(1, 3)):
        self.ngrams = ngrams
        self.base_tokenizer = base_tokenizer
        self.n_range = n_range

    def __call__(self, sent):
        return self.tokenize(sent)

    def tokenize(self, sent):
        if not sent:
            return []

        unigrams = self.base_tokenizer.tokenize(sent)  # pos(sent, join=True)

        n_begin, n_end = self.n_range
        ngrams = []
        for n in range(n_begin, n_end + 1):
            for ngram in self._to_ngrams(unigrams, n):
                ngrams.append('-'.join(ngram))
        return ngrams

    def _to_ngrams(self, words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngram = tuple(words[b:b+n])
            if ngram in self.ngrams:
                ngrams.append(ngram)
        return ngrams

def get_ngram_counter(dataframe, base_tokenizer, min_count=10, n_range=(1,5)):
    """
    리스트/Array가 들어오는데, 요소는 string이다.
    """
    def to_ngrams(words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
        return ngrams

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in tqdm(dataframe):
        words = base_tokenizer.tokenize(doc)  # Twitter의 경우 tokenize/그 외 pos

        for n in range(n_begin, n_end + 1):
            ngramList = to_ngrams(words, n)

            for ngram in ngramList:  
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram:count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter

###

class TwitterMod():
    """
    cKonlpy의 twitter에는 join이 없다. 단일 목적 클래스. twitter를 집어넣어서 object를 만든다.
    리스트/Array가 들어오는데, 요소는 string이다. object를 만들때, noun=True를 하면 명사만 반환한다.
    반환은 리스트/Array이고 요소는 tag가 join된 형태소의 리스트이다. 불용어를 제거하지 않으려면 stopword=False.
    """
    
    def __init__(self, base_tokenizer, noun=False):
        self.base_tokenizer = base_tokenizer
        self.noun = noun

    def __call__(self, sent):
        return self.tokenize(sent)

    def tokenize(self, sent, stopword=True, pos=True):

        container, container2 = [], []
        
        for words in self.base_tokenizer.pos(sent):
            if self.noun:
                if words[1] == "Noun":
                    container2.append("{0}/{1}".format(words[0], words[1]))
            else:
               container2.append("{0}/{1}".format(words[0], words[1])) 
        container.append(container2)
        
        if stopword:
            container = removeStopword(container, pos)
                
        if len(container) == 1:
            container = container[0]
        
        return container
 

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

def removeStopword(dataframe, pos=True):
    """
    리스트/Array가 들어오고, 요소는 리스트.
    태그가 있는 경우(pos=True)와 없는 경우 처리가 달라진다.
    불용어가 제거된 리스트/Array를 반환한다.
    """
    stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
    stopword = stopWord.stopword.to_list()
    container = []
    for packages in dataframe:
        container2 = []
        for words in packages:
            if pos:
                if words[0: words.index('/')] not in stopword:                
                    container2.append(words)
            else:
                if words not in stopword:
                    container2.append(words)                    
        container.append(container2)
    return container


# data preprocessing
os.chdir(r"C:\analytics")

df = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data", engine="openpyxl")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 1을 초과하는 입력값   

# Tokenization
twitter_Konlpy = Twitter()
twitter = TwitterMod(twitter_Konlpy, noun=True)

# N-gram Tuning
n_range=(1,2)
ngram_counter = get_ngram_counter(df["documents"], twitter, min_count=10, n_range=n_range)
ngram_tokenizer = NgramTokenizer(ngram_counter, twitter, n_range=n_range)

# Training Data
df2 = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data", engine="openpyxl")
df2 = df2.loc[:, ~df2.columns.str.contains('Unnamed')]  # 제거
df2 = df2.dropna()  # Nan 제거
df2["documents"] = doPreprocess(df2["documents"], desc="Preprocess")
df2["documents"].str.strip()  # Whitespace 제거
df2["count"] = df2["documents"].str.len()  # string 수 
df2 = df2[df2["count"] > 1]  # 입력내용이 3을 초과하는 입력값  

# Soynlp Tokenizer

noun_extractor = LRNounExtractor_v2(verbose=False, extract_compound=True)
nouns = noun_extractor.train_extract(df["documents"])
noun_scores = {noun: (score.score if score.score <= 1 else 1) for noun, score in nouns.items()}

word_extractor = WordExtractor()
word_extractor.train(df["documents"])
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

# n-gram Vectorizer
vectorizer = CountVectorizer(tokenizer = ngram_tokenizer,
                             lowercase = False,
                             )
x = vectorizer.fit_transform(df2["documents"])

# soynlp Vectorizer
vectorizer = CountVectorizer(tokenizer = l_tokenizer,
                             lowercase = False,
                             )
y = vectorizer.fit_transform(df2["documents"])


# Machine Learning Object
randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

# Evaluation
vectors = {"ngramVec": x,
           "soynlpVec": y,}
models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df2["Label"], model)
        print("---")
        returnCVScore(value, df2["Label"], model)
