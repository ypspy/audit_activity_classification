# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 23:32:14 2021

@author: yoonseok
"""

from soynlp.noun import LRNounExtractor_v2
from ckonlpy.tag import Twitter
from collections import defaultdict

import os
import pandas as pd
import re

from tqdm import tqdm

### n-gram 토크나이저 참조 https://lovit.github.io/nlp/2018/10/23/ngram/

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

def extractNgram(documents, N, base_tokenizer, min_count=10):
    """
    리스트/Array가 들어오는데, 요소는 string이다.     
    """
    
    ngramList = []

    ngram_counter = get_ngram_counter(documents,
                                      base_tokenizer,
                                      min_count, 
                                      n_range=(1,N))  # 2-3 단어까지 추출
    
    ngram_tokenizer = NgramTokenizer(ngram_counter, base_tokenizer)
    loopCounter = ngram_counter.items()
    
    for key in tqdm(loopCounter, desc="ListOut"):       
        out = []
        
        keyText = " ".join(key[0])
        counter = "-".join(key[0]).count("-")
        out = [keyText, counter]
        
        ngramList.append(out)
    
    tokenizedList = []
    
    for docs in documents:
        tokenizedList.append(ngram_tokenizer(docs))
        
    NgramDictionary = pd.DataFrame(ngramList)
    NgramDictionary["pos"] = tokenizedList
    NgramDictionary["tag"] = "Noun"
    NgramDictionary = NgramDictionary[[0, "pos", "tag"]]
        
    return NgramDictionary

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

# os.chdir(r"/home/yoonseokseong/downloads/idea-IC-212.4746.92/bin")
os.chdir(r"C:\analytics")

df = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data", engine="openpyxl")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 3을 초과하는 입력값   

# Tokenization
twitter_Konlpy = Twitter()
twitter = TwitterMod(twitter_Konlpy, noun=True)

n_range=(1,2)

ngram_counter = get_ngram_counter(df["documents"], twitter, min_count=1000, n_range=n_range)
ngram_tokenizer = NgramTokenizer(ngram_counter, twitter, n_range=n_range)

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel("tokenizerEvaluationData - tagging.xlsm", sheet_name="data", engine="openpyxl")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 제거
df = df.dropna()  # Nan 제거
df["documents"] = doPreprocess(df["documents"], desc="Preprocess")
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 1]  # 입력내용이 3을 초과하는 입력값  

vectorizer = CountVectorizer(
    tokenizer = ngram_tokenizer,
    lowercase = False,
)
x = vectorizer.fit_transform(df["documents"])



from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from konlpy.tag import Hannanum, Kkma, Okt, Twitter
from PyKomoran import Komoran
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


vectors = {"vec": x, }

randomForest = RandomForestClassifier(random_state=99)
nb = MultinomialNB()

models = [randomForest, nb]

for model in models:
    for key, value in vectors.items():
        print("-------------{0}:{1}-------------".format(key, model))
        evaluateVectorizer(value, df["Label"], model)
        print("---")
        returnCVScore(value, df["Label"], model)


























# # N-gram 적용 토큰화

# _2gram = extractNgram(df["Twitter_noun"], 2, twitter, min_count=100)
# _3gram = extractNgram(df["Twitter_noun"], 3, twitter, min_count=100)
# _4gram = extractNgram(df["Twitter_noun"], 4, twitter, min_count=100)
# _5gram = extractNgram(df["Twitter_noun"], 5, twitter, min_count=100)

# _2gram.to_csv('2gramdict.txt',
#               sep='\t',
#               header=False,
#               index=False)

# _3gram.to_csv('3gramdict.txt',
#               sep='\t',
#               header=False,
#               index=False)

# _4gram.to_csv('4gramdict.txt',
#               sep='\t',
#               header=False,
#               index=False)

# _5gram.to_csv('5gramdict.txt',
#               sep='\t',
#               header=False,
#               index=False)

# # Soynlp

# noun_extractor = LRNounExtractor_v2(verbose=False, extract_compound=True)
# nouns = noun_extractor.train_extract(df["documents"])

# soynlpNounList = []

# for value in tqdm(list(noun_extractor._compounds_components.items()), desc="soynlp"):
#     keyText = " ".join(value[1])
#     counter = "-".join(value[1]).count("-")
#     out = [keyText, counter]
    
#     soynlpNounList.append(out)

# soydf = pd.DataFrame(soynlpNounList).sort_values(by=[1], ascending=False)

# soydf["tag"] = "Noun"
# soydf = soydf[[0, "tag"]]
# soydf.to_csv('soynlpdict.txt',
#              sep='\t',
#              header=False,
#              index=False)                  
