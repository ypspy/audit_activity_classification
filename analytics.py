# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 00:48:39 2021

@author: yoonseok
"""

import pandas as pd
import numpy as np
import re
import random
import os

from ckonlpy.tag import Twitter, Postprocessor
import joblib
from soynlp.tokenizer import LTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def tokenizeDocuments(documentColumn, model, spacing=False):
    
    container = []
    for string in documentColumn:
        string = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘|\(\)\[\]\<\>`\'…》]', '', string)  # 특수문자 제거
        string = re.sub('\w*[0-9]\w*', '', string)  # 숫자 제거
        string = re.sub('\w*[a-zA-Z]\w*', '', string)  # 알파벳 제거
        string = string.strip()  # 문서 앞뒤 공백 제거
        if spacing:
            string = ''.join(string.split())  # Remove whitespace 
        else:
            string = ' '.join(string.split())  # Replace Multiple whitespace into one
        tokenList = model(string)
        for i in tokenList:
            if i in stopwords:
                tokenList.remove(i)
        container.append(tokenList)
    return container

def vectorizeCorpus(corpusArray, model):
    """    
    Parameters
    ----------
    corpusArray : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    container : TYPE
        DESCRIPTION.
    """
    container = []
    for i in corpusArray:
        vec = model.infer_vector(i)
        container.append(vec)
    return container

def transformDF2Corpus(df):
    df["document"] = df.document.apply(lambda x: x.split())
    container = []
    for i in df["docID"]:
        li = []
        li.append(i)
        container.append(li)
    df["docID"] = container
    doc_df = df[['docID','document']].values.tolist()
    train_corpus = [TaggedDocument(words=document2, tags=docID) for docID, document2 in doc_df]
    return train_corpus

def returnScore(y_test, y_predict, average):
    """
    Parameters
    ----------
    y_test : TYPE
        DESCRIPTION.
    y_predict : TYPE
        DESCRIPTION.
    average : TYPE
        DESCRIPTION.

    Returns
    -------
    total_score : TYPE
        DESCRIPTION.

    """
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

    total_score = [accuracy, precision, recall, f1]

    return total_score

def vectorizeDoc2Vec(df, model="doc2Vec"):
    corpusTrain = transformDF2Corpus(df)
    vecList = []
    for i in corpusTrain:
        vec = model.infer_vector(i.words)
        vecList.append(vec)
    return np.array(vecList)

# Set working directory
os.chdir(r"C:\analytics")

# Tune NLP Tool
twitter = Twitter()

dictionary = pd.read_csv("dataset6.dictionary.csv", names=["noun"])
nounList = dictionary.noun.to_list()
twitter.add_dictionary(nounList, 'Noun')

stopWord = pd.read_csv("dataset7.stopwords.csv", names=["stopword"])
stopwords = stopWord.stopword.to_list()
postprocessor = Postprocessor(twitter, stopwords = stopwords)

L_Tokenizer = joblib.load('L_tokenizer_auditing.pkl')
M_Tokenizer = joblib.load('M_tokenizer_auditing.pkl')

# import preprocessed dataset
df = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")

# preprocessing
df["twitter"] = tokenizeDocuments(df["documents"], twitter)
df["LToken"] = tokenizeDocuments(df["documents"], L_Tokenizer)
df["MToken"] = tokenizeDocuments(df["documents"], M_Tokenizer, spacing=True)

# # drop blank cells
# drop_index = df[df['document'] == ''].index
# df = df.drop(drop_index)

# # sample and export training data 
# dfLabel = df['document'].sample(n=1000, random_state=1)
# dfLabel.to_excel("dataset4.trainingData.xlsx") 

# # Word Embedding - Counter
# countVec = CountVectorizer()
# countVecMatrix = countVec.fit_transform(df["document"])

# # Word Embedding - TF-IDF
# tfidfVec = TfidfVectorizer()
# tfidfVecMatrix = tfidfVec.fit_transform(df["document"])

# # Word Embedding - LDA
# ldaVec = LatentDirichletAllocation(n_components=10, random_state=1)
# ldaVecMatrix = ldaVec.fit_transform(countVecMatrix)

# # Word Embedding - Doc2Vec 
# doc2Vec = Doc2Vec()
# dfDoc2Vec = df
# train_corpus = transformDF2Corpus(dfDoc2Vec)
# doc2Vec.build_vocab(train_corpus)
# doc2Vec.train(train_corpus, total_examples=doc2Vec.corpus_count, epochs=doc2Vec.epochs)
# doc2VecMatrix = vectorizeDoc2Vec(df)

# # Set Pipeline - NB
# NB = MultinomialNB()

# Pipeline_NB_TfIdf = Pipeline([
#     ('vect', tfidfVec),
#     ('clf', NB)
# 	])

# Pipeline_NB_LDA = Pipeline([
#     ('vect', ldaVec),
#     ('clf', NB)
# 	])

# Pipeline_NB_Doc2Vec = Pipeline([
#     ('vect', doc2Vec),
#     ('clf', NB)
# 	])
