# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 01:18:18 2021

@author: yoonseok
"""

import pandas as pd
import os

# Set working directory
os.chdir(r"C:\analytics")

# import preprocessed dataset
df = pd.read_excel("dataset3.preprocessed(2017-2019).xlsx", sheet_name="data")
                                       
# drop blank cells
drop_index = df[df['documents'] == ''].index
df = df.drop(drop_index)
df = df.dropna()
df["documents"].str.strip()  # Whitespace 제거
df["count"] = df["documents"].str.len()  # string 수 
df = df[df["count"] > 3]  # 입력내용이 3을 초과하는 입력값

# # sample and export training data 
dfLabel = df[['docID', 'documents']].sample(n=1000, random_state=1)
dfLabel.to_excel("tokenizerEvaluationData.xlsx")
