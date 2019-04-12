#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: preprocess.py 
@time: 2019-04-12 09:47
@description: 数据预处理
"""
import re
import pandas as pd
from tqdm import tqdm



# two commom ways to clean data
def cleaner(word):
    word = re.sub(r'\#\.', '', word)
    word = re.sub(r'\n', '', word)
    word = re.sub(r',', '', word)
    word = re.sub(r'\-', ' ', word)
    word = re.sub(r'\.', '', word)
    word = re.sub(r'\\', ' ', word)
    word = re.sub(r'\\x\.+', '', word)
    word = re.sub(r'\d', '', word)
    word = re.sub(r'^_.', '', word)
    word = re.sub(r'_', ' ', word)
    word = re.sub(r'^ ', '', word)
    word = re.sub(r' $', '', word)
    word = re.sub(r'\?', '', word)
    word = re.sub(r'é', '', word)
    word = re.sub(r'§', '', word)
    word = re.sub(r'¦', '', word)
    word = re.sub(r'æ', '', word)
    word = re.sub(r'\d+', '', word)
    word = re.sub('(.*?)\d+(.*?)', '', word)
    return word.lower().strip()


def hashing(word):
    word = re.sub(r'ain$', r'ein', word)
    word = re.sub(r'ai', r'ae', word)
    word = re.sub(r'ay$', r'e', word)
    word = re.sub(r'ey$', r'e', word)
    word = re.sub(r'ie$', r'y', word)
    word = re.sub(r'^es', r'is', word)
    word = re.sub(r'a+', r'a', word)
    word = re.sub(r'j+', r'j', word)
    word = re.sub(r'd+', r'd', word)
    word = re.sub(r'u', r'o', word)
    word = re.sub(r'o+', r'o', word)
    word = re.sub(r'ee+', r'i', word)
    if not re.match(r'ar', word):
        word = re.sub(r'ar', r'r', word)
    word = re.sub(r'iy+', r'i', word)
    word = re.sub(r'ih+', r'eh', word)
    word = re.sub(r's+', r's', word)
    if re.search(r'[rst]y', 'word') and word[-1] != 'y':
        word = re.sub(r'y', r'i', word)
    if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):
        word = re.sub(r'i$', r'y', word)
    if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):
        word = re.sub(r'h', '', word)
    word = re.sub(r'k', r'q', word)
    return word


def array_cleaner(array):
    # X = array
    print("processing sentences...")
    X = []
    for sentence in tqdm(array):
        clean_sentence = ''
        sentence=" ".join([ sent for sent in sentence.split() if sent])
        sentence=sentence.replace('\n','')
        sentence=sentence.lower()
        words = sentence.split(' ')
        # for word in words:
        #     if word!='':
        #         clean_sentence = clean_sentence + ' ' + cleaner(word)
        words=[cleaner(word) for word in words]
        words=[word.strip().replace('\n','') for word in words if word ]
        clean_sentence=" ".join(words)
        X.append(clean_sentence.strip())
    return X


train = pd.read_csv('data/train.csv', lineterminator='\n')
test = pd.read_csv('data/test.csv', lineterminator='\n')
train['review']=array_cleaner(train['review'])
train.to_csv('data/input/new_train.csv',index=None)
test['review']=array_cleaner(test['review'])
test.to_csv('data/input/new_test.csv',index=None)
