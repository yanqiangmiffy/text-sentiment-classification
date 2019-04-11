#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: create_input.py
@time: 2019-04-11 13:10
@description: 创建输入和输出
"""
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from create_dict import *
import numpy as np

train = pd.read_csv('data/train.csv', lineterminator='\n')
test = pd.read_csv('data/test.csv', lineterminator='\n')
# print(train['review'].apply(lambda x:len(x.split(' '))).describe())
# print(test['review'].apply(lambda x:len(x.split(' '))).describe())

# 设置参数
MAX_LEN = 20


def build_input():
    word_dcit, word_dcit_re = load_dict()
    label_dict, label_dict_re = load_label_dict()
    x_train = []
    y_train = []
    x_test = []
    for sent in train['review']:
        sent = clean_str(sent)
        x = [word_dcit[w] for w in sent.split()]
        x_train.append(x)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post")
    for label in train['label']:
        y = label_dict[label]
        if y==0:
            y_train.append([0,1])
        else:
            y_train.append([1,0])
        # if y==0:
        #     y_train.append([0])
        # else:
        #     y_train.append([1])
    y_train=np.asarray(y_train)
    for sent in test['review']:
        sent = clean_str(sent)
        x = [word_dcit[w] for w in sent.split()]
        x_test.append(x)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post")

    return x_train, y_train, x_test


