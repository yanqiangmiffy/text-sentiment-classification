#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: create_dict.py
@time: 2019-04-11 10:40
@description: 创建单词索引
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import re

# 设置默认字符
sepical_chars = ['_PAD_', '_EOS_', '_SOS_', '_UNK_']
_PAD_ = 0
_EOS_ = 1
_UNK_ = 2
_SOS_ = 3


def clean_str(string):
    """
    数据预处理
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  # 去除数字
    # 标点符号处理
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def create(file_dir, dict_path, label_path):
    """
    创建单词索引，标签索引
    :param file_dir: 训练集和测试集的路径
    :param dict_path: 保存的单词索引路径
    :param label_path: 保存的标签索引路径
    :return: None
    """
    print("save to:", dict_path, label_path)
    word_dict = dict()
    label_dict = dict()
    for index, word in enumerate(sepical_chars):
        word_dict[word] = index
    filenames = os.listdir(file_dir)
    for filename in filenames:
        if filename.endswith('.csv'):
            if filename == 'test.csv':
                df = pd.read_csv(file_dir + filename, lineterminator='\n', usecols=['ID', 'review'])
                # 因为test.csv标签为空，这里先初始化为Positive
                df['label'] = ['Positive'] * len(df)
            else:
                df = pd.read_csv(file_dir + filename, lineterminator='\n', usecols=['ID', 'review', 'label'])
            # 标签索引
            labels = df['label'].values
            labels = labels.astype('str')
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
            # 单词索引
            sentences = df['review'].values
            for sent in sentences:
                sent = clean_str(sent)
                words = sent.split(' ')
                for word in words:
                    if word not in word_dict:
                        word_dict[word] = len(word_dict)
    with open(os.path.join(dict_path), 'w', encoding='utf-8') as fout:
        json.dump(word_dict, fout)
    with open(os.path.join(label_path), 'w', encoding='utf-8') as fout:
        json.dump(label_dict, fout)
    print(len(word_dict))  # 29220
    print("build dict done.")


def load_dict():
    """
    返回单词索引，索引单词 词典
    :return:
    """
    word_dict_re = dict()
    dict_path = 'data/word.dict'
    with open(dict_path, 'r', encoding='utf-8') as fin:
        word_dict = json.load(fin)
    for k, v in word_dict.items():
        word_dict_re[v] = k

    return word_dict, word_dict_re


def load_label_dict():
    label_dict_re = dict()
    dict_path = 'data/label.dict'
    with open(dict_path, 'r', encoding='utf-8') as fin:
        label_dict = json.load(fin)
    for k, v in label_dict.items():
        label_dict_re[v] = k

    return label_dict, label_dict_re


def load_embedding_index():
    """
    加载与训练好的单词向量
    :return:
    """
    embedding_index = dict()
    glove_path = 'data/glove.txt'
    with open(glove_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index


def load_embedding_matrix(vocab, embedding_index, MAX_NUM_WORDS=30000, EMBEDDING_DIM=100):
    """
    构建词嵌入矩阵
    :param vocab: 词典
    :param embedding_index: 词向量字典
    :param MAX_NUM_WORDS: 最大单词数
    :param EMBEDDING_DIM: 词向量维度
    :return:
    """
    # embedding_index = load_embedding_index()
    # vocab, _ = load_dict()
    embedding_matrix = np.random.random((MAX_NUM_WORDS + 1, EMBEDDING_DIM))
    for word, i in vocab.items():
        if i < MAX_NUM_WORDS:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix




if __name__ == '__main__':
    create('data/', 'data/word.dict', 'data/label.dict')
    vocab,_ = load_label_dict()

    label_dict, label_dict_re = load_label_dict()
    print(label_dict, label_dict_re)

    embedding_index = load_embedding_index()
    embedding_matrix=load_embedding_matrix(vocab,embedding_index)
    print(embedding_matrix)
