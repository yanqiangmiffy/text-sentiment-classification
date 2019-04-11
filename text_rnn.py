#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: text_rnn.py 
@time: 2019-04-11 22:23
@description: RNN实现文本分类
"""
import keras
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras.layers import Conv1D, MaxPool1D, SpatialDropout1D, Dropout
from keras.layers import Bidirectional, LSTM, GRU
from create_dict import *
from create_input import build_input
from matplotlib import pyplot

# 超参数设置
# MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100
MAX_LEN = 20
CLASS_NUM = 2

# 加载数据
vocab, vocab_re = load_dict()
print(len(vocab))
label_dict, label_dict_re = load_label_dict()
embedding_index = load_embedding_index()
embedding_matrix = load_embedding_matrix(vocab, embedding_index, len(vocab), EMBEDDING_DIM)
# print(embedding_matrix)

x_train, y_train, x_test = build_input()

input = Input(shape=(MAX_LEN,), dtype="int32", name="input")
embed = Embedding(input_dim=len(vocab) + 1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix],
                  name="embedding_layer")(input)
embed = SpatialDropout1D(0.25)(embed)
bi_listm = Bidirectional(LSTM(units=100), name="bi_lstm")(embed)
bi_listm = Dropout(0.5)(bi_listm)

output = Dense(units=CLASS_NUM, activation='softmax', name='softmax')(bi_listm)
model = Model(input, output)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("Bidirectional LSTM")
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, validation_split=0.1)  # starts training

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred[:, 1])
test_df = pd.read_csv('data/test.csv', lineterminator='\n')
test_df['Pred'] = y_pred[:, 1]
test_df[['ID', 'Pred']].to_csv('resutl/bi_lstm.csv', index=None)
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
