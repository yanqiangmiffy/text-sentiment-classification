#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: text_cnn.py
@time: 2019-04-11 10:30
@description: 实现TextCNN文本分类
"""
import keras
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Embedding,Flatten,Concatenate,Dense
from keras.layers import Conv1D,MaxPool1D,SpatialDropout1D,Dropout,BatchNormalization
from create_dict import *
from create_input import build_input
from matplotlib import pyplot
# 超参数设置
# MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100
MAX_LEN=20
CLASS_NUM=2

# 加载数据
vocab, vocab_re = load_dict()
print(len(vocab))
MAX_NUM_WORDS=len(vocab)
label_dict, label_dict_re = load_label_dict()
embedding_index = load_embedding_index()
embedding_matrix = load_embedding_matrix(vocab, embedding_index,MAX_NUM_WORDS,EMBEDDING_DIM)
# print(embedding_matrix)

x_train,y_train,x_test=build_input()

# 创建模型
input=Input(shape=(MAX_LEN,),dtype="int32",name="input")
# Embedding layer
embed=Embedding(input_dim=MAX_NUM_WORDS+1,output_dim=EMBEDDING_DIM,weights=[embedding_matrix],name="embed_layer")(input)
embed=SpatialDropout1D(0.25)(embed)
# Conv layer
convs=[]
filter_sizes=[2,3,4,5]
for fs in filter_sizes:
    l_conv=Conv1D(filters=128,kernel_size=fs)(embed)
    l_conv=BatchNormalization()(l_conv)
    l_conv=Dropout(0.25)(l_conv)

    l_pool=MaxPool1D(MAX_LEN-fs+1)(l_conv)
    l_pool=Flatten()(l_pool)
    convs.append(l_pool)

merge=Concatenate()(convs)
output=Dropout(0.25)(merge)
output=Dense(units=128,activation='relu')(output)
output=Dropout(0.25)(output)
output=Dense(units=2,activation='softmax')(output)

model=Model([input],output)
model.summary()
model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['acc'])

checkpoint = ModelCheckpoint('data/output/model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
print("Traning Model...")
history=model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, callbacks=[checkpoint],validation_split=0.2)  # starts training

y_pred=model.predict(x_test)
print(y_pred)
print(y_pred[:,1])
test_df=pd.read_csv('data/test.csv',lineterminator='\n')
test_df['Pred']=y_pred[:,1]
test_df[['ID','Pred']].to_csv('result/attention.csv',index=None)
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()