#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: avg_result.py
@time: 2019-04-11 15:33
@description:
"""
import pandas as pd

# lr=pd.read_csv('result/lr.csv')
cnn=pd.read_csv('result/cnn.csv')
rnn=pd.read_csv('result/bi_lstm.csv')
att=pd.read_csv('result/attention.csv')
sgd=pd.read_csv('result/SGD_new.csv')
lstm=pd.read_csv('result/lstm_new.csv')
lgb=pd.read_csv('result/lgb_new.csv')
# lr['Pred']=lr['Pred']*0.7+cnn['Pred']*0.3 # 0.865
# lr['Pred']=lr['Pred']*0.6+cnn['Pred']*0.1+att['Pred']*0.2+rnn['Pred']*0.1 # 0.8665
# lr['Pred']=lr['Pred']*0.7+att['Pred']*0.3 # 0.86634
# lr['Pred']=lr['Pred']*0.2+att['Pred']*0.1+sgd['Pred']*0.6+cnn['Pred']*0.1 # 0.87151058
lgb['Pred']=lgb['Pred']*0.2+att['Pred']*0.1+sgd['Pred']*0.5+cnn['Pred']*0.1 +0.1*lstm['Pred']# 0.87304005
lgb['Pred']=lgb['Pred'].round(2)
lgb.to_csv('result/en.csv',index=None)