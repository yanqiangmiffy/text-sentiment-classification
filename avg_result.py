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

lr=pd.read_csv('lr.csv')
cnn=pd.read_csv('cnn.csv')

lr['Pred']=lr['Pred']*0.9+cnn['Pred']*0.1 # 0.865
print(lr)

lr.to_csv('en.csv',index=None)