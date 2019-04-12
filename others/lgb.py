#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: lgb.py 
@time: 2019-04-12 10:46
@description:
"""

# %%
import time
import re
import numpy as np
import pandas as pd
import warnings;

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# %%
df_train = pd.read_csv('../data/train.csv', lineterminator='\n')
df_test = pd.read_csv('../data/test.csv', lineterminator='\n')
# %%
df_train['label'] = df_train['label'].map({'Negative': 0, 'Positive': 1})
df_train.head()
# %%
df_train.isnull().sum()
# %%
df_train.isnull().sum()
# %%
df_train['label'].value_counts()
# %%
numpy_array = df_train.as_matrix()
numpy_array_test = df_test.as_matrix()
numpy_array[:4]
# %%
numpy_array_test[115]


# %%
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
    return word.lower()


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
    X = []
    for sentence in array:
        clean_sentence = ''
        words = sentence.split(' ')
        for word in words:
            clean_sentence = clean_sentence + ' ' + cleaner(word)
        X.append(clean_sentence)
    return X


# %%
X_test = numpy_array_test[:, 1]
X_test
# %%
# test if there are nan
counter = 1
for sentence in X_test:
    try:
        words = sentence.split(' ')
        counter += 1
    except:
        print(sentence)
        print(counter)
# %%
X_train = numpy_array[:, 1]
# Clean X here
X_train = array_cleaner(X_train)
X_test = array_cleaner(X_test)
y_train = numpy_array[:, 2]
X_train[:5]
# %%
print(len(X_train))
print(len(X_test))
# %%
y_train = np.array(y_train)
y_train = y_train.astype('int8')
print(y_train.shape)
y_train[:6]
# %%
test1 = pd.Series(y_train)
test1.unique()
# %%
ngram = 2
vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.5)
# %%
X_all = X_train + X_test  # Combine both to fit the TFIDF vectorization.
lentrain = len(X_train)

vectorizer.fit(X_all)
X_all = vectorizer.transform(X_all)
# %%
vectorizer.get_feature_names()[-5:]
# %%
X_all.shape
# %%
X_train_chuli = X_all[:lentrain]  # Separate back into training and test sets.
X_test_chuli = X_all[lentrain:]
# %%
X_train_chuli.shape
# %%
# bayesian optimization to find hyperparameter for lightgbm
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from bayes_opt import BayesianOptimization
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


# %%
def LGB_CV(
        min_data_in_leaf,
        feature_fraction,
        bagging_fraction,
):
    folds = KFold(n_splits=5, shuffle=True, random_state=2019)
    oof = np.zeros(X_train_chuli.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_chuli, y_train)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X_train_chuli[trn_idx],
                               label=y_train[trn_idx],
                               )
        val_data = lgb.Dataset(X_train_chuli[val_idx],
                               label=y_train[val_idx],
                               )

        param = {
            'max_depth': -1,
            'min_data_in_leaf': int(min_data_in_leaf),
            'objective': 'binary',
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'learning_rate': 0.005,
            "boosting": "gbdt",
            "bagging_freq": 5,
            "bagging_seed": 11,
            "metric": 'auc',
            "verbosity": -1
        }

        clf = lgb.train(param,
                        trn_data,
                        8000,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds=500)

        oof[val_idx] = clf.predict(X_train_chuli[val_idx],
                                   num_iteration=clf.best_iteration)

        del clf, trn_idx, val_idx

    return metrics.roc_auc_score(y_train, oof)


# %%
LGB_BO = BayesianOptimization(LGB_CV, {
    'min_data_in_leaf': (2, 40),
    'bagging_fraction': (0.01, 0.999),
    'feature_fraction': (0.01, 0.999)
})
# %%
LGB_BO.maximize(init_points=2, n_iter=2)
# %%
folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros(X_train_chuli.shape[0])
predictions = np.zeros(X_test_chuli.shape[0])
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_chuli, y_train)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(X_train_chuli[trn_idx],
                           label=y_train[trn_idx],
                           )
    val_data = lgb.Dataset(X_train_chuli[val_idx],
                           label=y_train[val_idx],
                           )

    param = {
        'max_depth': -1,
        'min_data_in_leaf': 2,
        'objective': 'binary',
        'bagging_fraction': 0.999,
        'feature_fraction': 0.999,
        'learning_rate': 0.005,
        "boosting": "gbdt",
        "bagging_freq": 5,
        "bagging_seed": 11,
        "metric": 'auc',
        "verbosity": -1
    }

    clf = lgb.train(param,
                    trn_data,
                    8000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=500,
                    early_stopping_rounds=500)

    oof[val_idx] = clf.predict(X_train_chuli[val_idx],
                               num_iteration=clf.best_iteration)
    predictions += clf.predict(X_test_chuli, num_iteration=clf.best_iteration) / folds.n_splits
# %%
print(len(predictions))
predictions[:4]
# %%
lgb_output = pd.DataFrame({"ID": df_test["ID"], "Pred": predictions})
lgb_output.to_csv('lgb_new.csv', index=False)
# %%
