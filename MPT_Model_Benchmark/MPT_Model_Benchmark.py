import os
import sys
for p in ['numpy','pandas','xlrd', 'XlsxWriter', 'scikit-learn']:
    if not p in sys.modules:
        print('Install {}'.format(p))
        pip = lambda: os.system('pip install {p}'.format(p=p))
        pip()

from sklearn.dummy import DummyClassifier
import pandas as pd
from numpy.random import choice
from sklearn.metrics import f1_score

# load the true labels
y_true = pd.read_excel('Training_Set.xlsx')['Category']
X = y_true

def F1(y_true,y_pred, type = None):
    """
    :param y_true: list containing the true labels
    :param y_pred: list containing the predicted labels
    :param type:   string containig the description of the benchmark classifier
    :return: None
    """
    print('f1_micro of {type} Classifier: {f1}'.format(type = type,
           f1 = round(f1_score(y_true = y_true, y_pred = y_pred, labels = [0,1,2,3,4], average = 'micro'),3)))


# Create Graph
for strat in ['most_frequent','stratified']:
    F1(y_true, DummyClassifier(strategy=strat, constant = 2).fit(X, y_true).predict(X), strat)

# Classify all as either the most or second most frequent class -> 0 (Prob: 0.504) or 2 (Prob: 0.496
F1(y_true,choice(a = [0,2], size = X.shape[0], p=[0.504,0.496]), 'Most two frequent classes')
