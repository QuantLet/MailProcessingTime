[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **MPT_Model_Benchmark** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : MPT_Model_Benchmark

Published in : Masterthesis 'Estimation of Email Processing Time in the Financial Services Industry'

Description : 'Random Benchmarks to evaluate the trained Email Processing Time model. 3 different Benchmarks were applied. A stratified random classifier, a classifier always selecting the most frequent class and a classifier selecting the two most frequent classes based on their probabilities.'

Keywords : 'Machine Learning (ML), NLP, Textual Analysis, Word Embeddings, SIF Embedding, FastText, Dummy Model Performance Benchmark'


Author : Marvin Gauer

Submitted : Marvin Gauer

Datafile : 

Input:
  - Training_Set.xlsx : Training data incl. the true category respectively label  

Output: 
  - Prints the micro averaged f1 score of the corresponding benchmark

```

### PYTHON Code
```python

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

```

automatically created on 2019-06-21