#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:56:10 2017

@author: garima
"""

from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import numpy as np



from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
import sys
import csv
import sklearn
import pandas as pd
from sklearn import neighbors
from numpy import genfromtxt
from math import *
import random

def svc_param_selection(X, y, nfolds):
    Cs = [ 1,3, 5,10,15,20]
    gammas = [ 0.01,0.05,.1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

# reading dataset
data = pd.read_csv("/Users/garima/Downloads/ForTUNE_Teller-master/dataset/final_debut_50.csv")
Y = data["label"]



seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)

#data preprocesing
X_data1 = data.drop("views",axis =1)
X_data2 = X_data1.drop("label",axis = 1)
X_DATA = X_data2.drop("ID",axis =1)

X_train1 = X_train.drop("views",axis =1)
X_train2 = X_train1.drop("label",axis = 1)
X_TRAIN = X_train2.drop("ID",axis =1)

X_test1 = X_test.drop("views",axis =1)
X_test2 = X_test1.drop("label",axis = 1)
X_TEST = X_test2.drop("ID",axis =1)

y_labels = X_test["label"]

#print (svc_param_selection(X_DATA,Y,5))

# YOUR CODE GOES HERE
clf = svm.SVC(kernel='rbf', C=3, gamma=.01)
clf.fit(X_TRAIN, y_train)

predictions = clf.predict(X_TEST)



acc_score = sklearn.metrics.accuracy_score(y_labels, predictions)
fscore = sklearn.metrics.confusion_matrix(y_labels, predictions)
print (acc_score)
print(fscore)
