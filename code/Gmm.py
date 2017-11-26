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
from sklearn import mixture
from numpy import genfromtxt
from math import *
import random

#data = np.genfromtxt('/home/naila/5thSemester/SMAI/Codes/file_20_feat.csv',delimiter = ',')
data = pd.read_csv("file_20_feat.csv")

Y = data["21"]
#print Y[1]

seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)

X_t = X_train.drop("20",axis =1)
X_tt= X_t.drop("21",axis =1)
X_TRAIN = X_tt.drop("data_set_id",axis =1)
X_t1 = X_test.drop("20",axis =1)
X_tEST= X_t1.drop("21",axis =1)
X_TEST = X_tEST.drop("data_set_id",axis =1)
y_labels = X_test["21"]

n_components = 10
cv_type = 'diag'

gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
gmm.fit(X_TRAIN)

predictions = gmm.predict(X_TEST)

#for i in range(len(predictions)):
#    print abs(predictions[i])

acc_score = sklearn.metrics.accuracy_score(y_labels, predictions)
fscore = sklearn.metrics.accuracy_score(y_labels, predictions)
print acc_score,fscore
print predictions
