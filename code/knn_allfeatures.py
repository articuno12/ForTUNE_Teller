import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
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

#data = np.genfromtxt('/home/naila/5thSemester/SMAI/Codes/file_20_feat.csv',delimiter = ',')
arg = sys.argv[1]
data = pd.read_csv(arg)

Y = data["viewCount"]


seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)
#print X_train.shape, X_test.shape, y_train.shape,y_test.shape
X_t = X_train.drop("viewCount",axis =1)
X_t = X_t.drop("data_set_id", axis=1)
X_tt= X_t.drop("label",axis =1)
X_TRAIN = X_tt.drop("ID",axis =1)
X_t1 = X_test.drop("viewCount",axis =1)
X_t1 = X_t1.drop("data_set_id", axis=1)
X_tEST= X_t1.drop("label",axis =1)
X_TEST = X_tEST.drop("ID",axis =1)

y_train = X_train["label"]
y_labels = X_test["label"]

n_neighbors = 7

model = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
model.fit(X_TRAIN, y_train)

predictions = model.predict(X_TEST)

#for i in range(len(predictions)):
#    print abs(predictions[i])

acc_score = sklearn.metrics.accuracy_score(y_labels, predictions)
cm = sklearn.metrics.confusion_matrix(y_labels, predictions)
print acc_score
print cm
#print y_train
