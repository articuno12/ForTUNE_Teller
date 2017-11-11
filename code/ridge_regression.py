import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
import sys
import csv
import sklearn
from sklearn import linear_model
from numpy import genfromtxt
from math import *
import random

data = pd.read_csv("../dataset/file_20_feat.csv")
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#X_1 = data.drop("data_set_id",axis=1)
#drop_label = X_1.drop("20",axis=1)
#X = drop_label.drop("21",axis=1)
#Y = data["20"]
#seed = 7
#test_size = 0.25
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#print X_train, X_test, y_train, y_test

Y = data["20"]
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

ridge_regress = linear_model.Ridge(alpha=0.000000001)
ridge_regress.fit(X_TRAIN, y_train)
#print ridge_regress
unrounded_y_pred = ridge_regress.predict(X_TEST)
y_pred = np.round(unrounded_y_pred)
#print y_pred
var = y_test - y_pred
inc = sum(abs(var))
flag = inc/len(X_TRAIN)
print "flag = " 
print flag
final_acc = 1 - flag
for i in range(len(y_pred)):
    print abs(y_pred[i])
label = []

for i in range(len(y_pred)):
	if(y_pred[i]<=20000):
		label.append(1)
	if(y_pred[i]>20000 and y_pred[i]<=60000):
		label.append(2)
	if(y_pred[i]>60000 and y_pred[i]<=300000):
		label.append(3)
	if(y_pred[i]>300000and y_pred[i]<=2000000):
		label.append(4)
	if(y_pred[i]>2000000 and y_pred[i]<=1900000000):
		label.append(5)

print "labels"

#for i in range(len(label)):
#	print i, label[i], y_labels.values[i]

acc_score = sklearn.metrics.accuracy_score(y_labels, label)
print acc_score