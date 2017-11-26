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

data = pd.read_csv(sys.argv[1])
Y = data["views"]
seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)
#print X_train.shape, X_test.shape, y_train.shape,y_test.shape
X_t = X_train.drop("views", axis =1)
X_tt= X_t.drop("label", axis=1)
X_TRAIN= X_tt.drop("ID", axis=1)
#X_TRAIN = X_ttt.drop("ID", axis=1)
X_t1 = X_test.drop("views", axis=1)
X_tEST= X_t1.drop("label", axis=1)
X_TEST= X_tEST.drop("ID", axis=1)
#X_TEST = X_ts.drop("ID", axis=1)
# x_test = X_TEST["1","2","7","14","18"]
# x_train = X_TRAIN["1","2","7","14","18"]
y_labels = X_test["label"]


ridge_regress = linear_model.Ridge(alpha=0.001)
ridge_regress.fit(X_TRAIN, y_train)
#print ridge_regress
unrounded_y_pred = ridge_regress.predict(X_TEST)
y_pred = np.round(unrounded_y_pred)
#print y_pred
var = y_test - y_pred
inc = sum(abs(var))
flag = inc/len(X_TRAIN)
#print "flag = " 
#print flag
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
	if(y_pred[i]>300000 and y_pred[i]<=3000000):
		label.append(4)
	if(y_pred[i]>3000000):
		label.append(5)

print "labels"

for i in range(len(label)):
	print label[i]

acc_score = sklearn.metrics.accuracy_score(y_labels, label)
print acc_score
cm = sklearn.metrics.confusion_matrix(y_labels, label)
print cm

#linear_model.ElasticNet(alpha=0.1,l1_ratio=0.5)