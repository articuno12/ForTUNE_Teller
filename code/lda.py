import sklearn
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/final_20.csv")

Y = data["views"]

seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)
#print X_train.shape, X_test.shape, y_train.shape,y_test.shape
X_t = X_train.drop("views",axis =1)
#X_t = X_t.drop("data_set_id", axis=1)
X_tt= X_t.drop("label",axis =1)
X_TRAIN = X_tt.drop("ID",axis =1)
X_t1 = X_test.drop("views",axis =1)
#X_t1 = X_t1.drop("data_set_id", axis=1)
X_tEST= X_t1.drop("label",axis =1)
X_TEST = X_tEST.drop("ID",axis =1)

y_train = X_train["label"]
y_labels = X_test["label"]

clf = LinearDiscriminantAnalysis()
clf.fit(X_TRAIN, y_train)
answer = clf.predict(X_TEST)
print answer
acc_score = sklearn.metrics.accuracy_score(y_labels, answer)
cm = sklearn.metrics.confusion_matrix(y_labels, answer)
print acc_score
print cm