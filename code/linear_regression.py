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

data = pd.read_csv("../dataset/final_dataset.csv")
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_1 = data.drop("data_set_id",axis=1)
X = X_1.drop("viewCount",axis=1)
Y = data["viewCount"]
seed = 7
test_size = 0.25
x_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#print X_train, X_test, y_train, y_test

linear_regress = linear_model.Lasso(alpha=0.01, normalize=True)
linear_regress.fit(x_train, y_train)
print linear_regress