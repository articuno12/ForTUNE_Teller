import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/final_debut_20.csv")
print data.shape

Y = data["label"]


seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)
print X_train.shape, X_test.shape, y_train.shape,y_test.shape
X_t = X_train.drop("views",axis =1)
X_tt= X_t.drop("label",axis =1)
X_TRAIN = X_tt.drop("ID",axis =1)
X_t1 = X_test.drop("views",axis =1)
X_tEST= X_t1.drop("label",axis =1)
X_TEST = X_tEST.drop("ID",axis =1)

xgb_params = {
    'learning_rate' :0.1,
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 1,
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1,
    'n_jobs' : 5,
    'colsample_bytree' : 1,
    'gamma' : 1
}
