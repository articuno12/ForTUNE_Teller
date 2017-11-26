import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt
import sys

data = pd.read_csv(sys.argv[1])
print data.shape

Y = data["views"]


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
# x_test = X_TEST["1","2","7","14","18"]
# x_train = X_TRAIN["1","2","7","14","18"]
y_labels = X_test["label"]
print X_TEST.shape,X_TRAIN.shape
from sklearn import preprocessing

x = X_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_N_train = pd.DataFrame(x_scaled)
x_n = X_test.values
x_scaled = min_max_scaler.fit_transform(x_n)
X_N_test = pd.DataFrame(x_scaled)

y_mean = np.mean(y_train)
xgb_params = {
    'learning_rate' :0.1,
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 1,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1,
    'n_jobs' : 5,
    'colsample_bytree' : 1,
    'gamma' : 1
}



param_test = { 'learning_rate':[0.08,0.05,0.1],'n_estimators':[500,800],'max_depth':[5,10], 'min_child_weight':[0.8,1], 'gamma' : [0.01,0.05], 'subsample' : [0.8,0.5], 'colsample_bytree' : [1,0.5,0.8] }
print("training parameters : ",param_test)
model = xgb.XGBRegressor(learning_rate = xgb_params['eta'], max_depth = xgb_params['max_depth'],
						n_estimators = xgb_params['n_estimators'], silent = xgb_params['silent'],
						objective = xgb_params['objective'],
						min_child_weight = 0.1, gamma = xgb_params['gamma'],
						subsample = xgb_params['subsample'], colsample_bytree = xgb_params['colsample_bytree'] )

gridsearch = GridSearchCV(estimator = model, param_grid = param_test, verbose=10 ,
			cv=5, scoring='neg_mean_absolute_error', n_jobs = 5,iid=False)

gridsearch.fit(X_train,y_train)
print gridsearch.best_params_
print("gridsearch complete")
