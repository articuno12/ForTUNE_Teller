import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/final_dataset.csv")
X_1 = data.drop("data_set_id",axis=1)
X = X_1.drop("viewCount",axis=1)
Y = data["viewCount"]
print X.shape, Y.shape
seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print X_train.shape, X_test.shape, y_train.shape,y_test.shape

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

# dtrain = xgb.DMatrix(X_train, y_train)
# # dtest = xgb.DMatrix(X_test)
#
#
# # Training num_boost_round
# cvresult = xgb.cv(xgb_params, dtrain, num_boost_round=1000,nfold=3,metrics='mae', early_stopping_rounds=200,verbose_eval=True)
# print("num_boost_round = ",cvresult.shape[0])
#
# xgb_params['n_estimators'] = cvresult.shape[0]
# print("n_estimators trained to : ", xgb_params['n_estimators'])
#
# param_test = { 'max_depth':[4,5,6], 'min_child_weight':[1,2], 'gamma' : [1,1.2,1.4], 'subsample' : [0.8,1], 'colsample_bytree' : [0.9,1] }
# print("training parameters : ",param_test)
model = xgb.XGBRegressor(learning_rate = xgb_params['eta'], max_depth = 4,
						n_estimators = 1, silent = xgb_params['silent'],
						objective = xgb_params['objective'],
						min_child_weight = 2, gamma = 1,
						subsample = 1, colsample_bytree = 1 )

# gridsearch = GridSearchCV(estimator = model, param_grid = param_test, verbose=10 ,
# 			cv=5, scoring='neg_mean_absolute_error', n_jobs = 6,iid=False)
#
# gridsearch.fit(X_train,y_train)
# print gridsearch.best_params_
# print("gridsearch complete")
# model.fit(X_train,y_train)
# fig, ax = plt.subplots(figsize=(100,100))
# xgb.plot_importance(model, ax=ax)
# plt.show()
y_predcited = model.predict(X_test)
Y_TEST = y_test.values
print(mean_squared_error(Y_TEST,y_predcited))
print(mean_absolute_error(Y_TEST,y_predcited))
# print type(y_predcited)
# for i in range(len(y_predcited)):
#     print i,Y_TEST[i], y_predcited[i]
# print type(Y_TEST)
