import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/file_20_feat.csv")
print data.shape

Y = data["20"]


seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)
print X_train.shape, X_test.shape, y_train.shape,y_test.shape
X_t = X_train.drop("20",axis =1)
X_tt= X_t.drop("21",axis =1)
X_TRAIN = X_tt.drop("data_set_id",axis =1)
X_t1 = X_test.drop("20",axis =1)
X_tEST= X_t1.drop("21",axis =1)
X_TEST = X_tEST.drop("data_set_id",axis =1)
y_labels = X_test["21"]
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
model.fit(X_TRAIN,y_train)
# fig, ax = plt.subplots(figsize=(100,100))
# xgb.plot_importance(model, ax=ax)
# plt.show()
y_predcited = model.predict(X_TEST)
label = y_predcited
for i in range(len(y_predcited)):
    if(y_predcited[i]<=20000):
        label[i]=1
    if (y_predcited[i]>20000 and y_predcited[i]<=60000):
        label[i]=2
    if(y_predcited[i]>60000 and y_predcited[i]<=300000):
        label[i]=3
    if(y_predcited[i]>300000 and y_predcited[i]<=2000000):
        label[i]=4
    if(y_predcited[i]>2000000 and y_predcited[i]<=1900000000):
        label[i]=5

print accuracy_score(y_labels,label)
