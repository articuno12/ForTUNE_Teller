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

Y = data["viewCount"]


seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=seed)
print X_train.shape, X_test.shape, y_train.shape,y_test.shape
X_t = X_train.drop("viewCount",axis =1)
X_tt= X_t.drop("label",axis =1)
X_ttt= X_tt.drop("data_set_id",axis =1)
X_TRAIN = X_ttt.drop("ID",axis =1)
X_t1 = X_test.drop("viewCount",axis =1)
X_tEST= X_t1.drop("label",axis =1)
X_ts= X_tEST.drop("data_set_id",axis =1)
X_TEST = X_ts.drop("ID",axis =1)
# x_test = X_TEST["1","2","7","14","18"]
# x_train = X_TRAIN["1","2","7","14","18"]
y_labels = X_test["label"]
print X_TEST.shape,X_TRAIN.shape


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

model = xgb.XGBRegressor(learning_rate = 0.03, max_depth = 12,
						n_estimators = 800, silent = xgb_params['silent'],
						objective = xgb_params['objective'],
						min_child_weight = 1, gamma = 0.01 ,
						subsample = 0.8, colsample_bytree = 1 )

model.fit(X_TRAIN,y_train)
print model.feature_importances_
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
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
print confusion_matrix(y_labels, label)
