import json
import calendar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import metrics
from numpy import loadtxt
# import xgboost as xgb
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
with open('../dataset/dataset.json') as json_file:
    data = json.load(json_file)
df = pd.read_json(r'../dataset/dataset.json')
df.to_csv (r'../dataset/dataset.csv', index = None)
# create a Data Frame
df1 = df[["DeviceId", "MeasurementTime", "Value"]]
# print 5 first rows
print("the first 5 rows : ", df.head)
print("Description : ", df1.describe())
# print(len(df1))
#-----------Remove null
df1 = df1.dropna()
# print(len(df1))
#-----------Drop the same row with the same value
df1 = df1.drop_duplicates(subset=["DeviceId", "MeasurementTime"], keep='last')
# print(df1)
# print(len(df1))
#Seprate Date and Time------------------------------------------------------
df1['Date'] = pd.to_datetime(df1['MeasurementTime'], utc=True)
df1['Day'] = df1['Date'].dt.day
df1['Month'] = df1['Date'].dt.month
df1['Year'] = df1['Date'].dt.year
df1['hour'] = df1['Date'].dt.hour
df1['Day_of_Week'] = df1['Date'].dt.dayofweek
df1['Is_weekend'] = df1['Day_of_Week'].apply(lambda x: "1" if x == 6 else ("1" if x == 7 else "0"))
# print(df1)
df1 = df1.drop(columns=["Date", "MeasurementTime"])
#------------- Reset the index whenever we change the context of our data frame------
df1 = df1.reset_index(drop=True)
X = df1.loc[:, ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
y = df1.loc[:, ["Value"]]
# print(X)
# print(y)
#--------- create Lag> use the target value for feature engineering--------------------
df1['lag_1'] = df1.groupby(['DeviceId'])['Value'].shift(1)
df1 = df1[['DeviceId', 'Day', 'Month', 'Year', 'hour', 'Day_of_Week', 'Is_weekend', 'Value', 'lag_1']]
# print(df1)
# the number of devices (Sensors)
print(df1["DeviceId"].nunique())
# Investigation the effect of other features like the weekends and day of work
df1['lag_day_mean'] = df1.groupby(['DeviceId', 'Day_of_Week'])['Value'].apply(lambda x: x.expanding().mean().shift())
df1 = df1.dropna()
df1 = df1.reset_index(drop=True)
# print(df1.head(300).to_string())
# print(df1[54000: 55000].to_string())
#-----------------------------------Categorical to Binary-----------------------------------------
ohe = OneHotEncoder(sparse=False)
X_ohe = ohe.fit_transform(X)
# print(X_ohe)
# Train and Test (x,y) / shuffle false because of importance roll of date in our study----------------------
train_x, test_x, train_y, test_y = train_test_split(X_ohe, y, test_size=0.3, shuffle=False)
# print(train_x)
# print(train_y)
# print(test_x)
# print(test_y)
#----------------------------- Linear Regression & training the algorithm----------------------
reg = LinearRegression().fit(X_ohe, y)
y_pred = reg.predict(test_x)
#variance score: 1 means perfect prediction
print('Reg Variance score: {}'.format(reg.score(test_x, test_y)))
print('Reg.Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
print('Reg.Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
print('Reg.Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
print('intercept:', reg.intercept_)
print('predicted response:', y_pred, sep='\n')
print('slope:', reg.coef_)
#print('Reg.Accuracy Score:', metrics.explained_variance_score(test_y, y_pred))
"""
#-----------------------------------------XgBoost model-----------------------------------
model = xgb.DMatrix(data=X_ohe, label=y)
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)
xg_reg.fit(train_x, train_y)
xgb_pred = xg_reg.predict(test_x)
xgb_rmse = np.sqrt(mean_squared_error(test_y, xgb_pred))
print("XGB.RMSE: %f" % (xgb_rmse))
##---------k-fold Cross Validation using XGBoost:--------------
a_dmatrix = xgb.DMatrix(data=X_ohe, label=y)
params = {"objective":"reg:squarederror", 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
"""