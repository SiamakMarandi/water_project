from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from gaussrank import *
import pandas as pd
import numpy as np
import sys
sys.path.insert(1, 'H:/Project/water_project/dataset')
import dataset 
dId = 57976040
year = 2019
month = 6
day = 19
hour = 23
computation_days = 4
what_hour = 1
dataset = dataset.main()
# ======================

# dataset.drop(["Day_of_Week", "Is_weekend"],axis=1, inplace=True)
# print("dataset : ", dataset)
# ========================

df_filtered = dataset[dataset['DeviceId'] == dId]
df_filtered.reset_index(inplace=True, drop=True)

# print("df_filtered : ", df_filtered)

# ============
# indexNames = X[X['Year'] == year and X['Month'] == month and X['Day'] == day and X['DeviceId'] == dId].index
indexHour = df_filtered[(df_filtered["DeviceId"] == dId) & 
(df_filtered['Year'] == year) & (df_filtered['Month']== month) & 
(df_filtered["hour"] == hour) & (df_filtered["Day"] == day)].index

# indexNames = dataset[(dataset["DeviceId"] == dId)].index
# print("indexNames : ", indexHour)

# final_dataset = df_filtered.iloc[indexHour - computation_days + what_hour, computation_days]
start_index = indexHour[0] - (computation_days + what_hour)
# print("start_index : ", start_index)
final_dataset = df_filtered[start_index : start_index + computation_days]
# print("final dataset : ", final_dataset)

X = final_dataset.loc[:, ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
y = final_dataset.loc[:, ["Value"]]

# print("X : ", X)
# print("y : ", y)

x_cols = y.columns[:]
x = y[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform(x)
assert x_.shape == x.shape
y[x_cols] = x_
# print('Number of data points in train data:', x)
#-----------------------------------Categorical to Binary-----------------------------------------
ohe = OneHotEncoder(sparse=False)
X_ohe = ohe.fit_transform(X)
# Train and Test (x,y) / shuffle false because of importance roll of date in our study----------------------
# train_x, test_x, train_y, test_y = train_test_split(X_ohe, y, stratify=y, test_size=0.3, shuffle=False)
# #################################

x_train, x_test, y_train, y_test = train_test_split(X_ohe, y, shuffle=False, test_size=0.2, random_state=42)

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

x_predicted = df_filtered.loc[indexHour[0]:indexHour[0], ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
y_predicted = df_filtered.loc[indexHour[0]:indexHour[0], ["Value"]]
print("x_train : ", x_train)
print("x_predicted : ", x_predicted)
print("y_predicted : ", y_predicted)
x_predicted_ohe = ohe.fit_transform(x_predicted)
print("x_predicted_ohe : ", x_predicted_ohe)


# print("x_train : ", x_train)
# print('Number of data points in train data:', x_train.shape)
# print('Number of data points in test data:', x_test.shape)
# print('Number of data points in test data:', x_cv.shape)