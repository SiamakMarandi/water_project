from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from gaussrank import *
import pandas as pd
import numpy as np
import sys
#sys.path.insert(1, 'H:/Project/water_project/dataset')
import dataset 
dId = 57976040
year = 2019
month = 6
day = 19
hour = 23
computation_range = 10
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

y_df_filtered = df_filtered.loc[:, ["Value"]]
x_df_filtered = df_filtered.loc[:, ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
indexHour = x_df_filtered[(x_df_filtered["DeviceId"] == dId) & 
(x_df_filtered['Year'] == year) & (x_df_filtered['Month']== month) & 
(x_df_filtered["hour"] == hour) & (x_df_filtered["Day"] == day)].index
# print("df_filtered : ", x_df_filtered)
# print("df_filtered : ", y_df_filtered)
ohe = OneHotEncoder(sparse=False)
x_df_filtered = ohe.fit_transform(x_df_filtered)
# indexNames = dataset[(dataset["DeviceId"] == dId)].index
# print("indexNames : ", indexHour)

# final_dataset = df_filtered.iloc[indexHour - computation_days + what_hour, computation_days]
start_index = indexHour[0] - (computation_range + what_hour)
# print("start_index : ", start_index)
x_dataset = x_df_filtered[start_index : start_index + computation_range]
# print("x_dataset : ", x_dataset)
# ===============
y_dataset = y_df_filtered[start_index : start_index + computation_range]
# ===============================
# print("X : ", X)
# print("y : ", y)

x_cols = y_df_filtered.columns[:]
x = y_df_filtered[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform(x)
assert x_.shape == x.shape
y_df_filtered[x_cols] = x_
# ===============
# print('Number of data points in train data:', x)
#-----------------------------------Categorical to Binary-----------------------------------------

# Train and Test (x,y) / shuffle false because of importance roll of date in our study----------------------
# train_x, test_x, train_y, test_y = train_test_split(X_ohe, y, stratify=y, test_size=0.3, shuffle=False)
# #################################
x_predict = x_df_filtered[indexHour[0]]
y_predict = y_df_filtered.iloc[indexHour[0]]
x_predict = x_predict.reshape(1,-1)
y_predict = y_predict.to_frame() 

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
print("x_dataset : ", x_dataset)
print("y_dataset : ", y_dataset)



