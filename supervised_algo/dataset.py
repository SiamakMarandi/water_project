import pandas as pd
import numpy as np
import sklearn
# from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from gaussrank import *
import seaborn as sns
sns.set()


def main():

    # df = pd.read_json('dataset_10.json')
    df = pd.read_csv('H:/Project/water_project/dataset/dataset_50.csv') 
    print("dataset description is : \n", df.describe())
    df = df[:500]
    # create a Data Frame
    df1 = df[["DeviceId", "MeasurementTime", "Value"]]
    
    #-----------Remove null
    df1 = df1.dropna()
    #-----------Drop the same row with the same value
    df1 = df1.drop_duplicates(subset=["DeviceId", "MeasurementTime"], keep='last')   
    #Seprate Date and Time------------------------------------------------------
    df1['Date'] = pd.to_datetime(df1['MeasurementTime'], utc=True)
    df1['Day'] = df1['Date'].dt.day
    df1['Month'] = df1['Date'].dt.month
    df1['Year'] = df1['Date'].dt.year
    df1['hour'] = df1['Date'].dt.hour
    df1['Day_of_Week'] = df1['Date'].dt.dayofweek
    df1['Is_weekend'] = df1['Day_of_Week'].apply(lambda x: "1" if x == 6 else ("1" if x == 7 else "0"))
    df1 = df1.drop(columns=["Date", "MeasurementTime"])
    
    #------------- Reset the index whenever we change the context of our data frame------
    df1 = df1.reset_index(drop=True)
    X = df1.loc[:, ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
    y = df1.loc[:, ["Value"]]
    print("dataset description is : \n", df1.describe())
    #--------- create Lag> use the target value for feature engineering--------------------
    df1['lag_1'] = df1.groupby(['DeviceId'])['Value'].shift(1)
    df1 = df1[['DeviceId', 'Day', 'Month', 'Year', 'hour', 'Day_of_Week', 'Is_weekend', 'Value', 'lag_1']]
    
    # the number of devices (Sensors)
    print("number of devices : ", df1["DeviceId"].nunique())
    # Investigation the effect of other features like the weekends and day of work
    df1['lag_day_mean'] = df1.groupby(['DeviceId', 'Day_of_Week'])['Value'].apply(lambda x: x.expanding().mean().shift())
    df1 = df1.dropna()
    df1 = df1.reset_index(drop=True)
       ########################################### APPLYING GUASSRANK NORMALIZATION

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
    # print("x_train : ", x_train)
    # print('Number of data points in train data:', x_train.shape)
    # print('Number of data points in test data:', x_test.shape)
    # print('Number of data points in test data:', x_cv.shape)

    data_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_cv": x_cv,
        "y_cv": y_cv

    }

    return data_dict


if __name__ == "__main__":
    main()


