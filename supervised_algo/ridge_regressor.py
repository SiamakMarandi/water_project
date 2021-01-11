import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
import dataset
import evaluator
import visualiser
import data_picker
import hyperparameter_tuning
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics
sns.set()

dId = 57975963
year = 2019
month = 6
day = 19
hour = 14
computation_range = np.arange(1,5, 1)
# computation_range = [5, 10, 17, 35, 55, 78, 1002]
what_hour = np.arange(1,3, 1)
dataset = dataset.main()
# ========================= filtering the dataset accoring to id, year, month, day, hour and weekend
print("dataset : ", dataset)
indexHour = dataset[(dataset['Year'] == year) & (dataset['Month']== month) & 
(dataset["hour"] == hour) & (dataset["Day"] == day)].index
print("index hour   :   ", indexHour)
try:
    if(int(dataset.loc[indexHour[0]].Is_weekend) == 0):   
        dataset = dataset.loc[dataset['Is_weekend'] == "0"]
        day_type = "Day is Not Weekend"
        # print("filtered datase : ", dataset)
    else:
        dataset = dataset.loc[dataset['Is_weekend'] == "1"]
        day_type = "Day is Weekend"
        # print("filtered datase : ", dataset)

    print("dataset : ", dataset)
    dId_list = dataset.DeviceId.unique()
    # print("dId_list : ", dId_list)

    # ===============================
    # print("dataset : ", dataset)
    y_dataset = dataset.Value
    x_dataset = dataset.drop(["Value"], axis=1)
    # print("y_dataset : ",y_dataset)
    # print("x_dataset : ", x_dataset)
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

    # print("x_train : ", x_train)
    # print("y_train : ", y_train)
    # print("x_test : ", x_test)
    # print("y_test : ", y_test)
    clf = Ridge(alpha=1.0)
    # ////////////////////////////////////////  hyperparameter tuning
    params = hyperparameter_tuning.ridge_hyperparameter_tuner(clf, x_train, y_train)
    clf.set_params(**params)
    # ////////////////////////////////////////
    clf.fit(x_train, y_train)
    # print("device id list", dId_list)
    # print("dataset : , ", dataset)

    data_picker.calculator(clf, dataset, dId_list, year, month, day, hour, computation_range, what_hour, dId, day_type)
except :
    print("Thers is an Error, please select a larger data batch")




