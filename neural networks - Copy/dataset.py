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
    # df = pd.read_csv('dataset_50.csv') 
    print("dataset description is : \n", df.describe())
    df = df[:5000]
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
    df1['Is_weekend'] = df1['Day_of_Week'].apply(lambda x: "1" if x == 6 or x == 5 else ("1" if x == 7 else "0"))
    df1 = df1.drop(columns=["Date", "MeasurementTime"])
    
    #------------- Reset the index whenever we change the context of our data frame------
    df1 = df1.reset_index(drop=True)

    return df1    
    
 


if __name__ == "__main__":
    main()




