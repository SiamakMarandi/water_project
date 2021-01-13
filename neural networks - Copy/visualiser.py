import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from gaussrank import *
import warnings
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')

def label_write(plt, x_axis, y_axis):
        for x,y in zip(x_axis, y_axis): 
            label = "{:.2f}".format(y)   
            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center



def plotter(model, x_train, y_train, x_true, y_true):
    y_pred = model.predict(x_true)
    x_var = np.arange(0, len(y_true))  
    plt.scatter(x_var, y_true,  color='black', label="original")
    plt.plot(x_var, y_pred, color='blue', linewidth=3, label="predicted")
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.show()   

def computation_range_plotter_mae(df, msg):
    result = df.loc[df.groupby("Computation Range")["Mean Absolout Error"].idxmin()]
    computation_range  = result ['Computation Range'].tolist()
    what_hour = result ['What Hour'].tolist()
    mean_absolout_error = result ['Mean Absolout Error'].tolist()
    plt.plot(computation_range, what_hour, label = 'What Hour',  marker='o', linewidth=2)  

    label_write(plt, computation_range, what_hour)              
    plt.plot(computation_range, mean_absolout_error, label = 'Mean Absolout Error', marker='o', linewidth=2)
    label_write(plt, computation_range, mean_absolout_error)
    plt.xlabel('Computation Range')


    plt.legend()
    plt.xticks(computation_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title(msg)
    plt.show()

def computation_range_plotter_r2(df, msg):
    result = df.loc[df.groupby("Computation Range")["r2_score"].idxmin()]
    computation_range  = result ['Computation Range'].tolist()
    what_hour = result ['What Hour'].tolist()
    r2_score = result ['r2_score'].tolist()
    plt.plot(computation_range, what_hour, label = 'What Hour',  marker='o', linewidth=2)
    label_write(plt, computation_range, what_hour)
    plt.plot(computation_range, r2_score, label = 'r2_score', marker='o', linewidth=2)
    label_write(plt, computation_range, r2_score)
    plt.xlabel('Computation Range')
    plt.legend()
    plt.xticks(computation_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title(msg)
    plt.show()
# gb_plotter(gk, 'What Hour', 'Mean Absolout Error', "Chart of sum of miminum MAE of all device ID")
def gb_plotter(pk, x_lable, y_label, title): 
    gk = pk.groupby([f'{x_lable}'], axis=0).sum() 
    gk.groupby([f'{x_lable}'], axis=0).sum().plot(kind="line", linewidth='2',
                label='MAE',marker="o",
                markerfacecolor="red", markersize=10)  
    # label_write(plt, np.arange(1, 5, 1), gk)
    # plt.xticks(())
    # plt.yticks(())
    # df1({'count' : pk.groupby([f'{x_lable}'], axis=0).sum()}).reset_index()
    df1 = pk.groupby([f'{x_lable}'], axis=0).sum().reset_index()
    # print("df1  :   ", df1)
    label_write(plt, df1[f'{x_lable}'] , df1[f'{y_label}'])
    plt.xlabel(x_lable)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.show()   

