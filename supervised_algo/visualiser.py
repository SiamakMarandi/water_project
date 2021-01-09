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
    plt.plot(computation_range, mean_absolout_error, label = 'Mean Absolout Error', marker='o', linewidth=2)
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
    mean_absolout_error = result ['r2_score'].tolist()
    plt.plot(computation_range, what_hour, label = 'What Hour',  marker='o', linewidth=2)
    plt.plot(computation_range, mean_absolout_error, label = 'r2_score', marker='o', linewidth=2)
    plt.xlabel('Computation Range')
    plt.legend()
    plt.xticks(computation_range)
    # plt.yticks(np.arange(0, len(what_hour) + 1, 1))
    plt.title(msg)
    plt.show()


