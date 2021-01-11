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

def plotter(history):
    print()
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')  
    # plt.legend(loc='best',fancybox=True, shadow=True)
    plt.show()



