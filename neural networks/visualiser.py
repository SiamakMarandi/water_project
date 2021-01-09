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



