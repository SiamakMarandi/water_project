import pandas as pd
import numpy as np
import math
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
import data_picker
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


def evaluate_preds(model, x_train, y_train, x_true, y_true, x_cv, y_cv, x_predict):
    y_pred = model.predict(x_true)    
    pred = model.predict(x_predict) 
    # print("y_pred : ", y_pred)
    # print("pred : ", pred)
    print("pred : ", pred[0])
    print("Name of the kernel : ", model)
    print('Model Variance score: {}'.format(model.score(x_true, y_true)))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_true, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_true, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    print("explained variance regression score : ", explained_variance_score(y_true, y_pred))
    print("Max error : ", max_error(y_true, y_pred))
    print("R² score, the coefficient of determination  : ", r2_score(y_true, y_pred))  


    """
    metric_dict = {
                   "Mean Absolute Error": round(metrics.mean_absolute_error(y_true, y_pred), 3),
                   "Root Mean Squared Error": round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 3),                   
                   "R-squered": round(r2_score(y_true, y_pred), 3),
                    "Explained variance score": round(explained_variance_score(y_true, y_pred), 3)}
    
    """
    
    # ==================
    """
    cross_validation_score = cross_val_score(model, x_train, y_train, cv=2)
    print("Cross validation score : ", cross_validation_score)
    cross_validation_predict = cross_val_predict(model, x_train, y_train, cv=2)
    # print("Cross validation predict : ", cross_validation_predict)
    cross_val_accuracy = np.mean(cross_validation_score) * 100
    print("cross validation accuracy : ", cross_val_accuracy)
    #return metric_dict
   """
 
    eval_dict = {
        "mean_absolute_error": metrics.mean_absolute_error(y_true, y_pred),               
        "predicted_value": pred,
        "r2_score": r2_score(y_true, y_pred),
    }

    return  eval_dict


