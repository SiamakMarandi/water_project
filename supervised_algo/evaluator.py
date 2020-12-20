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


def evaluate_preds(model, x_train, y_train, x_true, y_true, x_cv, y_cv):
    y_pred = model.predict(x_true)   
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
    
    cross_validation_score = cross_val_score(model, x_train, y_train, cv=2)
    print("Cross validation score : ", cross_validation_score)
    cross_validation_predict = cross_val_predict(model, x_train, y_train, cv=2)
    # print("Cross validation predict : ", cross_validation_predict)
    cross_val_accuracy = np.mean(cross_validation_score) * 100
    print("cross validation accuracy : ", cross_val_accuracy)
    #return metric_dict
   
    elbo = []
    list_k = list(range(2, 5))

    for k in list_k:
        cross_validation_score = cross_val_score(model, x_cv, y_cv, cv=k)        
        elbo.append(cross_validation_score[-1])

    cross_validation_predict = cross_val_predict(model, x_train, y_train, cv=k)
    # print("Cross validation predict : ", cross_validation_predict)
    print("elbo : ", elbo)
    
    # Plot 
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, elbo, '-o')
    plt.xlabel(r'Number of croos validation')
    plt.ylabel('cross val score')
    plt.show()




