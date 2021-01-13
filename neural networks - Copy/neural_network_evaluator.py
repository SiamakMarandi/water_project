import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
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
# tf.keras.losses.MeanSquaredError(
#     reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_error'
# )
from gaussrank import *
import warnings
import visualiser
# import data_picker
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


def evaluate_ann(history, model, x_train, y_train, x_true, y_true, x_cv, y_cv, x_predict):
    print("x_true : ", x_true)
    print("y_true : ", y_true)
    print("x_train : ", x_train)
    print("y_train : ", y_train)
    x_true = x_true.to_numpy()  
    x_true=np.reshape(x_true,(x_true.shape[0], 7, 1))   
    # x_predict = x_predict.to_numpy()
    x_predict = x_predict.ravel()
    print("x_predict shape :", x_predict.shape)
    x_predict = x_predict.reshape(1, -1)
    print("x_predict shape :", x_predict.shape)
    x_predict = np.reshape(x_predict,(x_predict.shape[0], 7, 1))  
    y_pred = model.predict(x_true)    
    pred = model.predict(x_predict) 
    history = model.fit(x_train, y_train, epochs=20, batch_size=len(x_train), verbose=2)
    # mse_value, mae_value = model.evaluate(x_true, y_true, verbose=0)    
    # print("y_pred : ", y_pred)
    print("Prediction : ", pred)    
    print("R² score, the coefficient of determination  : ", r2_score(y_true, y_pred))
    print("history keis :   ", history.history.keys())
    print("Loss value :   ", history.history['loss'])
    print("Mean Squared Erro  :   ", history.history['mean_squared_error'])
    print("Mean Absolute Error  :   ", history.history['mean_absolute_error'])
    print("Mean Absolute Percentage Error  :   ", history.history['mean_absolute_percentage_error'])
    # print("Cosine Proximity  :   ", history.history['cosine'])
    
    # plot metrics
    visualiser.plotter(history)
 
    eval_dict = {           
        "history": history,            
        "predicted_value": pred,
    }

    return  eval_dict


