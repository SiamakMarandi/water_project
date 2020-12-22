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
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import dataset
import evaluator
import visualiser
import data_picker
import hyperparameter_tuning
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics
sns.set()

x_train = data_picker.x_train
y_train = data_picker.y_train
x_test = data_picker.x_test
y_test = data_picker.y_test
x_cv = data_picker.x_cv
y_cv = data_picker.y_cv

clf = RandomForestRegressor(max_depth=3, random_state=42, n_jobs=-1)
params = hyperparameter_tuning.rfr_hyperparameter_tuner(clf, x_train, y_train)
# print(RandomForestRegressor().get_params())
clf.set_params(**params)
clf.fit(x_train, y_train)
evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv)
visualiser.plotter(clf, x_train, y_train, x_test, y_test)



