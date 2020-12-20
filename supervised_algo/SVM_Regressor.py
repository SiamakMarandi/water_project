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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import dataset
import evaluator
import visualiser
import hyperparameter_tuning
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics
sns.set()

dataset = dataset.main()
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]
x_cv = dataset["x_cv"]
y_cv = dataset["y_cv"]

clf = SVR(kernel = 'rbf')
clf.fit(x_train, y_train)
# evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv)
# visualiser.plotter(clf, x_train, y_train, x_test, y_test)
params = hyperparameter_tuning.func(clf, x_train, y_train)
# print(RandomForestRegressor().get_params())
clf.set_params(**params)
evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv)


