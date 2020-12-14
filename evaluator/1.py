
# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomTreesEmbedding
import xgboost as xgb
import sys
sys.path.insert(1, 'H:/Project/water_project/dataset')
import dataset 
#import ../dataset/prepossessed_dataset
from sklearn import linear_model

import warnings

warnings.filterwarnings('ignore')

# preprocessing
# dataset.main()
dataset = dataset.main()

print("x_train  :  ", dataset["x_train"])
print("y_train  :  ", dataset["y_train"])
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]
x_cv = dataset["x_cv"]
y_cv = dataset["y_cv"]
print('Number of data points in train data:', x_train.shape[0])
print('Number of data points in test data:', x_test.shape[0])
print('Number of data points in test data:', x_cv.shape[0])
# ## ###########################



model_factory = [   
    AdaBoostRegressor(n_estimators=100),     
    XGBRegressor(nthread=1),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    Ridge(alpha=1.0),
    SVR(C=1.0, epsilon=0.2),
    GradientBoostingRegressor(),
    Lasso(alpha=0.1),
    BaggingRegressor(),
    StackingRegressor(),
    VotingRegressor(),
    ExtraTreesRegressor(),
    KNeighborsRegressor(),
]


estimators = [
    ('knn', KNeighborsRegressor(n_neighbors=5)),
    ('rfc', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('adab', AdaBoostRegressor(n_estimators=100, random_state=0)),
    ('gb', GradientBoostingRegressor()),
    ('bc', BaggingRegressor(base_estimator=SVC(), n_estimators=10, random_state=0)),
    ('etc', ExtraTreesRegressor()),
    ('hgbc', HistGradientBoostingRegressor()),
    ('xgb', XGBRegressor(nthread=1)),
    ('lasso', Lasso(alpha=0.1)),
    ('ridg', Ridge(alpha=1.0)),
    ]

# #######################################  SVR
clf_svr = SVR(C=1.0, epsilon=0.2)
clf_svr.fit(x_train, y_train)
svr_pred = clf_lasso.predict(x_test)
svr_matrices = evaluate_preds(clf_svr, x_test, y_test, svr_pred)

# #################################################################
# #######################################  Lasso
clf_lasso = Lasso(alpha=0.1)
clf_lasso.fit(x_train, y_train)
lasso_pred = clf_lasso.predict(x_test)
lasso_matrices = evaluate_preds(clf_lasso, x_test, y_test, lasso_pred)

# #################################################################
# #######################################  XGBRegressor
clf_xgb = XGBRegressor(nthread=1)
clf_xgb.fit(x_train, y_train)
xgb_pred = clf_xgb.predict(x_test)
xgb_matrices = evaluate_preds(clf_xgb, x_test, y_test, xgb_pred)

# #################################################################
# #######################################  KNeighborsRegressor
clf_knn = KNeighborsRegressor(n_neighbors=5)
clf_knn.fit(x_train, y_train)
knn_pred = clf_knn.predict(x_test)
knn_matrices = evaluate_preds(clf_knn, x_test, y_test, knn_pred)

# #################################################################
# ################################################ AdaBoostRegressor
clf_adab = AdaBoostRegressor(n_estimators=100, random_state=0)
clf_adab.fit(x_train, y_train)
adab_pred = clf_adab.predict(x_test)
adab_matrices = evaluate_preds(clf_adab, x_test, y_test, adab_pred)
# ################################################ 
# ################################################ RandomForestRegressor
clf_rfc = RandomForestRegressor()
clf_rfc.fit(x_train, y_train)
rfc_pred = clf_rfc.predict(x_test)
rfc_matrices = evaluate_preds(clf_rfc, x_test, y_test, rfc_pred)
# ############################################################
# ################################################ GradientBoostingRegressor
clf_gbc = GradientBoostingRegressor()
clf_gbc.fit(x_train, y_train)
clf_pred = clf_gbc.predict(x_test)
gbc_matrices = evaluate_preds(clf_gbc, x_test, y_test, clf_pred)
# ############################################################
# ############################################################ BaggingRegressor
clf_bc = BaggingRegressor(base_estimator=SVC(), n_estimators=10, random_state=0)
clf_bc.fit(x_train, y_train)
bc_pred = clf_bc.predict(x_test)
bc_matrices = evaluate_preds(clf_bc, x_test, y_test, bc_pred)
# ################################################ ExtraTreesvRegressor
clf_etc = ExtraTreesvRegressor()
clf_etc.fit(x_train, y_train)
etc_pred = clf_etc.predict(x_test)
etc_matrices = evaluate_preds(clf_etc, x_test, y_test, etc_pred)
# ############################################################
# ############################################################ HistGradientBoostingRegressor
clf_hgbc = HistGradientBoostingRegressor()
clf_hgbc.fit(x_train, y_train)
hgbc_pred = clf_hgbc.predict(x_test)
hgbc_matrices = evaluate_preds(clf_hgbc, x_test, y_test, hgbc_pred)
# ############################################################
# ############################################################ Ridge
clf_ridg = Ridge(alpha=1.0)
clf_ridg.fit(x_train, y_train)
ridg_pred = clf_lr.predict(x_test)
ridg_matrices = evaluate_preds(clf_lr, x_test, y_test, ridg_pred)
# ############################################################
# ############################################################ StackingRegressor
clf_sc = StackingRegressor(estimators=estimators,
         final_estimator=RandomForestRegressor(n_estimators=10),
         random_state=42)
clf_sc.fit(x_train, y_train)
clf_pred = clf_sc.predict(x_test)
sc_matrices = evaluate_preds(clf_sc, x_test, y_test, clf_pred)
# ############################################################
# ############################################################   VotingRegressor
clf_vc = VotingRegressor(estimators=[
                            ('ridg', clf_ridg)
                            ('lasso', clf_lasso),
                            ('xgb', clf_xgb),
                            ("knn", clf_knn),
                            ('adab', clf_adab),
                            ('rfc', clf_rfc),
                            ('gnc', clf_gbc),
                            ("bc", clf_bc),
                            ("etc", clf_etc),
                            ("hgbc", clf_hgbc),
                            ("lr", clf_lr)], voting='soft')

clf_vc.fit(x_train, y_train)
clf_pred = clf_vc.predict(x_test)
vc_matrices = evaluate_preds(clf_vc, x_test, y_test, clf_pred)

# ############################################################


compare_matrices = pd.DataFrame({
                                "Lasso": lasso_matrices,
                                "Ridg": ridg_matrices,
                                "KNeighbors": knn_matrices,
                                "RandomForest": rfc_matrices,
                                "AdaBoost" : adab_matrices,
                                "GradientBoos": gbc_matrices,
                                "Bagging": bc_matrices,
                                "ExtraTrees": et_matrices,
                                "HistGradientBoosting": hgb_matrices,
                                "LogisticRegression": lr_matrices,
                                "StackingRegressor": sc_matrices,
                                "VotingRegressor": vc_matrices,
                                "XGBRegressor": xgb_matrices,
                                 })

compare_matrices.plot.bar(rot=0)
plt.show()



