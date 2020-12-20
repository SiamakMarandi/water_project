import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def func(model, x_train, y_train):
    RandomForestRegressor_parameters = {
                                        "n_estimators": np.arange(100, 1000, 100),
                                        "criterion": ['mse', 'mae'],
                                        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                        "min_samples_split": [2, 4, 6, 8, 10],                                      
                                        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
                                        "max_features": ['auto', 'sqrt'],
                                        "bootstrap": [True, False],
    }
    

    g_search = GridSearchCV(estimator=model,
                        param_grid=RandomForestRegressor_parameters,
                        cv=2, 
                        n_jobs = -1,
                        verbose = True)
    g_search.fit(x_train, y_train.values.ravel())
    # print("model hypermarateres : ", model.get_params())
    print("Best hyperparameters : ", g_search.best_params_)
    # print(g_search.best_params_["criterion"])
    return g_search.best_params_