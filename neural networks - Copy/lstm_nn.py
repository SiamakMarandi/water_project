import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import dates as mpl_dates
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import dataset
import neural_network_evaluator
import visualiser
import test2
import dataset
import visualiser
import data_picker
import seaborn as sns
from sklearn import metrics
sns.set()

dId = 57975963
year = 2019
month = 6
day = 21
hour = 13
# computation_range = np.arange(1,9, 1)
computation_range = [35]
what_hour = np.arange(1,3, 1)
dataset = dataset.main()
# ========================= filtering the dataset accoring to id, year, month, day, hour and weekend
print("dataset : ", dataset)
print("dataset tail : ", dataset.tail(30))

indexHour = dataset[(dataset['Year'] == year) & (dataset['Month']== month) & 
(dataset["hour"] == hour) & (dataset["Day"] == day)].index
print("index hour   :   ", indexHour)
print("index hour size  :   ", indexHour.size)
try:
    if(int(dataset.loc[indexHour[0]].Is_weekend) == 0):   
        dataset = dataset.loc[dataset['Is_weekend'] == "0"]
        day_type = "Day is Not Weekend"
        # print("filtered datase : ", dataset)
    else:
        dataset = dataset.loc[dataset['Is_weekend'] == "1"]
        day_type = "Day is Weekend"
        # print("filtered datase : ", dataset)

    
    dId_list = dataset.DeviceId.unique()
    # print("dId_list : ", dId_list)
    print("id_conter :  ", dId_list.size)

    # ===============================
    # print("dataset : ", dataset)
    y_dataset = dataset.Value
    x_dataset = dataset.drop(["Value"], axis=1)
    # print("y_dataset : ",y_dataset)
    # print("x_dataset : ", x_dataset)
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

    # print("x_train : ", x_train)
    # print("y_train : ", y_train)
    # print("x_test : ", x_test)
    # print("y_test : ", y_test)
    print("x_train  :   \n", x_train)
    print("x_train shape :   \n", x_train.shape)
    # //////////////////////////////////////
    x_train = x_train.to_numpy()
    print("x_train  :   \n", x_train)
    x_train=np.reshape(x_train,(x_train.shape[0], 7, 1))
    print("x_train shape :   \n", x_train.shape)

    def keras_model(input):
        inputs = keras.Input(shape=(input, 1))
        model = layers.LSTM(12, return_sequences=True)(inputs)
        model = layers.LSTM(12)(model)  
        model = layers.Dense(10)(model)
        outputs = layers.Dense(1)(model)
        model = keras.Model(inputs=inputs, outputs=outputs, name="water_predictor")
        return model


    model = keras_model(7)

    print("output_shape  :   ", model.output_shape)

    # Model summary
    model.summary()

    # ===================================plotting the model as a graph start
    # keras.utils.plot_model(model, "my_first_model.png")
    # keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
    # ===================================plotting the model as a graph end
    # Model config
    # print("get_config  :   ",model.get_config())

    # List all weight tensors 
    # print("get_weights  :   ", model.get_weights())

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape'])  

    # history = model.fit(x_train, y_train,epochs=5, batch_size=50, verbose=1)
    model.fit(x_train, y_train,epochs=5, batch_size=50, verbose=1)
    # //////////////////////////////////////
    print("dataset : ", dataset)
    test2.calculator(model, dataset, dId_list, year, month, day, hour, computation_range, what_hour, dId, day_type)
    # data_picker.calculator(model, dataset, dId_list, year, month, day, hour, computation_range, what_hour, dId, day_type)
except :
    print("Thers is an Error, please select a larger data batch")




