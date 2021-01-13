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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import dataset
import neural_network_evaluator
import visualiser
import dataset
import visualiser
import data_picker
import test3
import seaborn as sns
from sklearn import metrics
sns.set()

# dId = 57975963
# year = 2019
# month = 6
# day = 21
# hour = 13
# dId = 57975963
# year = 2019
# month = 6
# day = 19
# hour = 19
dId = 57975964
year = 2019
month = 6
day = 25
hour = 23
computation_range = np.arange(3,10, 1)
# computation_range = [35]
what_hour = np.arange(3,7, 1)
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
    ohe = OneHotEncoder(sparse=False)
    x_dataset = ohe.fit_transform(x_dataset)
    # print("y_dataset : ",y_dataset)
    print("x_dataset : ", x_dataset)
    print("x_dataset size  : ", x_dataset.size)
    print("x_dataset shape  : ", x_dataset.shape)
    y_dataset = y_dataset.values.reshape(-1, 1) 

    # final_dataset = df_filtered.iloc[indexHour - computation_days + what_hour, computation_days]

    # print("start_index : ", start_index)

    # //////////////// normalization
    scaler = StandardScaler()
    # print(y_dataset)
    y_dataset = scaler.fit_transform(y_dataset)

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)

    # print("x_train : ", x_train)
    # print("y_train : ", y_train)
    # print("x_test : ", x_test)
    # print("y_test : ", y_test)
    print("x_train  :   \n", x_train)
    print("x_train shape :   \n", x_train.shape)
    # //////////////////////////////////////
    # x_train = x_train.to_numpy()

    x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    print("x_train shape :   \n", x_train.shape)

    def keras_model(input):
        inputs = keras.Input(shape=(input, 1))
        model = layers.LSTM(12, return_sequences=True)(inputs)
        model = layers.LSTM(12)(model)  
        model = layers.Dense(10)(model)
        outputs = layers.Dense(1)(model)
        model = keras.Model(inputs=inputs, outputs=outputs, name="water_predictor")
        return model


    model = keras_model(x_train.shape[1])

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
    model.fit(x_train, y_train,epochs=2, batch_size=100, verbose=1)
    # //////////////////////////////////////
    # print("dataset : ", dataset)
    # print("dataset shape    :   ", dataset.shape)
    # print("x_dataset shape    :   ", x_dataset.shape)
    # print("y_dataset   :   ", y_dataset)
    # print("y_dataset shape    :   ", y_dataset.shape)
    data_picker.calculator(model, dataset, x_dataset, y_dataset, dId_list, year, month, day, hour, computation_range, what_hour, dId, day_type)
    
except :
    print("Thers is an Error, please select a larger data batch")





