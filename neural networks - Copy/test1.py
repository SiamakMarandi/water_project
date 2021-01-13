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
# import data_picker
dId = 57975963
year = 2019
month = 6
day = 21
hour = 13



computation_range = np.arange(1,5, 1)
what_hour = np.arange(1,3, 1)
dataset = dataset.main()

# x_predict = x_predict.values.ravel()
# print("x_predict shape :", x_predict.shape)
# ========================= filtering the dataset accoring to id, year, month, day, hour
# print("dataset : ", dataset)
dId_list = dataset.DeviceId.unique()
# print("dId_list : ", dId_list)

# ===============================
# print("dataset : ", dataset)
y_dataset = dataset.Value
x_dataset = dataset.drop(["Value"], axis=1)

# =============
indexH = dataset[(dataset["hour"] == hour) & (dataset["Day"] == day) & (dataset["Month"] == month)].index
print("index :: ", indexH)
print("index :: ", indexH[0])
x_predict = x_dataset.iloc[indexH[0]]
print("x_predict shape :", x_predict.shape)

x_predict = x_predict.ravel()
print("x_predict shape :", x_predict.shape)

x_predict = x_predict.reshape(-1, 1)
print("x_predict shape :", x_predict.shape)

x_predict = x_predict.reshape(1, -1)
print("x_predict shape :", x_predict.shape)
# =============

# print("y_dataset : ",y_dataset)
# print("x_dataset : ", x_dataset)
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
print("x_train : ", x_train)
print("y_train : ", y_train)
print("x_train shape : ", x_train.shape)
print("y_train shape  : ", y_train.shape)

#////////////////////////////////////////////////////////////
# x_train,y_train=np.array(x_train),np.array(y_train)
# x_train.to_numpy()
x_train = x_train.to_numpy()
print("x_train  :   \n", x_train)
x_train=np.reshape(x_train,(x_train.shape[0],7,1))
print("x_train shape :   \n", x_train.shape)

model=Sequential()
# lstm_model.add(LSTM(100, input_shape=(7,1)))
model.add(LSTM(units=50,return_sequences=True,input_shape=(7, 1)))
# lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# =============
# def keras_model(input):
#     inputs = keras.Input(shape=(input, 1))
#     model = layers.LSTM(8)(inputs)
#     model = layers.LSTM(8)(model)
#     outputs = layers.Dense(1)(model)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
#     return model


# model = keras_model(7)

# ===============

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])
history = model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)

