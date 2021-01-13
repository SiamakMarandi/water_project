import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

def model_maker(inputs):

    def keras_model(input):
        inputs = keras.Input(shape=(input, 1))
        model = layers.LSTM(12, return_sequences=True)(inputs)
        model = layers.LSTM(12)(model)  
        model = layers.Dense(10)(model)
        outputs = layers.Dense(1)(model)
        model = keras.Model(inputs=inputs, outputs=outputs, name="water_predictor")
        return model


    model = keras_model(inputs)

    print("output_shape  :   ", model.output_shape)

    # Model summary
    model.summary()

    # ===================================plotting the model as a graph start
    keras.utils.plot_model(model, "my_first_model.png")
    keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
    # ===================================plotting the model as a graph end
    # Model config
    # print("get_config  :   ",model.get_config())

    # List all weight tensors 
    # print("get_weights  :   ", model.get_weights())

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae', 'mape'])  
    return model
