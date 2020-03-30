import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import SimpleRNN,Activation, LSTM, Dense
from keras import backend as K
from datetime import datetime
import random
import os


def cog_lstm(input,
             weight = None,
             is_2layers=True,
             Tx = 600,
             classes_in = 71,
             classes_out = 33,
             h_units1 = 512,
             h_units2 = 256):
    """ 
    Neural network composed of LSTMs created to solve cognitive tasks
    
    INPUT:
    input - Training set. Needed to define size of samples for 1st layer
    weight - String directed for Weights of a pre-trained model to upload 
    to the model. Default is None.
    """

    model = Sequential()
    model.add(LSTM(h_units1, input_shape=(input.shape[1], input.shape[2]),return_sequences=True))#, return_state= True))
    if is_2layers:
        model.add(LSTM(h_units2, return_sequences=True))#, return_state= True))
    model.add(Dense(classes_out, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
    optimizer='adam',metrics=['categorical_accuracy'])
    model.summary()

    if weight is not None:
        model.load_weights(weight)

    first_layer = Model(model.input, model.get_layer('lstm_1').output)
    if is_2layers:
        second_layer = Model(model.input,  model.get_layer('lstm_2').output)
    else:
        second_layer = None

    return model, first_layer, second_layer
