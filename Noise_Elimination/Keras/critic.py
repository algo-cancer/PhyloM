#/usr/bin/python3


# import tensorflow as tf
# from tensorflow import keras
from __future__ import absolute_import, division, print_function, unicode_literals

from encoder import encoder
from keras.layers import Input, Embedding, BatchNormalization, Reshape, Conv1D, Activation, add, Lambda, Layer, LSTM, LSTMCell, Add, Concatenate, multiply, Dense, Subtract

from keras.initializers import glorot_uniform
from keras import backend as K






def critic(inp_, n_hidden, config):
    enc_inp = encoder(inp_, config)
    frame = Lambda(K.sum, arguments = {'axis' : 1}, trainable = True, name = 'critic_frame')(enc_inp)
    h0 = Dense(n_hidden, activation = 'relu', kernel_initializer = glorot_uniform(), name = 'critic_h0')(frame)
    # Layer2 = MyLayer(1, activation = Activation('relu'))
    # critic_layer = Layer2(h0)
    critic_predictions = Dense(1, activation = 'relu', kernel_initializer = glorot_uniform(), name = 'critic_predictions1')(h0)
    critic_predictions1 = Lambda(K.squeeze, arguments = {'axis' : 1}, trainable = False, name = 'critic_predictions2')(critic_predictions)
    return critic_predictions1


