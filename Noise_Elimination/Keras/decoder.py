#/usr/bin/python3


from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Model
from keras.layers import Input, Embedding, BatchNormalization, Reshape, Conv1D, Activation, add, Lambda, Layer, LSTM, Add, Concatenate, multiply, Dense, Subtract

import tensorflow as tf
from keras.initializers import glorot_uniform
from keras import backend as K

from utils import Simple
from attention import AttentionLayer

from tensorflow.random import categorical

from utils import categorical_log_prob, gather_nd
from utils import Decode



def decoder(encoded_input, f, n_hidden, config):
    Layer1 = Simple(n_hidden, name = 'rand_inp')
    first_input_ = Layer1(f)
    lstm = LSTM(8, kernel_initializer = glorot_uniform(), return_sequences = True, stateful = True, name = 'lstm')  
    positions = []
    log_softmax = []
    Layer2 = Decode(config.batch_size, name = 'decode_layer') 
    attn_lyr = AttentionLayer(name = 'attention_layer')
    i = first_input_
    for step in range(config.nCells * config.nMuts):
        output = lstm(i)
        attn_out, attn_stat = attn_lyr([encoded_input, output])
        i, position, log_soft1 = Layer2([encoded_input, attn_out])
        positions.append(position)
        log_softmax.append(log_soft1)

    poss = Concatenate(axis = -1, trainable = True, name = 'poss')(positions)
    log_s = Concatenate(axis = -1, trainable = True, name = 'log_s')(log_softmax)
    return poss, log_s
