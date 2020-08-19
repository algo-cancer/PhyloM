#/usr/bin/python3


from __future__ import absolute_import, division, print_function, unicode_literals
from keras.layers import Embedding, BatchNormalization, Reshape, Conv1D, Activation, add
from keras.initializers import glorot_uniform

from keras import backend as K



def encoder(inp_, config):
    embedded_input = Embedding(input_dim = 200, output_dim = config.hidden_dim, embeddings_initializer = glorot_uniform(), name = 'encoder_embedding')(inp_) # max(config.nCells, config.nMuts) + 3  
    embedded_input_r = Reshape((config.nCells*config.nMuts, config.input_dimension*config.hidden_dim), name = 'encoder_reshape')(embedded_input)

    enc = BatchNormalization(axis = 2, name = 'encoder_batchN1')(embedded_input_r)

    feedforward_output = Conv1D(filters = 4*config.input_dimension*config.hidden_dim, kernel_size = 1, activation = 'relu', use_bias = True, name = 'encoder_ff1')(enc)
    feedforward_output1 = Conv1D(filters = config.input_dimension*config.hidden_dim, kernel_size = 1, activation = 'relu', use_bias = True, name = 'encoder_ff2')(feedforward_output)

    enc_output = add([enc, feedforward_output1], name = 'encoder_add')

    encoded_input = BatchNormalization(axis = 2,  name = 'encoder_batchN2')(enc_output)

    return encoded_input


