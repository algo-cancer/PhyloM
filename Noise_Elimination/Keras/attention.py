#/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Input
from keras.initializers import glorot_uniform
from keras.utils.vis_utils import plot_model

from config import get_config, print_config
# from dataset import DataSet
import matplotlib.pyplot as plt
from keras import backend as K




class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def energy_step(self, encoder_out_seq, decoder_out_seq, states):
        assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
        assert isinstance(states, list) or isinstance(states, tuple), assert_msg
        en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
        de_hidden = decoder_out_seq.shape[-1]
        reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
        W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
        U_a_dot_h = K.dot(decoder_out_seq, self.U_a)
        reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
        e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
        e_i = K.softmax(e_i)
        return e_i, [e_i]

    def context_step(self, encoder_out_seq, ei , states):
        c_i = K.sum(encoder_out_seq * K.expand_dims(ei, -1), axis=1)
        return c_i, [c_i]


    def create_inital_state(self, inputs, hidden_size):
        # We are not using initial states, but need to pass something to K.rnn funciton
        fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
        fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
        fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
        fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
        return fake_state

    def call(self, inputs):
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        fake_state_c = self.create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = self.create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])
        e_outputs, e_outputs_ = self.energy_step(encoder_out_seq, decoder_out_seq, [fake_state_e])
        c_outputs, c_outputs_ = self.context_step(encoder_out_seq, e_outputs, [fake_state_c])
        return [e_outputs, c_outputs] 

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[0][0], input_shape[0][1])),
            tf.TensorShape((input_shape[0][0], input_shape[0][2]))
        ]

  
if __name__ == "__main__":
    # get config
    config, _ = get_config()

    input1 = Input(shape = (25, 4), name = 'input1')
    input2 = Input(shape = (25, 4), name = 'input2')

    attn_lyr = AttentionLayer(name = 'attention_layer')
    attn_out, attn_stat= attn_lyr([input1, input2])


    model = Model(inputs = [input1, input2], outputs = [attn_out, attn_stat])
    plot_model(model, to_file= config.output_dir + '/encoder_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())

    print_config()

    dataset = DataSet(config)

    fn = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fn_r.txt')]
    fp = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fp_r.txt')]
    l = []
    for i in range(config.nCells):
        for j in range(config.nMuts):
            l.append([i,j])
    l = np.asarray(l)
    batch_size = config.batch_size
    nb_epoch=2
    for i in tqdm(range(config.starting_num, config.nb_epoch)): # epoch i
 
        input1 = dataset.train_batch(i, fp, fn, l)
        input2 = dataset.train_batch(i, fp, fn, l)
        attn_out, attn_stat = model.predict({'input1': input1, 'input2': input2}, batch_size = batch_size)
        print('\n attn_out \n', attn_out)
        print(np.shape(attn_out))
        print(type(attn_out))

        print('\n attn_stat \n', attn_stat)
        print(np.shape(attn_stat))
        print(type(attn_out))

        for j in range(np.shape(attn_out)[0]):
	        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (32, 32))
	        im1 = ax.imshow(attn_out[j, :, :], cmap='gist_earth')
	        ax.set_xticks(np.arange(attn_out[j, :, :].shape[0]))
	        ax.set_yticks(np.arange(attn_out[j, :, :].shape[1]))
	        ax.set_xticklabels(input1[j, :, 3])
	        ax.set_yticklabels(input2[j, :, 3])

	        # im2 = ax[1].imshow(attn_stat[j, :, :])
	        fig.colorbar(im1)
	        fig.savefig( "attn{}.png".format(j))   
	        plt.close(fig)




