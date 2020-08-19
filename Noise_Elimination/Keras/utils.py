#/usr/bin/python3


from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as K
import tensorflow as tf
from keras.layers import Layer, multiply, Embedding
from tensorflow.random import categorical
from tensorflow.contrib.distributions import Categorical
from cost import cost
from attention import AttentionLayer


def categorical_sample(p):
    return Categorical(p).sample()

def categorical_log_prob(p, x):
    return Categorical(p).log_prob(x) 

def gather_nd(inp, x):
    return K.gather(inp, indices = x)[0, :, :]

def costum_loss(x, y):
    y = K.sum(y, axis = 1)
    def loss(y_true, y_pred):
        mul = multiply([x, y])
        return K.sum(mul, axis = 0)
    return loss

def costum_loss1(x, y):
    def loss(y_true, y_pred):
        return K.mean(K.square(x - y), axis = -1)
    return loss

class Simple(Layer):

    def __init__(self, output_dim, **kwargs): #, activation
        self.output_dim = output_dim
        super(Simple, self).__init__(**kwargs)
        self.__name__ = 'Simple'

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Simple, self).build(input_shape) 

    def call(self, x):
        return K.expand_dims(K.dot(x, self.kernel), axis = 1)

    def get_config(self):
        config = super(Simple, self).get_config()
        config.update({'output_dim' : self.output_dim})
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, self.output_dim)


class Decode(Layer):

    def __init__(self, batch_size, **kwargs): 
        self.batch_size = batch_size
        super(Decode, self).__init__(**kwargs)
        self.__name__ = 'Decode'

    def build(self, input_shape):
        self.mask = tf.fill((self.batch_size, input_shape[0][1]), 0.0)
        super(Decode, self).build(input_shape) 

    def activation_decode(self, x):
        self.encoded_input, self.attn_out = x
        masked_scores = self.attn_out - 10000000*self.mask
        self.position = categorical(masked_scores, num_samples = 1, dtype = 'int32')
        position1 = K.squeeze(self.position, axis = 1)
        log_soft = categorical_log_prob(masked_scores, position1)
        self.log_soft1 = K.expand_dims(log_soft, axis = 1)
        mask1 = Embedding(input_dim = K.int_shape(self.attn_out)[1], output_dim = K.int_shape(self.attn_out)[1], embeddings_initializer = 'identity', trainable = True)(position1)
        self.mask = self.mask + mask1
        h = K.permute_dimensions(self.encoded_input, pattern = (1, 0, 2))
        i1 = gather_nd(h, position1)
        self.i = K.expand_dims(i1, axis = 1)
        return self.i, self.position, self.log_soft1

    def call(self, x):
        i, position, log_soft1 = self.activation_decode(x)
        return [i, position, log_soft1]

    def get_config(self):
        config = super(Decode, self).get_config()
        config.update({'batch_size' : self.batch_size})
        return config 

    def compute_output_shape(self, input_shape):
        return [K.int_shape(self.i), K.int_shape(self.position), K.int_shape(self.log_soft1)]



class StopGrad(Layer):

    def __init__(self, **kwargs): 
        super(StopGrad, self).__init__(**kwargs)
        self.__name__ = 'StopGrad'

    def build(self, input_shape):
        super(StopGrad, self).build(input_shape)  

    def call(self, x):
        self.critic_predictions, self.cost = x
        return K.stop_gradient(self.cost - self.critic_predictions)

    def compute_output_shape(self, input_shape):
        return [K.int_shape(self.critic_predictions)]

class Cost(Layer):

    def __init__(self, positions, config, **kwargs): 
        self.positions = positions
        self.config = config
        super(Cost, self).__init__(**kwargs)
        self.__name__ = 'Cost'

    def build(self, input_shape):
        super(Cost, self).build(input_shape)  

    def call(self, x):
        return cost(x, self.positions, self.config)

    def get_config(self):
        config = super(Cost, self).get_config()
        config.update({'config' : self.config, 'positions' : self.positions})
        return config

    def compute_output_shape(self, input_shape):
        return [K.int_shape(self.positions)[0]]




