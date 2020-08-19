#/usr/bin/python3

from config import get_config, print_config
config, _ = get_config()

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from attention import AttentionLayer
from encoder import encoder
from decoder import decoder
from utils import *
from critic import critic
from cost import count3gametes, cost
from predict import solve
from data import *


def compile_models():

    input_data = Input(shape = (config.nCells * config.nMuts, config.input_dimension), name = 'main_input')
 
    ########################### Encoder ###########################
    encoded_input = encoder(input_data, config)
    n_hidden = K.int_shape(encoded_input)[2] # num_neurons

    ########################### Critic ############################
    critic_predictions = critic(input_data, n_hidden, config)


    ########################### Decoder ###########################
    f = Input(batch_shape = (config.batch_size, n_hidden), name = 'f_input')  # random tensor as first input of decoder
    poss, log_s = decoder(encoded_input, f, n_hidden, config)

    cost_layer = Cost(poss, config, name = 'Cost_layer')
    cost_v = cost_layer(input_data)

    reward_baseline_layer = StopGrad(name = 'stop_gradient')
    reward_baseline = reward_baseline_layer([critic_predictions, cost_v])

    ########################### Models ###########################
    AdamOpt_actor = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, amsgrad=False)  
    AdamOpt_critic = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, amsgrad=False)

    model_critic = Model(inputs = input_data, outputs = critic_predictions)  
    model_critic.compile(loss = costum_loss1(critic_predictions, cost_v), optimizer = AdamOpt_critic)


    model_actor = Model(inputs = [input_data, f], outputs = poss)  
    model_actor.compile(loss = costum_loss(reward_baseline, log_s), optimizer = AdamOpt_actor)
    return model_critic, model_actor




if __name__ == "__main__":

    ########################### configs ###########################
    # config, _ = get_config()
    K.set_learning_phase(1)
    K.tensorflow_backend._get_available_gpus()
    K.clear_session()

    ########################### Training mode ###########################
    if not config.inference_mode:

        model_critic, model_actor = compile_models()
        print_config()

        ########################### Dataset ###########################

        data = data(config.nb_epoch*config.batch_size, config.nCells, config.nMuts, config.ms_dir, config.alpha, config.beta)
        print('Dataset was created!')
        matrices_p, matrices_n = data
        l = []
        for i in range(config.nCells):
            for j in range(config.nMuts):
                l.append([i,j])
        l = np.asarray(l)

        f_input = np.random.randn(config.batch_size, config.input_dimension*config.hidden_dim)
        target_actor = [np.zeros((config.batch_size))] 
        target_critic = [np.zeros((config.batch_size))]
        act_loss = []
        crc_loss = []

        ########################### Training ###########################
        for i in tqdm(range(config.nb_epoch)):
            actor_loss = model_actor.train_on_batch([train_batch(config, np.asarray(matrices_n), l, i), f_input], target_actor)
            critic_loss = model_critic.train_on_batch(train_batch(config, np.asarray(matrices_n), l, i), target_critic)
            act_loss.append(actor_loss)
            crc_loss.append(critic_loss)

            if (i % 10 == 0) and (i != 0):
                # serialize model to JSON
                model_json = model_actor.to_json()
                with open(config.save_to + "/model_{}.json".format(i), "w") as json_file:
                    json_file.write(model_json)

                model_actor.save_weights(config.save_to + "/model_{}.h5".format(i))
                print("Saved model to disk")
            if i == config.nb_epoch - 1:

                model_json = model_actor.to_json()
                with open(config.save_to + "/actor.json", "w") as json_file:
                    json_file.write(model_json)

                model_actor.save_weights(config.save_to + "/actor.h5")
                print("Saved actor to disk")


        ########################### Trainig loss plot ###########################
        f, axes = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace = 0.35)


        axes[0].plot(act_loss)
        axes[0].set_title('training loss, actor')
        axes[0].set(xlabel = 'batch' , ylabel = 'loss')
        axes[0].legend(['actor training'], loc='upper left')

        axes[1].plot(crc_loss)
        axes[1].set_title('training loss, critic')
        axes[1].set(xlabel = 'batch' , ylabel = 'loss')
        axes[1].legend(['critic training'], loc='upper left')

        plt.savefig('{}.png'.format('loss_{nCells}x{nMuts}'.format(nCells = config.nCells, nMuts = config.nMuts)))
        plt.close(f)

    ########################### Inference mode ###########################
    else:

        model_critic, model_actor = compile_models()
        ##################### Evaluation dataset ######################
        data = data(config.nTestMats, config.nCells, config.nMuts, config.ms_dir, config.alpha, config.beta)
        print('Dataset was created!')

        ################# Loading weights from disk ###################
        model_actor.load_weights(config.restore_from + "/actor.h5")
        print("Weights were loaded to the model!")

        ##################### Error Correction ######################
        solve(model_actor, config, config.input_dimension*config.hidden_dim, data)



        # with CustomObjectScope({'Simple': Simple, 'Decode': Decode, 'StopGrad' : StopGrad, 'Cost' : Cost, 'AttentionLayer': AttentionLayer}):
        #     json_file = open(config.restore_from + '/actor.json', 'r')
        #     loaded_model_json = json_file.read()
        #     json_file.close()
        #     loaded_model = model_from_json(loaded_model_json)

        #     # load weights into new model
        #     loaded_model.load_weights(config.restore_from + "/actor.h5")
        #     print("Loaded model from disk")

        # solve(loaded_model, config, config.input_dimension*config.hidden_dim)






