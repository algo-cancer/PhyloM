#!/usr/bin/env python

# importing data from a file

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import pandas as pd
from random import sample
from random import seed
seed(30)
import copy
from time import time


from actor import Actor
from config import get_config, print_config
# from dataset import DataSet
from data import *

### Model: Critic (state value function approximator) = slim mean Attentive (parametric baseline ***)
###        w/ moving baseline (init_value default = 7 for TSP20, 20 for TSP40)       
###        Encoder = w/ FFN ([3] num_stacks / [16] num_heads / inner_FFN = 4*hidden_dim / [0.1] dropout_rate)
###        Decoder init_state = train, mean(enc)                                
###        Decoder inputs = Encoder outputs
###        Decoder Glimpse = Attention_g on ./(mask - first) + Residual connection

def Calc_C(S, N):
    fn_n = np.sum(N - S == -1) 
    fp_n = np.sum(S - N == -1)
    return fp_n*1.0/(np.sum(S == 0) + fp_n) , fn_n*1.0/(np.sum(S == 1)+ fn_n)


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()


    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    # Creating dataset
    if not config.inference_mode:
        l = []
        for i in range(config.nCells):
            for j in range(config.nMuts):
                l.append([i,j])
        l = np.asarray(l)

    
    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0, max_to_keep= 1000)  

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Training mode
        if not config.inference_mode:

            dataset = data(config.nb_epoch*config.batch_size, config.nCells, config.nMuts, config.ms_dir, config.alpha, config.beta)
            print('Dataset was created!')
            matrices_p, matrices_n = dataset

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)): 
             
                feed = {actor.input_: train_batch(config, np.asarray(matrices_n), l, i)}

                # Forward pass & train step
                summary, train_step1, train_step2 = sess.run([actor.merged, actor.train_step1, actor.train_step2], feed_dict=feed)

            print("Training COMPLETED !")
            saver.save(sess, config.save_to + "/actor.ckpt")
        # Inference mode
        else:

            dataset = data(config.nTestMats, config.nCells, config.nMuts, config.ms_dir, config.alpha, config.beta)
            print('Dataset was created!')
            matrices_p, matrices_n = dataset
            matrices_n_t = np.asarray(matrices_n)
            matrices_p_t = np.asarray(matrices_p)
            nMats = np.shape(matrices_n_t)[0]

            saver.restore(sess, config.restore_from + "/actor.ckpt")
            print("Model restored.")
            
            V_o = np.zeros((nMats, 1), dtype = np.float64)
            f_1_to_0_o = np.zeros((nMats, 1), dtype = np.float64)
            f_0_to_1_o = np.zeros((nMats, 1), dtype = np.float64)
            N00_o = np.zeros((nMats, 1), dtype = np.float64)
            N11_o = np.zeros((nMats, 1), dtype = np.float64)            
            N00_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
            N11_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
            N10_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
            N01_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
            NLL_o = np.zeros((nMats, 1), dtype = np.float64)
            V_o = np.zeros((nMats, 1), dtype = np.float64)
            
            
            fp_fn = np.zeros((nMats, config.nCells, config.nMuts), dtype = np.float32)
            for k in range(np.shape(matrices_n_t)[0]):
                
                fp_fn[k, matrices_n_t[k,:,:] == 1] = config.alpha
                fp_fn[k, matrices_n_t[k,:,:] == 0] = config.beta
                
                
                N01_o_ = np.sum(matrices_n_t[k,:,:] - matrices_p_t[k,:,:] == -1) 
                N10_o_ = np.sum(matrices_p_t[k,:,:] - matrices_n_t[k,:,:] == -1)
                N11_o_ = np.sum(matrices_p_t[k,:,:] + matrices_n_t[k,:,:] == 2)
                N00_o_ = np.sum(matrices_p_t[k,:,:] - matrices_n_t[k,:,:] == 0) - N11_o_
                
                f_1_to_0_o[k, 0] = N10_o_
                f_0_to_1_o[k, 0] = N01_o_
                # fp_o = config.alpha
                # fn_o = config.beta
                

                N00_o[k, 0] = N00_o_
                N11_o[k, 0] = N11_o_
                N00_NLL_o[k, 0] = N00_o_*np.log(1/(1-config.beta))
                N11_NLL_o[k, 0] = N11_o_*np.log(1/(1-config.alpha))
                N01_NLL_o[k, 0] = N01_o_*np.log(1/config.beta)
                N10_NLL_o[k, 0] = N10_o_*np.log(1/config.alpha)
                NLL_o[k, 0] = np.sum([N00_NLL_o[k, 0], N11_NLL_o[k, 0], N01_NLL_o[k, 0], N10_NLL_o[k, 0]])
                
                k += 1
                     
            l = []
            for i in range(config.nCells):
                for j in range(config.nMuts):
                    l.append([i,j])
            l = np.asarray(l)
            max_length = config.nCells * config.nMuts
            a = np.expand_dims(matrices_n_t.reshape(-1, actor.max_length),2)
            b = np.expand_dims(fp_fn.reshape(-1, actor.max_length),2)
            x = np.tile(l,(nMats,1,1))
            c = np.squeeze(np.concatenate([x,b,a], axis = 2))
            d = np.asarray([np.take(c[i,:,:],np.random.permutation(c[i,:,:].shape[0]),axis=0,out=c[i,:,:]) for i in range(np.shape(c)[0])])
            
            output_ = np.zeros((nMats, 14), dtype = np.float64)
            for j in tqdm(range(nMats)): # num of examples
                start_t = time()

                input_batch = np.tile(d[j,:,:],(actor.batch_size,1,1))
                
                feed = {actor.input_: input_batch}

                
                pos  = sess.run([actor.positions] , feed_dict=feed)[0]


                inp_ = tf.convert_to_tensor(input_batch, dtype=tf.float32)
                pos =  tf.convert_to_tensor(pos, dtype=tf.int32)

                
                r = tf.range(start = 0, limit = actor.batch_size, delta = 1)
                r = tf.expand_dims(r ,1)
                r = tf.expand_dims(r ,2)
                r3 = tf.cast(tf.ones([actor.max_length , 1]) * tf.cast(r, tf.float32), tf.int32)
                r4 = tf.squeeze(r, axis = 2)
                i = 0
                while i < int(max_length/10):    
                    r5 = tf.expand_dims(tf.fill([actor.batch_size], i), axis = 1)
                    u = tf.ones_like(r5)
                    r4_r5 = tf.concat([r4, r5], axis = 1)

                    pos_mask = tf.squeeze(tf.scatter_nd(indices = r4_r5, updates = u, shape = [actor.batch_size, actor.max_length, 1]), axis = 2)

                    pos_mask_cum1 = tf.cumsum(pos_mask, reverse = True, exclusive = True, axis = 1)
                    pos_mask_cum2 = tf.cumsum(pos_mask, reverse = False, exclusive = False, axis = 1) # for calculating NLL

                    per_pos = tf.concat([r3, tf.expand_dims(pos, axis = 2)], axis = 2)

                    per_ = tf.gather_nd(inp_, indices = per_pos)
            
                    per_matrix = per_[:,:,3:4]

                    # flipping the input
                    m1 = tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum1, tf.float32))
                    m1 = tf.subtract(tf.cast(pos_mask_cum1, tf.float32) , m1)
                    m2 = tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum2, tf.float32))
                    T_f = tf.add(m1, m2)

                    per_flipped = tf.concat([per_[:,:,0:3], tf.expand_dims(T_f, axis = 2)], axis = 2)
                    idx = tf.concat([r3, tf.cast(per_flipped[:,:,0:2], tf.int32)], axis = 2)
                    m_f = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = per_flipped[:,:,3:4], shape = tf.constant([actor.batch_size, actor.config.nCells, actor.config.nMuts]))           
                    c_v = actor.count3gametes(m_f) # cost for flipped matrix
                    V_rl = c_v.eval()
                    g = np.min(V_rl)
                    
                    # Calculating NLL
                    per_fp_fn = per_[:,:,2:3]
                    per_fp_fn_log = tf.log(1/per_fp_fn) # for N01 and N10
                    per_fp_fn_com = tf.subtract(tf.ones_like(per_fp_fn), per_fp_fn) # for N00 and N11
                    per_fp_fn_com_log = tf.log(1/per_fp_fn_com)

                    NLL_N10_N01 = tf.reduce_sum(tf.multiply(tf.squeeze(per_fp_fn_log, axis = 2), tf.cast(pos_mask_cum1, tf.float32)), axis = 1, keepdims = True)

                    per_matrix_mul_cum2 = tf.multiply(tf.squeeze(per_[:,:,3:4], axis = 2), tf.cast(pos_mask_cum2, tf.float32))
                    N11 = tf.reduce_sum(per_matrix_mul_cum2, axis = 1, keepdims = True)
                    N11_rl = tf.squeeze(N11, axis = 1).eval()
                    sum_mask_cum2 = tf.reduce_sum(tf.cast(pos_mask_cum2, tf.float32), axis = 1, keepdims = True )
                    N00 = tf.subtract(sum_mask_cum2, N11)
                    N00_rl = tf.squeeze(N00, axis = 1).eval()

                    sum_per_matrix = tf.reduce_sum(tf.squeeze(per_matrix, axis = 2) , axis = 1)
                    sum_per_fp =  tf.reduce_sum(tf.squeeze(tf.multiply(per_fp_fn, per_matrix) , axis = 2) , axis = 1)
                    fp = tf.divide(sum_per_fp, sum_per_matrix)
                    fp_r = fp.eval()

                    sum_per_fn = tf.subtract(tf.reduce_sum(tf.squeeze(per_fp_fn, axis = 2), axis = 1), sum_per_fp)
                    q = tf.cast(tf.tile(tf.constant([actor.max_length]), tf.constant([actor.batch_size])), tf.float32)
                    fn = tf.divide(sum_per_fn, tf.subtract(q, sum_per_matrix) )
                    fn_r = fn.eval()

                    fp_com = tf.log(1/tf.subtract(tf.cast(tf.tile(tf.constant([1]), tf.constant([actor.batch_size])), tf.float32), fp))
                    fn_com = tf.log(1/tf.subtract(tf.cast(tf.tile(tf.constant([1]), tf.constant([actor.batch_size])), tf.float32), fn))

                    N00_NLL = tf.multiply(tf.expand_dims(fp_com, axis = 1), N00)
                    N11_NLL = tf.multiply(tf.expand_dims(fn_com, axis = 1), N11)

                    NLL = tf.scalar_mul(config.gamma, tf.add_n([NLL_N10_N01, N00_NLL, N11_NLL ]))            
                    NLL_rl = tf.squeeze(NLL, axis =1).eval()
                    
                    g_w = np.where(V_rl == g)[0]
                    g_w_nll = np.argmin(NLL_rl[g_w])
                    gg = g_w[g_w_nll]    

                    
                    if g == 0:
                        c_v_rl = V_rl[gg]
                        m_rl = m_f.eval()[gg]                    
                        N10 = tf.reduce_sum(tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum1, tf.float32)), axis = 1, keepdims = True)
                        f_1_to_0_rl = tf.squeeze(N10, axis = 1)[gg].eval()
                        sum_mask_cum1 = tf.reduce_sum(tf.cast(pos_mask_cum1, tf.float32), axis = 1, keepdims = True )
                        N01 = tf.subtract(sum_mask_cum1, N10)
                        f_0_to_1_rl = tf.squeeze(N01, axis = 1)[gg].eval()
                        n_f = copy.deepcopy(i)
                        
                        # cost of original
                        idx = tf.concat([r3, tf.cast(inp_[:,:,0:2], tf.int32)], axis = 2)
                        m = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = inp_[:,:,3:4], shape = tf.constant([actor.batch_size, actor.config.nCells, actor.config.nMuts]))
                        c_v_o = actor.count3gametes(m)
                        c_n = c_v_o[0].eval()
                        fp_v = fp_r[gg]
                        fn_v = fn_r[gg]
                        c2 = copy.deepcopy(NLL_rl[gg])
                        
                        df = pd.DataFrame(m_rl.astype(int) , index = ['cell' + str(k1) for k1 in range(np.shape(m_rl)[0])], \
                                          columns = ['mut' + str(h1) for h1 in range(np.shape(m_rl)[1])])
                        df.index.rename('cellID/mutID', inplace=True)
                        df.to_csv(config.output_dir + '/mrl_{}.txt'.format(j + 1), sep='\t')
                        break
                        
                    c_t = tf.add(tf.squeeze(NLL, axis = 1), tf.cast(c_v, tf.float32))
                    
                    if i == 0:
                        c2 = copy.deepcopy(NLL_rl[gg])
                        c_v_rl = V_rl[gg]
                        n_f = copy.deepcopy(i)
                        f_0_to_1_rl = 0
                        f_1_to_0_rl = 0
                        m_rl = m_f.eval()[gg]
                        fp_v = fp_r[gg]
                        fn_v = fn_r[gg]
                        g1 = copy.deepcopy(g)
                        
                    if g1 > g: #c2 > NLL_rl[gg]:
                        c2 = copy.deepcopy(NLL_rl[gg])
                        c_v_rl = V_rl[gg]
                        n_f = copy.deepcopy(i)
                        f_0_to_1_rl = tf.squeeze(N01, axis = 1)[gg].eval()
                        f_1_to_0_rl = tf.squeeze(N10, axis = 1)[gg].eval()
                        m_rl = m_f.eval()[gg] 
                        fp_v = fp_r[gg]
                        fn_v = fn_r[gg]
                        g1 = copy.deepcopy(g)
                     
                    if i == int(max_length/10) - 1: 
                        # cost of original
                        idx = tf.concat([r3, tf.cast(inp_[:,:,0:2], tf.int32)], axis = 2)
                        m = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = inp_[:,:,3:4], shape = tf.constant([actor.batch_size, actor.config.nCells, actor.config.nMuts]))
                        c_v_o = actor.count3gametes(m)
                        c_n = c_v_o[0].eval()
                        df = pd.DataFrame(m_rl.astype(int) , index = ['cell' + str(k1) for k1 in range(np.shape(m_rl)[0])], \
                                          columns = ['mut' + str(h1) for h1 in range(np.shape(m_rl)[1])])
                        df.index.rename('cellID/mutID', inplace=True)
                        df.to_csv(config.output_dir + '/mrl_{}.txt'.format(j + 1), sep='\t') 
                    i += 1  
                dur_t = time() - start_t

                output_[j,0] = fp_v
                output_[j,1] = fn_v 
                output_[j,2] = c2  # cost (NLL part)
                output_[j,3] = c_v_rl  # cost (violation part)
                output_[j,4] = c_n # number of violations  for noisy matrix
                output_[j,5] = n_f # total number of flips based on rl
                output_[j,6] = f_0_to_1_rl
                output_[j,7] = f_1_to_0_rl
                output_[j,8] = dur_t
                # output_[j,9] = s_m[j]
                    
                    
                    
            output_[:,9] = np.squeeze(N00_o)
            output_[:,10] = np.squeeze(N11_o)
            output_[:,11] = np.squeeze(NLL_o)
            output_[:,12] = np.squeeze(f_1_to_0_o)
            output_[:,13] = np.squeeze(f_0_to_1_o)
            
            df = pd.DataFrame(output_, index = ["test" + str(k) for k in range(nMats)], \
                             columns = ["fp", "fn","NLL_rl", "V_rl", "V_o", "n_f", "f_0_to_1_rl", "f_1_to_0_rl",\
                                        "time", "N00_o", "N11_o", "NLL_o", "f_1_to_0_o", "f_0_to_1_o"])
            df.to_csv(config.output_dir + '/test_{nCells}x{nMuts}.csv'.format(nCells = config.nCells, nMuts = config.nMuts), sep = ',')
            
            
if __name__ == "__main__":
    main()
