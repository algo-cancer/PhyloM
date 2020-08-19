#/usr/bin/python3


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import itertools
import tensorflow as tf


def count3gametes(input_, config):
    columnPairs = list(itertools.permutations(range(config.nMuts), 2))
    nColumnPairs = len(columnPairs)
    columnReplicationList = np.array(columnPairs).reshape(-1)
    l = []
    for i in range(input_.get_shape()[0]):
        for j in range(config.nCells):
            for k in columnReplicationList:
                l.append([i,j,k])
    replicatedColumns = tf.reshape(tf.gather_nd(input_, l), [input_.get_shape()[0], config.nCells, len(columnReplicationList)])
    replicatedColumns = tf.transpose(replicatedColumns, perm = [0,2,1])
    x = tf.reshape(replicatedColumns, [input_.get_shape()[0], nColumnPairs, 2, config.nCells])
    col10 = tf.count_nonzero(tf.greater(x[:,:,0,:], x[:,:,1,:]), axis = 2)# batch_size * nColumnPairs
    col01 = tf.count_nonzero(tf.greater(x[:,:,1,:], x[:,:,0,:]), axis = 2)# batch_size * nColumnPairs
    col11 = tf.count_nonzero(tf.equal(x[:,:,0,:]+x[:,:,1,:],2), axis = 2)# batch_size * nColumnPairs
    eachColPair = col10 * col01 * col11 # batch_size * nColumnPairs
    return tf.reduce_sum(eachColPair, axis = 1) # batch_size 


def cost(input_, positions, config):
    inp_ = tf.identity(input_)
    pos = tf.identity(positions)

    max_length = config.nCells * config.nMuts
    x = tf.zeros([int(max_length/2), config.batch_size], tf.float32)
    for i in range(int(max_length/2)):

        r = tf.range(start = 0, limit = config.batch_size, delta = 1)
        r = tf.expand_dims(r ,1)
        r = tf.expand_dims(r ,2)
        r3 = tf.cast(tf.ones([max_length , 1]) * tf.cast(r, tf.float32), tf.int32)


        r4 = tf.squeeze(r, axis = 2)
        r5 = tf.expand_dims(tf.fill([config.batch_size], i), axis = 1)
        u = tf.ones_like(r5)
        r4_r5 = tf.concat([r4, r5], axis = 1)

        pos_mask = tf.squeeze(tf.scatter_nd(indices = r4_r5, updates = u, shape = [config.batch_size, max_length, 1]), axis = 2)
        
        pos_mask_cum1 = tf.cumsum(pos_mask, reverse = True, exclusive = True, axis = 1)
        pos_mask_cum2 = tf.cumsum(pos_mask, reverse = False, exclusive = False, axis = 1) # for calculating NLL

        per_pos = tf.concat([r3, tf.expand_dims(pos, axis = 2)], axis = 2)

        per_ = tf.gather_nd(inp_, indices = per_pos)
        per_fp_fn = per_[:,:,2:3]
        per_fp_fn_log = tf.log(1/per_fp_fn) # for N01 and N10
        per_fp_fn_com = tf.subtract(tf.ones_like(per_fp_fn), per_fp_fn) # for N00 and N11
        per_fp_fn_com_log = tf.log(1/per_fp_fn_com)

        NLL_N10_N01 = tf.reduce_sum(tf.multiply(tf.squeeze(per_fp_fn_log, axis = 2), tf.cast(pos_mask_cum1, tf.float32)), axis = 1, keepdims = True)

        per_matrix_mul_cum2 = tf.multiply(tf.squeeze(per_[:,:,3:4], axis = 2), tf.cast(pos_mask_cum2, tf.float32))
        N11 = tf.reduce_sum(per_matrix_mul_cum2, axis = 1, keepdims = True)
        sum_mask_cum2 = tf.reduce_sum(tf.cast(pos_mask_cum2, tf.float32), axis = 1, keepdims = True )
        N00 = tf.subtract(sum_mask_cum2, N11)

        per_matrix = per_[:,:,3:4]

        sum_per_matrix = tf.reduce_sum(tf.squeeze(per_matrix, axis = 2) , axis = 1)
        sum_per_fp = tf.reduce_sum(tf.squeeze(tf.multiply(per_fp_fn, per_matrix) , axis = 2) , axis = 1)
        fp = tf.divide(sum_per_fp, sum_per_matrix)

        sum_per_fn = tf.subtract(tf.reduce_sum(tf.squeeze(per_fp_fn, axis = 2), axis = 1), sum_per_fp)
        q = tf.cast(tf.tile(tf.constant([max_length]), tf.constant([config.batch_size])), tf.float32)
        fn = tf.divide(sum_per_fn, tf.subtract(q, sum_per_matrix))

        fp_com = tf.log(1/tf.subtract(tf.cast(tf.tile(tf.constant([1]), tf.constant([config.batch_size])), tf.float32), fp))
        fn_com = tf.log(1/tf.subtract(tf.cast(tf.tile(tf.constant([1]), tf.constant([config.batch_size])), tf.float32), fn))

        N00_NLL = tf.multiply(tf.expand_dims(fp_com, axis = 1), N00)
        N11_NLL = tf.multiply(tf.expand_dims(fn_com, axis = 1), N11)

        NLL = tf.scalar_mul(config.gamma, tf.add_n([NLL_N10_N01, N00_NLL, N11_NLL ]))            


        m1 = tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum1, tf.float32))
        m1 = tf.subtract(tf.cast(pos_mask_cum1, tf.float32) , m1)
        m2 = tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum2, tf.float32))
        T_f = tf.add(m1, m2)

        per_flipped = tf.concat([per_[:,:,0:3], tf.expand_dims(T_f, axis = 2)], axis = 2)

        idx = tf.concat([r3, tf.cast(per_flipped[:,:,0:2], tf.int32)], axis = 2)
        m_f = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = per_flipped[:,:,3:4], shape = tf.constant([config.batch_size, config.nCells, config.nMuts]))           
        c_v = count3gametes(m_f, config)
        c_t = tf.expand_dims(tf.add(tf.squeeze(NLL, axis = 1), tf.cast(c_v, tf.float32)), axis = 0)
        ind = []
        for i1 in range(x.get_shape()[1]):
            ind.append([i,i1])
        ind = tf.convert_to_tensor(ind)
        ind = tf.expand_dims(ind , axis = 0)
        x_n = tf.scatter_nd(indices = ind , updates = c_t, shape = x.get_shape())
    x_m = tf.reduce_min(x, axis = 0)
    return tf.identity(x_m)



