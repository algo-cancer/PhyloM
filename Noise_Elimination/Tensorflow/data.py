
import numpy as np
from random import uniform
from copy import deepcopy
import itertools
import os
import subprocess


def count3gametes(matrix,nCells, nMuts):
    columnPairs = list(itertools.permutations(range(nMuts), 2))
    nColumnPairs = len(columnPairs)
    columnReplicationList = np.array(columnPairs).reshape(-1)
    replicatedColumns = matrix[:, columnReplicationList].transpose()
    x = replicatedColumns.reshape((nColumnPairs, 2, nCells), order="A")
    col10 = np.count_nonzero( x[:,0,:]<x[:,1,:]     , axis = 1)
    col01 = np.count_nonzero( x[:,0,:]>x[:,1,:]     , axis = 1)
    col11 = np.count_nonzero( (x[:,0,:]+x[:,1,:]==2), axis = 1)
    eachColPair = col10 * col01 * col11
    return np.sum(eachColPair)


def ms(nMats, nCells, nMuts, ms_dir):
    matrices = np.zeros((nMats, nCells, nMuts), dtype = np.int8)
    cmd = "{ms_dir}/ms {nCells} 1 -s {nMuts} | tail -n {nCells}".format(ms_dir = ms_dir, nCells = nCells, nMuts = nMuts)
    for i in range(nMats):
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell = True)
        out = out.decode("utf-8").splitlines()
        matrices[i,:,:] = np.array([list(q) for q in out]).astype(int)  # Original matrix
    return matrices
        

def data(nMats, nCells, nMuts, ms_dir, alpha, betta):
    matrices = ms(nMats, nCells, nMuts, ms_dir)
    matrices_n = []
    matrices_p = []
    for i in range(np.shape(matrices)[0]):
        v = 0
        matrix = deepcopy(matrices[i,:,:].reshape(1, -1))
        while ((count3gametes(matrix.reshape(nCells, nMuts), nCells, nMuts) == 0) and (v < nCells*nMuts)):
            matrix = deepcopy(matrices[i,:,:].reshape(1, -1))
            Zs = np.where(matrix  == 0)[1]
            s_fp = np.random.choice([True, False], (1, len(Zs)), p = [alpha, 1 - alpha])  # must be flipped from 0 to 1
            Os = np.where(matrix  == 1)[1] 
            s_fn = np.random.choice([True, False], (1, len(Os)), p = [betta, 1 - betta]) # must be flipped from 1 to 0
            matrix[0, Zs[np.squeeze(s_fp)]] = 1
            matrix[0, Os[np.squeeze(s_fn)]] = 0
            v += 1
        # if count3gametes(matrix.reshape(nCells, nMuts), nCells, nMuts) != 0:
        matrices_n.append(matrix.reshape(nCells, nMuts))
        matrices_p.append(matrices[i,:,:])
    return matrices_p, matrices_n


# Read a batch for training procedure
def train_batch(config, matrices, l, batch_num):
    matrices_n = np.zeros((config.batch_size, config.nCells, config.nMuts), dtype = np.int8)
    fp_fn = np.zeros((config.batch_size, config.nCells, config.nMuts), dtype = np.float32)
    k = 0
    for i in range(batch_num * config.batch_size, (batch_num + 1) * config.batch_size):
        matrices_n[k,:,:] = matrices[i,:,:]
        fp_fn[k, matrices_n[k,:,:] == 1] = config.alpha
        fp_fn[k, matrices_n[k,:,:] == 0] = config.beta
        k += 1
        
    a = np.expand_dims(matrices_n.reshape(-1, config.nCells*config.nMuts),2)
    b = np.expand_dims(fp_fn.reshape(-1, config.nCells*config.nMuts),2)
    x = np.tile(l,(config.batch_size,1,1))
    c = np.squeeze(np.concatenate([x,b,a], axis = 2))
    d = np.asarray([np.take(c[i,:,:],np.random.permutation(c[i,:,:].shape[0]),axis=0,out=c[i,:,:]) for i in range(np.shape(c)[0])])
    del matrices_n
    return d


    
