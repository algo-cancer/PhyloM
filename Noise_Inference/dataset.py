

import os
import h5py
from copy import deepcopy
import numpy as np
import subprocess
import itertools

class DataSet(object):

    def __init__(self, config):
        self.config = config

    def count3gametes(self, matrix):
        columnPairs = list(itertools.permutations(range(self.config.nMuts), 2))
        nColumnPairs = len(columnPairs)
        columnReplicationList = np.array(columnPairs).reshape(-1)
        replicatedColumns = matrix[:, columnReplicationList].transpose()
        x = replicatedColumns.reshape((nColumnPairs, 2, self.config.nCells), order="A")
        col10 = np.count_nonzero( x[:,0,:]<x[:,1,:]     , axis = 1)
        col01 = np.count_nonzero( x[:,0,:]>x[:,1,:]     , axis = 1)
        col11 = np.count_nonzero( (x[:,0,:]+x[:,1,:]==2), axis = 1)
        eachColPair = col10 * col01 * col11
        return np.sum(eachColPair)

    def ms(self, nMats):
	    matrices = np.zeros((nMats, self.config.nCells, self.config.nMuts), dtype = np.int8)
	    cmd = "{ms_dir}/ms {nCells} 1 -s {nMuts} | tail -n {nCells}".format(ms_dir = self.config.ms_dir, nCells = self.config.nCells, nMuts = self.config.nMuts)
	    for i in range(nMats):
	        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell = True)
	        out = out.decode("utf-8").splitlines()
	        matrices[i,:,:] = np.array([list(q) for q in out]).astype(int)  # Original matrix
	    return matrices
	        

    def pure_noisy(self, nMats):
	    matrices = self.ms(nMats)
	    matrices_n = []
	    matrices_p = []
	    for i in range(np.shape(matrices)[0]):
	        v = 0
	        matrix = deepcopy(matrices[i,:,:].reshape(1, -1))
	        while ((self.count3gametes(matrix.reshape(self.config.nCells, self.config.nMuts)) == 0) and (v < self.config.nCells*self.config.nMuts)):
	            matrix = deepcopy(matrices[i,:,:].reshape(1, -1))
	            Zs = np.where(matrix  == 0)[1]
	            s_fp = np.random.choice([True, False], (1, len(Zs)), p = [self.config.alpha, 1 - self.config.alpha])  # must be flipped from 0 to 1
	            Os = np.where(matrix  == 1)[1] 
	            s_fn = np.random.choice([True, False], (1, len(Os)), p = [self.config.beta, 1 - self.config.beta]) # must be flipped from 1 to 0
	            matrix[0, Zs[np.squeeze(s_fp)]] = 1
	            matrix[0, Os[np.squeeze(s_fn)]] = 0
	            v += 1
	        matrices_n.append(matrix.reshape(self.config.nCells, self.config.nMuts))
	        matrices_p.append(matrices[i,:,:])
	    return matrices_p, matrices_n


    # shuffling rows and columns
    def permute(self, m):
        assert len(m.shape) == 2
        rowPermu = np.random.permutation(m.shape[0])
        colPermu = np.random.permutation(m.shape[1])
        return m[rowPermu, :][:, colPermu]

    def create_data(self, nMats):
        m_p, m_n = self.pure_noisy(nMats)
        inps_p = np.asarray(m_p)
        inps_n = np.asarray(m_n)
        for i in range(nMats):
            if self.config.lexSort == True:
                inps_p[i,:,:] = inps_p[i, :, np.lexsort(inps_p[i, :, :])]
                inps_n[i,:,:] = inps_n[i, :, np.lexsort(inps_n[i, :, :])]
            else:
                inps_p[i,:,:] = self.permute(inps_p[i,:,:])
                inps_n[i,:,:] = self.permute(inps_n[i,:,:]) 

        l_n = np.ones((nMats, 1), dtype = np.int8)
        l_p = np.zeros((nMats, 1), dtype = np.int8)
        m = np.concatenate((inps_n, inps_p), axis = 0)  # matrices
        l = np.concatenate((l_n, l_p), axis = 0)  # labels
        Permu = np.random.permutation(m.shape[0])
        m = m[Permu, :, :]
        l = l[Permu, :]
        return m, l


    def data(self, train):
    	if train == True:
    		m, l = self.create_data(self.config.nTrain)
    	else:
    		m, l = self.create_data(self.config.nTest)
    	return m, l

    def saveDataSet(self, X, y, fileName = None):
        if not fileName:
            fileName = f"dataset_{X.shape}_{y.shape}.h5" #
        fileAddress = os.path.join(self.config.h5_dir, fileName)
        # print(fileAddress)
        h5f = h5py.File(fileAddress, 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('y', data=y)
        h5f.close()

    def loadDataSet(self, fileName):
        fileAddress = os.path.join(self.config.h5_dir, fileName)
        assert(os.path.exists(fileAddress))
        h5f = h5py.File(fileAddress, 'r')
        X = h5f['X'][:]
        y = h5f['y'][:]
        h5f.close()
        return X, y

    
