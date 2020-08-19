from config import msAddress
from keras import backend as K
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from random import sample
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

import argparse
import copy
import h5py
import itertools
import keras
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import re
import re
import shutil
import sklearn as sk
import sys
import tensorflow as tf
import time


def save_models(models_list):
  for name, model in models_list:
    model.save_weights(os.path.join(tempDir, f"tempModel_{name}.h5"))


def add_dense_layer_classification(modelsList,
                                inputDim=1,
                                nameSuffix="0",
                                hiddenSize=10, nLayers = 3,
                                dropOutRate=0.8, dropOutAfterFirst=True,
                                activation = "sigmoid",
                                useSoftmax = True
                                ):
  if isinstance(hiddenSize, list):
    for arg in hiddenSize:
      add_dense_layer_classification(modelsList, inputDim, nameSuffix,
                                  arg,
                                  nLayers, dropOutRate, dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(nLayers, list):
    for arg in nLayers:
      add_dense_layer_classification(modelsList, inputDim, nameSuffix, hiddenSize,
                                  arg,
                                  dropOutRate, dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(dropOutRate, list):
    for arg in dropOutRate:
      add_dense_layer_classification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers,
                                  arg,
                                  dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(dropOutAfterFirst, list):
    for arg in dropOutAfterFirst:
      add_dense_layer_classification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers, dropOutRate,
                                  arg,
                                  activation, useSoftmax)
  elif isinstance(activation, list):
    for arg in activation:
      add_dense_layer_classification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers,
                                  dropOutRate, dropOutAfterFirst,
                                  arg,
                                  useSoftmax)
  elif isinstance(useSoftmax, list):
    for arg in useSoftmax:
      add_dense_layer_classification(modelsList, inputDim, nameSuffix, hiddenSize,nLayers,
                                  dropOutRate, dropOutAfterFirst, activation,
                                  arg)
  else:
    assert (not dropOutAfterFirst) or dropOutRate != None
    name = f"DenseDrOut_{nLayers}_{hiddenSize}_{activation}_{dropOutRate}_{dropOutAfterFirst}_{useSoftmax}_{nameSuffix}_{len(modelsList)}"
    model = Sequential()

    model.add(Dense(hiddenSize, input_shape=(inputDim,), activation = activation))
    if dropOutRate:
      model.add(Dropout(dropOutRate))
    for i in range(nLayers-2):
      model.add(Dense(hiddenSize, activation = activation))
      if dropOutAfterFirst:
        model.add(Dropout(dropOutRate))

    if useSoftmax:
      model.add(Dense(2, activation='softmax'))
    else:
      model.add(Dense(2, activation=activation))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[ 'categorical_accuracy',
                         #     acc_m,
                         #     recall_m,
                         # precision_m,
                            ])
    modelsList.append((name, model))


def permute(m):
    assert len(m.shape) == 2
    rowPermu = np.random.permutation(m.shape[0])
    colPermu = np.random.permutation(m.shape[1])
    return m[rowPermu, :][:, colPermu]


def run_ms(filename, nCell, nMut):
    cmd = f"{msAddress} {nCell} 1 -s {nMut} | tail -n {nCell} > {filename}"
    os.system(cmd)
    with open(filename, 'r') as f:
        l = [line for line in f]
    l1 = [s.strip('\n') for s in l]
    l2 = np.array([[int(s) for s in q] for q in l1])  # Original matrix
    return l2


def make_dataset(nCells, nMuts, n):  # TODO: the weights of rows might be bias

    X = np.zeros((2 * n, nCells * nMuts), dtype=np.int8)
    for i in range(n):
        res = True
        while res:
            original1 = run_ms("temp.txt", nCells, nMuts)
            res = is_linear(original1)

        # linearInputUtil.sort_bin(original1)
        original1 = permute(original1)
        X[i] = original1.reshape(-1)

    for i in range(n):
        w = np.sum(X[i])
        original2 = make_constrained_linear_input(nCells, nMuts, w)
        original2 = permute(original2)
        # linearInputUtil.sort_bin(original2)
        X[i + n] = original2.reshape(-1)

    y = np.array([(1, 0)] * n + [(0, 1), ] * n)
    return X, y


def is_linear(x):
  colOrder = np.argsort(-np.sum(x, axis=0), )
  x = x[:, colOrder]
  y = x[np.lexsort(np.rot90(x))]
  for j in range(y.shape[1]-1):
    u = int(y.shape[0] - np.sum(y[:,j]))
    remain = np.count_nonzero(y[:u, (j+1):])
    # print(j, u, remain)
    # print(y[(j+1):, :u])
    if remain!= 0:
      return False
  return True


def make_constrained_linear_input(n, m, w):
  """
  :param n:
  :param m:
  :param w:  Number of ones, i.e., weight.
  :return:
  """
  x = np.zeros((n, m))
  u = 0
  for j in range(m):
    if n - w > u:
      u = n - w
    uu = int(n - w / (m - j))+1
    u = np.random.randint(u, uu)
    w -= n-u
    x[u:, j] = 1
  return x


def add_noise(original_matrix, rates=None, ks=None):
  by_rate = rates is not None
  by_k = ks is not None
  assert by_rate != by_k, "Exactly one of rates and ks should be None"
  noisy_matrix = np.array(original_matrix.copy(), dtype=np.int)
  if by_k:
    n_values = len(ks)
  elif by_rate:
    n_values = len(rates)
  else:
    assert False, "upper assert should have caught"

  for val in range(n_values):
    x, y = np.where(original_matrix == val)
    if by_rate:
      sample_mask = np.random.random_sample(size=len(x)) < rates[val]
      sample_indices = np.nonzero(sample_mask)[0]
    else:
      k = min(ks[val], len(x))
      sample_indices = np.random.choice(len(x), size=k, replace=False)
    for i in sample_indices:
      assert noisy_matrix[x[i], y[i]] == val
      noisy_matrix[x[i], y[i]] = 1 - val
  return noisy_matrix


def make_noisy(X_clean, n_cells, n_muts, rates=None, ks=None, ):
  X = np.empty(X_clean.shape)
  for row_ind in range(X_clean.shape[0]):
    X[row_ind] = add_noise(X_clean[row_ind].reshape(n_cells, n_muts), rates=rates, ks=ks).reshape(n_cells * n_muts)
  return X
