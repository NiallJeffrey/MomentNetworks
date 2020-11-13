
import gc
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from chainconsumer import ChainConsumer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Input
from tensorflow.keras.layers import concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import os, sys
import random
import emcee as mc
import getdist


def generate__signal(size, n_training):
  cov_signal_fourier = np.identity(size)*np.logspace(size,size*0.9,size)/np.sum(np.logspace(size,size*0.9,size))

  training_z = np.random.multivariate_normal(np.zeros(size),cov_signal_fourier,n_training) + \
                  1j*np.random.multivariate_normal(np.zeros(size), cov_signal_fourier,n_training)
  training_z = np.fft.fft(training_z).real

  cov_signal = np.identity(size)*0.
  mean_z = np.mean(training_z,axis=0)
  for i in range(n_training-1):
    cov_signal += np.outer(training_z[i] - mean_z,training_z[i]-mean_z) /np.float(n_training-1.)

  cov_signal = np.where((cov_signal<0.9)&(cov_signal>0.2),cov_signal+0.3,cov_signal)
  training_z = np.random.multivariate_normal(np.zeros(size),cov_signal,n_training)

  cov_signal = np.identity(size)*0.
  mean_z = np.mean(training_z,axis=0)
  for i in range(n_training-1):
    cov_signal += np.outer(training_z[i] - mean_z,training_z[i]-mean_z) /np.float(n_training-1.)

  cov_signal_inv = np.linalg.pinv(cov_signal)

  return training_z, cov_signal, cov_signal_inv
