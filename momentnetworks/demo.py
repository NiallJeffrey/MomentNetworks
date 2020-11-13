
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
import time


class simple_leaky():
    """
    A simple MLP with LeakyReLU activation
    """
    
    def __init__(self, input_size, output_size, learning_rate=None):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        
    def model(self):
        print(self.input_size)
        
        input_data = (Input(shape=(self.input_size,)))

        x1 = Dense(self.input_size, input_dim=self.input_size, kernel_initializer='normal')(input_data)
        x1 = LeakyReLU(alpha=0.1)(x1)
        x2 = Dense(self.input_size*2, kernel_initializer='normal')(x1)
        x2 = LeakyReLU(alpha=0.1)(x2)
        x3 = Dense(self.input_size, kernel_initializer='normal')(x2)
        x3 = LeakyReLU(alpha=0.1)(x3)
        x4 = Dense(self.output_size, kernel_initializer='normal')(x3)
        

        dense_model = Model(input_data, x4)
        dense_model.summary()

        if self.learning_rate is None:
            dense_model.compile(optimizer='adam', loss='mse')
        else:
            dense_model.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')

        return dense_model

def generate__signal(size, n_training):
    """
    Generate high-dimensional signal with covariance
    """
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

def log_prior(theta, data,cov_signal_inv):
    return -0.5*np.inner(theta,np.inner(cov_signal_inv,theta))

def log_likelihood(theta, data,cov_noise_inv):
    return -0.5*np.inner(theta-data,np.inner(cov_noise_inv,theta-data))

def log_posterior_delfi(theta, data,cov_signal_inv):
    return log_prior(theta, data,cov_signal_inv) + DelfiEnsemble.log_likelihood_stacked(theta,data=data) 

def log_posterior_likelihood(theta, data,cov_signal_inv,cov_noise_inv):
    return log_prior(theta, data,cov_signal_inv)  + log_likelihood(theta, data,cov_noise_inv)


def initial_parameters(theta, relative_sigma):
    """
    This is not randomise the initial position of the
    :param theta: list/array of parameter values
    :param relative_sigma: controls variance of random draws
    :return: the theta array but with random shifts
    """
    theta = np.array(theta)
    return np.random.normal(theta, np.abs(theta * relative_sigma))


def grav_wave_series(n_training, mass_1_samples, mass_2_samples, distance_samples,data_length=256, t_samples = 2048):
  
  from pycbc.waveform import get_td_waveform
  import pycbc.noise
  import pycbc.psd
  flow = 30.0
  delta_f = 1.0 / 16
  flen = int(t_samples / delta_f) + 1
  psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
  delta_t = 1.0 / t_samples
  tsamples = int(1 / delta_t)

  
  sim_data = np.empty((n_training*8, data_length))
  sim_data_noisy = np.empty((n_training*8, data_length))

  params_rescaled = np.empty((n_training*2,2))

  for i in range(n_training):
    if (i%200)==0: print(str(i) + '/' + str(n_training))

    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                          mass1=mass_1_samples[i],
                          mass2=mass_2_samples[i],
                          delta_t=1.0/t_samples,
                          tsamples=t_samples,
                          f_lower=10,
                          distance=distance_samples[i])
    
    signal_temp = np.array(hp)[np.where((np.array(hp.sample_times)>-1)&(np.array(hp.sample_times)<0))]*1e21
    noisy_signal = signal_temp +   pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed = int(time.time()))*1e21
    
    sim_data[i*8] = signal_temp[:256]
    sim_data[i*8+1] = signal_temp[256:512]
    sim_data[i*8+2] = signal_temp[512:768]
    sim_data[i*8+3] = signal_temp[768:1024]
    sim_data[i*8+4] = signal_temp[1024:1280]
    sim_data[i*8+5] = signal_temp[1280:1536]
    sim_data[i*8+6] = signal_temp[1536:1792]
    sim_data[i*8+7] = signal_temp[1792:]

    sim_data_noisy[i*8] = noisy_signal[:256] 
    sim_data_noisy[i*8+1] = noisy_signal[256:512] 
    sim_data_noisy[i*8+2] = noisy_signal[512:768] 
    sim_data_noisy[i*8+3] = noisy_signal[768:1024]
    sim_data_noisy[i*8+4] = noisy_signal[1024:1280] 
    sim_data_noisy[i*8+5] = noisy_signal[1280:1536] 
    sim_data_noisy[i*8+6] = noisy_signal[1536:1792] 
    sim_data_noisy[i*8+7] = noisy_signal[1792:]

  sim_data= np.vstack([sim_data[:,:128],sim_data[:,-128:]])
  sim_data_noisy= np.vstack([sim_data_noisy[:,:128],sim_data_noisy[:,-128:]])

  return sim_data, sim_data_noisy
