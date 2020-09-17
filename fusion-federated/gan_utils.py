import tensorflow as tf

import numpy as np

def normalization (data):
  _, dim = data.shape
  norm_data = data.copy()
  
  min_val = np.zeros(dim)
  max_val = np.zeros(dim)
  
  for i in range(dim):
    min_val[i] = np.nanmin(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
    max_val[i] = np.nanmax(norm_data[:,i])
    norm_data[:,i] = (norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6))
    norm_data[:,i] = norm_data[:,i] * 2 - 1
    
  norm_parameters = {'min_val': min_val,
                     'max_val': max_val}
      
  return norm_data, norm_parameters

def renormalization (norm_data, norm_parameters):
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = (renorm_data[:,i] + 1) / 2
    renorm_data[:,i] = (renorm_data[:,i] * (max_val[i] + 1e-6))
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data

def binary_sampler(p, rows, cols):
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  