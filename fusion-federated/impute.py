import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
from findgan import FindGAN
from gan_utils import normalization, renormalization, binary_sampler
from matrix_utils import nan_to_csr
import matplotlib.pyplot as plt
def findgan_impute(data_x):
  
  data_x[data_x == None] = np.nan
  data_x = data_x.astype(np.float32)
  
  # Define mask matrix
  data_m = 1-np.isnan(data_x)

  # System parameters
  no, dim = data_x.shape
  batch_size = 128
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  sparse_data_x = nan_to_csr(data_x, data_m)

  if np.count_nonzero(np.isnan(norm_data_x)) > 0:
    print('Warning! NaNs remaining on GAN input.')
    exit()

  # Train FindGAN
  sess = tf.Session()
  gan = FindGAN(dim, batch_size)
  
  imputed_data_norm = gan.train(sess, norm_data_x, data_m, sparse_data_x)
  
  # Renormalization
  imputed_data = renormalization(imputed_data_norm, norm_parameters)  
  #imputed_data = rounding(imputed_data, data_x)  
  
  return imputed_data
