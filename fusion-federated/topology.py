import tensorflow as tf
import numpy as np

def batch_normalization(batch, depth):
  
  beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
  gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
  batch_mean, batch_var = tf.nn.moments(batch, [0], name='moments', keepdims=True)
  
  batch_normed = tf.nn.batch_normalization(batch, batch_mean, batch_var, beta, gamma, 1e-6)
  
  return batch_normed

def middle_layer(x_in, size=32, batch_norm=False, dropout=None, l1_reg=0.015, l2_reg=0.0001):
  
  x_out = tf.layers.dense(x_in,size,
    activation=tf.nn.swish,
    kernel_regularizer=tf.keras.regularizers.l1(l1_reg),
    activity_regularizer=tf.keras.regularizers.l2(l2_reg),
    kernel_initializer=tf.keras.initializers.he_uniform())
  
  if batch_norm:
    x_out = batch_normalization(x_out, size)
  
  if dropout is not None:
    x_out = tf.nn.dropout(x_out, rate=dropout)
  
  return x_out

def skip_layer(x_in, x_orig):
  
  x_out = tf.concat(values=[x_in, x_orig], axis=1)
  
  return x_out

def output_layer(x_in, size=32, l1_reg=0.01, l2_reg=0.0001, activation=tf.nn.tanh):
  
  x_out = tf.layers.dense(x_in,size,
    activation=activation,
    kernel_regularizer=tf.keras.regularizers.l1(l1_reg),
    activity_regularizer=tf.keras.regularizers.l2(l2_reg),
    kernel_initializer=tf.keras.initializers.he_uniform())
  
  return x_out

def downsample_block(inpt, dropout=None):
  
  l1 = middle_layer(inpt, size=32, batch_norm=True, dropout=dropout, l1_reg=0.05, l2_reg=0)
  l2 = middle_layer(l1, size=28, batch_norm=True, dropout=dropout, l1_reg=0.05, l2_reg=0)
  l3 = middle_layer(l2, size=24, batch_norm=True, dropout=dropout, l1_reg=0.05, l2_reg=0)
  l4 = middle_layer(l3, size=20, batch_norm=True, dropout=dropout, l1_reg=0.05, l2_reg=0)
  l5 = middle_layer(l4, size=16, batch_norm=True, dropout=dropout, l1_reg=0.05, l2_reg=0)
  
  return l5

def stacked_downsample_block(inpt, out_size=32, activation=tf.nn.tanh):
  
  l1 = downsample_block(inpt, dropout=0.5)
  l2 = downsample_block(inpt, dropout=0.4)
  l3 = downsample_block(inpt, dropout=0.3)
  l4 = downsample_block(inpt, dropout=0.2)
  l5 = downsample_block(inpt, dropout=0.1)
  
  l6 = tf.concat(values=[l1,l2,l3,l4,l5], axis=1)
  l7 = output_layer(l6, size=out_size, activation=activation)
  
  return l7

def upsample_residual_block(inpt, orig, dropouts=[0, 0, 0, 0], l1_reg=0.02, out_size=32, activation=tf.nn.tanh):
  
  l1 = middle_layer(inpt, size=32, batch_norm=True, dropout=dropouts[0])
  l2 = middle_layer(l1, size=48, batch_norm=True, dropout=dropouts[1])
  l3 = middle_layer(l2, size=64, batch_norm=True, dropout=dropouts[2])
  l4 = middle_layer(l3, size=128, batch_norm=True, dropout=dropouts[3], l1_reg=l1_reg)
  l5 = output_layer(l4, size=32)
  l6 = skip_layer(l5, orig)
  l7 = middle_layer(l6, size=64)
  l8 = output_layer(l7, size=out_size, activation=activation)
  
  return l8

class BaseNetwork(object):
 
  @property
  def vars(self):
    return list(filter(
        lambda var: self.name in var.name,
        tf.global_variables()
    ))
  
class Encoder(BaseNetwork):
  
  def __init__(self, name='GAN/encoder', activation=tf.nn.tanh, out_size=32):
    self.name = name
    self.activation = activation
    self.out_size = out_size
      
  def __call__(self, x, use_dropout=False):
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
      
      l1 = stacked_downsample_block(x, activation=self.activation, out_size=self.out_size)
      l2 = output_layer(l1, size=x.shape[1], activation=self.activation)
      
      return l2

class Decoder(BaseNetwork):
  
  def __init__(self, name='GAN/decoder', activation=tf.nn.tanh, out_size=32):
    self.name = name
    self.activation = activation
    self.out_size = out_size
      
  def __call__(self, x):
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
      return upsample_residual_block(x, x, activation=self.activation, out_size=self.out_size)
      
class MaskGenerator(BaseNetwork):
  
  def __init__(self, name='GAN/mask_generator'):
    self.name = name
      
  def __call__(self, e, x):
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
      
      inpt = tf.concat(values=[e, x], axis=1)
      return upsample_residual_block(inpt, x, out_size=e.shape[1])
 
class MaskDiscriminator(BaseNetwork):
  
  def __init__(self, name='GAN/mask_discriminator'):
    self.name = name

  def __call__(self, m, x):

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as var_scope:
      inpt = tf.concat(values=[m, x], axis=1)
      return upsample_residual_block(inpt, m, out_size=m.shape[1])
      
class DataGenerator(BaseNetwork):
  
  def __init__(self, name='GAN/data_generator'):
    self.name = name
      
  def __call__(self, z, m):
  
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
      
      inpt = tf.concat(values=[z, m], axis=1)
      return upsample_residual_block(inpt, m, dropouts=[0.1, 0.2, 0.25, 0.3], l1_reg=0.001, out_size=z.shape[1])
      
      
class DataDiscriminator(BaseNetwork):
  
  def __init__(self, name='GAN/data_discriminator'):
    self.name = name

  def __call__(self, x, m, training_phase=None):
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as var_scope:
      inpt = tf.concat(values=[x, m], axis=1)

      return upsample_residual_block(inpt, x, dropouts=[0.02, 0.03, 0.04, 0.05], out_size=x.shape[1])
      