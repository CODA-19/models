import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, Dropout, Conv2D, Input, Dense, \
  Flatten, MaxPooling2D, BatchNormalization, GaussianDropout, concatenate

def get_fusion_classifier(n_features, n_time_bins, img_size, lr=1e-5):

    # CNN to process multivariate numerical input
  input_numbers = Input(shape=(n_features, n_time_bins,1))
  l1_numbers = Conv2D(32, kernel_size=4, activation="relu",
    input_shape=(n_features, n_time_bins, 1))(input_numbers)
  l1_numbers = BatchNormalization()(l1_numbers)
  l2_numbers = MaxPooling2D(pool_size = (2, 2))(l1_numbers)
  l3_numbers = Conv2D(32, kernel_size=2, activation="relu")(l2_numbers)
  l3_numbers = Dropout(0.15)(l3_numbers)
  l4_numbers = MaxPooling2D(pool_size = (2, 2))(l3_numbers)
  l5_numbers = Flatten()(l4_numbers)
  numbers_pre_output = Dense(32, activation="relu", 
    kernel_regularizer=tf.keras.regularizers.l1(1e-3))(l5_numbers)

  # CNN to process imaging
  input_imaging = Input(shape=(img_size, img_size,1))
  l1_imaging = Conv2D(32, kernel_size=3, activation="relu",
    input_shape=(img_size, img_size, 1))(input_imaging)
  l2_imaging = BatchNormalization()(l1_imaging)
  l3_imaging = MaxPooling2D(pool_size = (2, 2))(l2_imaging)
  l4_imaging = Conv2D(32, kernel_size=3, activation="relu")(l3_imaging)
  l4_imaging = GaussianDropout(0.15)(l4_imaging)
  l5_imaging = MaxPooling2D(pool_size = (2, 2))(l4_imaging)
  l6_imaging = Flatten()(l5_imaging)
  imaging_pre_output = Dense(32, activation="relu")(l6_imaging)

  # Pool 2 CNNs and then apply batch normalization
  l_combined = concatenate([numbers_pre_output, imaging_pre_output], axis=-1)
  l_combined_bn = BatchNormalization()(l_combined)
  l_scaled = Dense(activation = "tanh", units = 32)(l_combined_bn)
  l_output = Dense(1, activation="sigmoid")(l_scaled)

  # In this case, one global classifier that is updated incrementally
  optimizer = Adam(lr=lr)
  classifier = Model([input_numbers, input_imaging], l_output)

  classifier.compile(
    optimizer = optimizer, 
    loss='mse', 
    metrics = ['accuracy']
  )

  return classifier