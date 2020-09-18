import os, csv, sqlite3
import h5py
import pickle

import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt 
from PIL import Image

def load_fusion_data(DATA_FILE, IMAGING_DIR, n_features, n_time_bins, img_size):
  data = pickle.load(open(DATA_FILE, 'rb'))

  selected_features = []
  selected_labels = []
  selected_imagings = []

  for i, imaging_file in enumerate(data['imaging'][0:400]):

    print('Loading image file', i)
    if imaging_file is None: 
      print('No imaging file for patient.')
      exit()
    
    # Open data file
    slice_pixels = None
    moved_imaging_file = [IMAGING_DIR] + imaging_file.split('/')[4:]
    moved_imaging_file = '/'.join(moved_imaging_file)

    with h5py.File(moved_imaging_file, 'r') as data_file:
      slice_pixels = data_file['dicom'][:]

    patient_features = data['features'][i,:].reshape((n_features, n_time_bins))
    
    # Remove PA
    slice_pixels = np.array(Image.fromarray(slice_pixels).resize((img_size*2,img_size*2)))
    slice_pixels = slice_pixels[int(img_size/4):int(img_size*2-img_size/4),int(img_size/4):int(img_size*2-img_size/4)] 
    selected_features.append(patient_features)
    
    # Equalize histogram
    equalized_pixels = np.sort(slice_pixels.ravel()).searchsorted(slice_pixels)
    selected_labels.append(data['labels'][i])
    selected_imagings.append(equalized_pixels)

    # Show example chest X-rays
    #plt.imshow(equalized_pixels)
    #plt.show()

  img_size = 192

  n_selected_patients = len(selected_imagings)

  # Wrap in Numpy arrays and cast everything to float32
  selected_features = np.asarray(selected_features).astype(np.float32)
  selected_features -= np.min(selected_features)
  selected_features /= np.max(selected_features)
  selected_features = np.expand_dims(selected_features, -1)

  # Normalize pixel data to 0..1
  selected_imagings = np.asarray(selected_imagings).astype(np.float32)
  selected_imagings -= np.min(selected_imagings)
  selected_imagings /= np.max(selected_imagings)
  selected_imagings = np.expand_dims(selected_imagings, -1)

  # Resize features, labels and imagingx
  selected_labels = np.asarray(selected_labels).astype(int)
  selected_labels = selected_labels.reshape((-1,1))

  # Inspect shapes
  print(selected_features.shape)
  print(selected_imagings.shape)
  print(selected_labels.shape)

  # Stack features and imaging (use NumPy cuz different sizes)
  X = [[selected_features[i,:,:], selected_imagings[i,:,:]] \
    for i in range(0,n_selected_patients)]
  y = selected_labels

  return [X, y]