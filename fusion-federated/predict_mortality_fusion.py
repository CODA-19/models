import os, csv, sqlite3
import h5py
import pickle

import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt 
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold


from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, Dropout, Conv2D, Input, Dense, \
  Flatten, MaxPooling2D, BatchNormalization, GaussianDropout, concatenate
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

IMAGING_DIR = '/data8/projets/dev_scratch_space/lmullie/scratch/BLOB_export'
DATA_FILE = 'fusion_data.sav'
n_features = 16
n_time_bins = 12
img_size = 128

data = pickle.load(open(DATA_FILE, 'rb'))

selected_features = []
selected_labels = []
selected_imagings = []

for i, imaging_file in enumerate(data['imaging'][0:400]):

  print(i)
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

# Split into 5 folds (80% for training/validation and 20% for testing)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plt.figure()
fold = 0
tprs = []
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in kfold.split(X,y):
  
  X_feat_train = np.asarray([X[i][0] for i in train_index])
  X_img_train = np.asarray([X[i][1] for i in train_index])
  y_train = np.asarray([y[i] for i in train_index]).ravel()

  X_feat_test = np.asarray([X[i][0] for i in test_index])
  X_img_test = np.asarray([X[i][1] for i in test_index])
  y_test = np.asarray([y[i] for i in test_index]).ravel()
  
  # Tone down effect of class imbalance using class weighing
  ones_weight = 1 / (np.count_nonzero(y_train == 1) / y_train.shape[0])
  
  # Ensure no variable reuse; create new optimizer every time
  optimizer = Adam(lr=1e-5)
  classifier = Model([input_numbers, input_imaging], l_output)

  classifier.compile(
    optimizer = optimizer, 
    loss='mse', 
    metrics = ['accuracy']
  )

  #classifier.summary()
  plot_model(classifier, to_file='classifier.dot', show_shapes=True)

  es_val = EarlyStopping(
    monitor='val_acc', 
    min_delta=0.01, 
    patience=10, 
    restore_best_weights=True)

  # Perform classification with early stopping
  classifier.fit([X_feat_train, X_img_train], y_train,
    batch_size=32,
    epochs=150,
    verbose=True,
    callbacks=[es_val],
    class_weight={0: 1, 1: ones_weight},
    validation_split=0.30)

  # Predict class probabilities
  y_score = classifier.predict([X_feat_test, X_img_test])
  y_score = np.asarray(y_score).astype(np.float32)

  # Obtain corresponding classes
  y_score[y_score > 0.5] = 1
  y_score[y_score < 1] = 0

  # Plot ROC curve
  fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
  tprs.append(np.interp(mean_fpr, fpr, tpr))
  roc_auc = auc(fpr, tpr)

  print('Fold %i (AUC = %0.2f)' % (fold, roc_auc))
  plt.plot(fpr, tpr, lw=1, color='gray', linestyle='--', \
    label='Fold %i (AUC = %0.2f)' % (fold, roc_auc))

  fold += 1

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, lw=2, color='green', \
  label='Average %i (AUC = %0.2f)' % (fold, mean_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Prediction of 90-day mortality based on labs, vital signs and chest X-ray')
plt.legend(loc="lower right")
plt.show()

print('Number of patients (total): %d' % n_selected_patients)
print('Number of + events (train): %d' % np.count_nonzero(y_train))
print('Number of + events (test): %d' % np.count_nonzero(y_test))
