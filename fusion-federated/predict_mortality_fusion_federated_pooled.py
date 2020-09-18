import numpy as np
from matplotlib import pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from get_fusion_classifier import get_fusion_classifier
from load_fusion_data import load_fusion_data

IMAGING_DIR = '/data8/projets/dev_scratch_space/lmullie/scratch/BLOB_export'
DATA_FILE = 'fusion_data.sav'
n_features = 16
n_time_bins = 12
img_size = 128

# Load fusion data set
X, y = load_fusion_data(DATA_FILE, IMAGING_DIR, n_features, n_time_bins, img_size)

img_size = 192

# Create a copy of the original (full) data set
all_X = np.copy(X)
all_y = np.copy(y)

# Get the fusion classifier
classifier = get_fusion_classifier(n_features, n_time_bins, img_size, lr=0.5*1e-5)

# Split into 3 sites at random (no stratification)
global_kfold = KFold(n_splits=3, shuffle=True, random_state=42)
global_kfolds = global_kfold.split(all_X)

site = 0

X_feat_test_all, X_img_test_all, y_test_all = [], [], []

for other_sites_index, current_site_index in global_kfolds:
  
  site += 1
  print('Site: %s' % (site))

  site_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  X = all_X[current_site_index]
  y = all_y[current_site_index]
  
  # Split 80% training / validation and hold out 20% for testing at the end
  train_val_index, test_index = next(site_kfold.split(X,y))
  
  X_feat_train_val = np.asarray([X[i][0] for i in train_val_index])
  X_img_train_val = np.asarray([X[i][1] for i in train_val_index])
  y_train_val = np.asarray([y[i] for i in train_val_index]).ravel()

  X_feat_test = np.asarray([X[i][0] for i in test_index])
  X_img_test = np.asarray([X[i][1] for i in test_index])
  y_test = np.asarray([y[i] for i in test_index]).ravel()
  
  X_feat_test_all.append(X_feat_test)
  X_img_test_all.append(X_img_test)
  y_test_all.append(y_test)

  # Now do K-fold cross validation in the 80% of non-held out data
  X = X[train_val_index]
  y = y[train_val_index]

  # Train in multiple rounds: coarse to fine grained
  for patience in [3, 5]:
    
    fold = 0
    subsite_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for training_index, validation_index in subsite_kfold.split(X, y):
    
      fold += 1
      print('Fold: %s' % (fold))

      X_feat_training = np.asarray([X[i][0] for i in training_index])
      X_img_training = np.asarray([X[i][1] for i in training_index])
      y_training = np.asarray([y[i] for i in training_index]).ravel()

      X_feat_val = np.asarray([X[i][0] for i in validation_index])
      X_img_val = np.asarray([X[i][1] for i in validation_index])
      y_val = np.asarray([y[i] for i in validation_index]).ravel()
 
      ones_weight = 1 / (np.count_nonzero(y == 1) / y.shape[0])

      es_val = EarlyStopping(
        monitor='val_acc', 
        min_delta=0.01, 
        patience=patience, 
        restore_best_weights=True)
  
      # Perform classification with early stopping
      classifier.fit([X_feat_training, X_img_training], y_training,
        batch_size=32,
        epochs=10,
        verbose=True,
        callbacks=[es_val],
        class_weight={0: 1, 1: ones_weight},
        validation_data=[[X_feat_val, X_img_val], y_val])

# Test on combined held out data sets
X_feat_test_all = np.vstack(X_feat_test_all)
X_img_test_all = np.vstack(X_img_test_all)
y_test_all = np.hstack(y_test_all)

plt.figure()

tprs = []  
mean_fpr = np.linspace(0, 1, 100)

# Predict class probabilities
y_score = classifier.predict([X_feat_test_all, X_img_test_all])
y_score = np.asarray(y_score).astype(np.float32)

# Obtain corresponding classes
y_score[y_score > 0.5] = 1
y_score[y_score < 1] = 0

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_all.ravel(), y_score.ravel())
tprs.append(np.interp(mean_fpr, fpr, tpr))
roc_auc = auc(fpr, tpr)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, lw=2, color='green', \
  label='AUC = %0.2f' % (mean_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Prediction of 90-day mortality based on labs, vital signs and chest X-ray')
plt.legend(loc="lower right")
plt.show()