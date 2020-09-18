import numpy as np
from matplotlib import pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

from load_fusion_data import load_fusion_data
from get_fusion_classifier import get_fusion_classifier

from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

IMAGING_DIR = '/data8/projets/dev_scratch_space/lmullie/scratch/BLOB_export'
DATA_FILE = 'fusion_data.sav'
n_features = 16
n_time_bins = 12
img_size = 128

X, y = load_fusion_data(DATA_FILE, IMAGING_DIR, n_features, n_time_bins, img_size)

img_size = 192

# Split into 3 folds (80% for training/validation and 20% for testing)
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

all_X = np.copy(X)
all_y = np.copy(y)
site = 0

for other_sites_index, current_site_index in kfold.split(all_X):

  site_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
  X = all_X[current_site_index]
  y = all_y[current_site_index]
  
  site += 1
  fold = 0

  plt.figure()

  tprs = []  
  mean_fpr = np.linspace(0, 1, 100)

  for train_index, test_index in site_kfold.split(X,y):
  
    X_feat_train = np.asarray([X[i][0] for i in train_index])
    X_img_train = np.asarray([X[i][1] for i in train_index])
    y_train = np.asarray([y[i] for i in train_index]).ravel()
 
    X_feat_test = np.asarray([X[i][0] for i in test_index])
    X_img_test = np.asarray([X[i][1] for i in test_index])
    y_test = np.asarray([y[i] for i in test_index]).ravel()
  
    # Tone down effect of class imbalance using class weighing
    ones_weight = 1 / (np.count_nonzero(y_train == 1) / y_train.shape[0])
  
    # Ensure no variable reuse; create new optimizer every time
    classifier = get_fusion_classifier(n_features, n_time_bins, img_size)

    #classifier.summary()
    #plot_model(classifier, to_file='classifier.dot', show_shapes=True)

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
