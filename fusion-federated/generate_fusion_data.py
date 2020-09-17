import os, csv, sqlite3, time
import numpy as np
import seaborn as sns
import h5py
import pickle
from matplotlib import pyplot as plt 

from constants import SQLITE_DIRECTORY
from sqlite_utils import sql_fetch_all
from time_utils import get_hours_between_datetimes

from impute import findgan_impute

selected_variables = [
  'urea', 'sodium', 'creatinine', 'total_bilirubin', 'ast',
  'white_blood_cell_count', 'hemoglobin', 'platelet_count', 'lactic_acid',
  'systolic_blood_pressure', 'diastolic_blood_pressure', 'oxygen_saturation',
  'heart_rate', 'temperature', 'respiratory_rate', 'fraction_inspired_oxygen'
]

include_covid_negative = False

if include_covid_negative:
  inclusion_flag = ""
else:
  inclusion_flag = " AND patient_data.patient_covid_status = 'positive'"

db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

query_covid = "SELECT patient_data.patient_site_uid from patient_data WHERE " + \
        " patient_data.patient_covid_status = 'positive' "

query_icu = "SELECT episode_data.patient_site_uid from episode_data INNER JOIN " + \
        " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
        " (episode_data.episode_unit_type = 'intensive_care_unit' OR " + \
        "  episode_data.episode_unit_type = 'high_dependency_unit' OR " + \
        "  episode_data.episode_unit_type = 'inpatient_ward')  " + \
        inclusion_flag

query_deaths = "SELECT diagnosis_data.patient_site_uid from diagnosis_data INNER JOIN " + \
        " patient_data ON diagnosis_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
        " diagnosis_data.diagnosis_type = 'death' " + \
        inclusion_flag

covid_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_covid)])
icu_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_icu)])
death_pt_ids = set([str(x[0]) for x in sql_fetch_all(conn, query_deaths)])

query = "SELECT episode_data.patient_site_uid, episode_start_time from episode_data INNER JOIN " + \
         " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
         " (episode_data.episode_unit_type = 'intensive_care_unit' OR " + \
        "  episode_data.episode_unit_type = 'high_dependency_unit' OR " + \
        "  episode_data.episode_unit_type = 'inpatient_ward' OR " + \
         " episode_data.episode_unit_type = 'emergency_room')" + \
         inclusion_flag

res = sql_fetch_all(conn, query)

dead_patients = set()
eligible_patients = set()
eligible_episodes = {}

for patient_id, episode_start_time in res:
  patient_id = str(patient_id)
  episode_start_time = str(episode_start_time)

  if patient_id not in eligible_episodes:
    query = "SELECT patient_site_uid, diagnosis_time from diagnosis_data WHERE " \
          "diagnosis_type=\"death\" AND patient_site_uid=\""+patient_id+"\" "
  
    pcr_res = sql_fetch_all(conn, query)
    all_res = [[str(value[0]), str(value[1])] for value in pcr_res]
    
    if len(all_res) == 0:
      eligible_patients.add(patient_id)
      eligible_episodes[patient_id] = episode_start_time

    found_death_within = False

    for each_res in all_res:
      if each_res[1] == 'None' or len(each_res[1]) < 10: continue
      if get_hours_between_datetimes(each_res[1], episode_start_time) < 24*90:
        eligible_episodes[patient_id] = episode_start_time
        eligible_patients.add(patient_id)
        dead_patients.add(patient_id)
        found_death_within = True
        print('Admitted/died', episode_start_time, each_res[1])
    
    if not found_death_within:
      eligible_patients.add(patient_id)
      eligible_episodes[patient_id] = episode_start_time

query = "SELECT lab_data.patient_site_uid, lab_name, lab_sample_time, lab_result_value, lab_sample_type from lab_data " + \
  " INNER JOIN patient_data ON " + \
  " lab_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
  " (lab_data.lab_sample_type = 'venous_blood' OR " + \
  " lab_data.lab_sample_type = 'arterial_blood' OR " + \
  " lab_data.lab_sample_type = 'unspecified_blood') AND " + \
  " lab_data.lab_result_status = 'resulted' " + \
  inclusion_flag
  
lab_res = sql_fetch_all(conn, query)

query = "SELECT observation_data.patient_site_uid, observation_name, observation_time, observation_value from observation_data " + \
  " INNER JOIN patient_data ON " + \
  " observation_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
  " observation_data.observation_value IS NOT NULL " + \
  inclusion_flag
  
obs_res = sql_fetch_all(conn, query)

full_data = [[str(value[0]), str(value[1]), str(value[2]), float(value[3]), str(value[4])] for value in lab_res] + \
            [[str(value[0]), str(value[1]), str(value[2]), float(value[3]), ''] for value in obs_res]

ylabels = []

left_limit, right_limit, hours_per_period = [0, 72, 6]
left_offset = int(np.abs(left_limit))
n_time_points = int((right_limit - left_limit) / hours_per_period)

var_bins = {}
var_names = []
patients_with_data = {}
patient_ids = []
patient_num = 0
total_var_num = 0
total_entries_num = 0

percentages = [[]]
percentages_total = []
patient_infos = {}

for patient_id, var_name, var_sample_time, var_value, var_sample_type in full_data:
  if patient_id not in eligible_patients: continue
  if var_name not in selected_variables: continue
  if patient_id not in eligible_episodes: continue

  if patient_id not in var_bins:
    var_bins[patient_id] = {}
  if var_name not in var_names:
    var_names.append(var_name)
  if var_name not in patients_with_data:
    patients_with_data[var_name] = set()
  if var_name not in var_bins[patient_id]:
    var_bins[patient_id][var_name] = [None for x in range(0,n_time_points)]
  
  episode_start_time = eligible_episodes[patient_id]
  
  hours_since_admission = get_hours_between_datetimes(episode_start_time, var_sample_time)
  if hours_since_admission > left_limit and hours_since_admission < right_limit:
    bin_num = int(hours_since_admission / hours_per_period + left_offset / hours_per_period)
    #print('Labs', episode_start_time, var_sample_time, hours_since_admission, bin_num, hours_since_admission / hours_per_period, left_offset / hours_per_period, left_limit, right_limit)
    
    if bin_num >= n_time_points: continue
    if var_name in var_bins[patient_id] and \
      bin_num < len(var_bins[patient_id][var_name])-1 and \
      var_bins[patient_id][var_name][bin_num] is not None:
      # Pick the most abnormal (simplified here to highest)
      if var_value > var_bins[patient_id][var_name][bin_num]: 
        var_bins[patient_id][var_name][bin_num] = var_value
    else: 
      var_bins[patient_id][var_name][bin_num] = var_value
      patients_with_data[var_name].add(patient_id)
      total_var_num += 1
    total_entries_num += 1
  patient_ids.append(patient_id)
  patient_infos[patient_id] = [
    int(patient_id in death_pt_ids), 
    int(patient_id in icu_pt_ids)]

selected_variables = np.unique(var_names)
patient_ids = np.unique(patient_ids)
n_patients = len(patient_ids)

values_for_variables = []

for variable_name in selected_variables:
  values_for_variable = []
  for patient_id in patient_ids:
    if variable_name in var_bins[patient_id]:
      v = var_bins[patient_id][variable_name]
      values_for_variable.extend(v)
    else:
      values_for_variable.extend([None for i \
        in range(0,n_time_points)])

  values_for_variables.append(values_for_variable)

values_for_variables = np.asarray(values_for_variables)

# More intuitive naming
test_first_features = values_for_variables

# Prepare imputation using FindGAN
pat_first_features = []
selected_patient_ids = []

# Exclude patients that are filled with NaNs
for i, patient_id in enumerate(patient_ids):
  row_flat = test_first_features[:, i*n_time_points:n_time_points*(i + 1)]
  row_flat = np.asarray(row_flat.ravel())

  if len(row_flat[row_flat == None]) != row_flat.shape[0]:
    pat_first_features.append(row_flat)
    selected_patient_ids.append(patient_id)
  else:
    print('Excluding patient filled with NaNs')
    

pat_first_features = np.asarray(pat_first_features)
test_first_features = []

for i in range(0, pat_first_features.shape[1]):
  row_flat = pat_first_features[:,i].ravel()
  if len(row_flat[row_flat == None]) != row_flat.shape[0]:
    test_first_features.append(row_flat)
  else:
    feature_name = selected_variables[int(i/n_time_points)]
    feature_time = int(n_time_points*(i/n_time_points - int(i/n_time_points)))
    print('Excluding feature filled with NaNs', feature_name, feature_time)

pat_first_features = np.asarray(test_first_features).transpose()
print('Patients left: ', len(selected_patient_ids))
time.sleep(2)

# Perform imputation using FindGAN
#it = KNN(k=10)
#pat_first_features = it.fit_transform(pat_first_features)
# Transpose since we want each patient as a column (test_first)
pat_first_features = findgan_impute(pat_first_features) # knn.fit_transform(pat_first_features)

for x in pat_first_features: 
  print(x)

# Create outcome labels
patient_labels = []

i = 0
for patient_id in selected_patient_ids:
  is_dead = int(patient_id in dead_patients)
  patient_labels.append(is_dead)
  i += 1

patient_labels = np.asarray(patient_labels).reshape([-1,1])

# Filter entries, keep only those with chest X-ray within +/- 48h
selected_features = []
selected_labels = []
selected_imagings = []

for j, patient_id in enumerate(selected_patient_ids):
  episode_start_time = eligible_episodes[patient_id]

  query = "SELECT imaging_data.patient_site_uid, imaging_accession_uid, imaging_acquisition_time from imaging_data WHERE " + \
          "patient_site_uid = \"" + patient_id + "\" AND imaging_modality=\"xr\""

  res = sql_fetch_all(conn, query)

  for imaging_study in res:

    patient_id = imaging_study[0]
    imaging_accession_uid = imaging_study[1]
    imaging_acquisition_time = imaging_study[2]
    
    if imaging_acquisition_time is None: continue

    hours_since_admission = get_hours_between_datetimes(
      episode_start_time, imaging_acquisition_time)

    # If within timeframe, retrieve the slice data for the study
    if np.abs(hours_since_admission) < 96:
      print('Imaging', episode_start_time, imaging_acquisition_time)
      query = "SELECT slice_data.slice_data_uri from slice_data WHERE " + \
          "imaging_accession_uid = \"" + imaging_accession_uid + "\""

      res_slice = sql_fetch_all(conn, query)

      # If slice can't be found, skip this imaging study
      if len(res_slice) == 0: continue
      
      # If there is an associated slice, keep the patient
      print('Found imaging for', j)
      imp_feat_tmp = pat_first_features[j]
      pat_lab_tmp = patient_labels[j]

      selected_imagings.append(res_slice[0][0])
      selected_features.append(imp_feat_tmp.ravel())
      selected_labels.append(pat_lab_tmp)

      # Maximum one study per patient
      break

selected_features = np.asarray(selected_features)
selected_labels = np.asarray(selected_labels)
selected_imagings = np.asarray(selected_imagings)

print(selected_features.shape)
print(selected_labels.shape)
print(selected_imagings.shape)

data = {
  'features': selected_features,
  'labels': selected_labels,
  'imaging': selected_imagings
}

with open('fusion_data.sav', 'wb') as f:
  pickle.dump(data, f)
