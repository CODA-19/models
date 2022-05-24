-- This version returns first labs for all patients within 24h of admission in narrow format
-- Because SQLITE does not support the FIRST aggregate function impossible to do directly
-- In this version labs for all the duration
-- Can easily modify script to modify for different hours
WITH labs_sample_first AS (
WITH labs_sample AS (
	-- In this script we are taking into consideration the ENTIRE COVID EPISODE AS ONE WITH covidepisodes AS (
WITH covidepisodes AS (
WITH mergedepisodes AS (
WITH episodes AS (
				SELECT
					episode_admission_uid,
					patient_site_uid,
					episode_start_time,
					episode_end_time,
					SUM(flag) OVER (PARTITION BY patient_site_uid ORDER BY episode_start_time) stayid
				FROM (
				SELECT
					*,
					strftime ('%s',
					episode_start_time) - strftime ('%s',
				LAG(episode_end_time,
				1,
				datetime (episode_start_time,
				'-1 hour')) OVER (PARTITION BY patient_site_uid ORDER BY episode_start_time)) > 12 * 3600 flag -- 12 hours delay considered as same single episode
		FROM
			episode_data))
SELECT
	patient_site_uid,
	min(episode_start_time) episode_start_time,
	max(episode_end_time) episode_end_time,
	stayid
FROM
	episodes
GROUP BY
	patient_site_uid,
	stayid
)
SELECT
	mergedepisodes.patient_site_uid,
	mergedepisodes.episode_start_time,
	mergedepisodes.episode_end_time
FROM
	mergedepisodes
	INNER JOIN patient_data ON mergedepisodes.patient_site_uid = patient_data.patient_site_uid
WHERE
	datetime (patient_data.pcr_sample_time) BETWEEN datetime (mergedepisodes.episode_start_time,
'-7 day') -- we consider a covid episode when positive test was done in the 7 days preceding the admission
	AND datetime (mergedepisodes.episode_end_time)
AND patient_data.patient_covid_status = 'positive'
)
SELECT
	covidepisodes.patient_site_uid,
	covidepisodes.episode_start_time,
	covidepisodes.episode_end_time,
	lab_name,
	lab_sample_type,
	lab_sample_time,
	lab_result_value
FROM
	covidepisodes
	INNER JOIN lab_data ON covidepisodes.patient_site_uid = lab_data.patient_site_uid
	-- Can modulate time here to make sure the lab occured at a specific time from the onset of hospitalization i.e. first 24 hours
WHERE
	datetime (lab_sample_time) BETWEEN datetime (covidepisodes.episode_start_time, '-24 hour') 
	AND datetime (covidepisodes.episode_start_time, '+72 hour')
)
SELECT
	patient_site_uid,
	lab_name,
	lab_sample_type,
	FIRST_VALUE(lab_result_value) OVER (
	PARTITION BY lab_name, lab_sample_type, patient_site_uid
	ORDER BY lab_sample_time) AS lab_result_value
FROM 
	labs_sample
GROUP BY
	patient_site_uid, lab_name, lab_sample_type
	)
SELECT
	patient_site_uid, -- min(lab_result_value) AS lab_min
	-- *CBC**
	-- Hemoglobin
	min(CASE WHEN lab_name = 'hemoglobin' THEN
		lab_result_value
	ELSE
		NULL
	END) AS hemoglobin,
	-- Platelet
	min(
		CASE WHEN lab_name = 'platelet_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS plt,
	-- WBC
	min(
		CASE WHEN lab_name = 'white_blood_cell_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS wbc,
	-- *CHEM*
	-- Albumin
	min(
		CASE WHEN lab_name = 'albumin' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS albumin,
	-- Globulins
	min(
		CASE WHEN lab_name = 'globulins' THEN
			lab_result_value
		ELSE
			NULL
		END) AS globulin,
	-- Total Protein
	min(
		CASE WHEN lab_name = 'total_protein' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS protein,
	-- Sodium
	min(
		CASE WHEN lab_name = 'sodium' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS sodium,
	-- Chloride
	min(
		CASE WHEN lab_name = 'chloride' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS chloride,
	-- Potassium
	min(
		CASE WHEN lab_name = 'potassium' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS potassium,
	-- Bicarbonate
	min(
		CASE WHEN lab_name = 'bicarbonate' THEN
			lab_result_value
		ELSE
			NULL
		END) AS bicarbonate,
	-- BUN
	min(
		CASE WHEN lab_name = 'urea' THEN
			lab_result_value
		ELSE
			NULL
		END) AS bun,
	-- Calcium
	min(
		CASE WHEN lab_name = 'corrected_total_calcium' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS calcium,
	-- Magnesium
	min(
		CASE WHEN lab_name = 'magnesium' THEN
			lab_result_value
		ELSE
			NULL
		END) AS magnesium, 
	-- Total Phosphate
	min(
		CASE WHEN lab_name = 'phosphate' THEN
			lab_result_value
		ELSE
			NULL
		END) AS phosphate,
	-- Creatinine
	min(
		CASE WHEN lab_name = 'creatinine' AND lab_sample_type != 'urine' THEN
			lab_result_value
		ELSE
			NULL
		END) AS creatinine,
	-- DFG
	min(
		CASE WHEN lab_name = 'estimated_gfr' THEN
			lab_result_value
		ELSE
			NULL
		END) AS gfr,
	-- Glucose
	min(
		CASE WHEN lab_name = 'glucose' THEN
			lab_result_value
		ELSE
			NULL
		END) AS glucose,
	-- AnionGAP original
	min(
		CASE WHEN lab_name = 'anion_gap' THEN
			lab_result_value
		ELSE
			NULL
		END) AS anion_gap,
-- Total **DIFF**
	-- Eosinophils
	min(
		CASE WHEN lab_name = 'eosinophil_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS eos,
	-- Lymphocytes
	min(
		CASE WHEN lab_name = 'lymphocyte_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS lymph,
	-- Neutrophils
	min(
		CASE WHEN lab_name = 'neutrophil_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS neutrophil,
	-- Monocytes
	min(
		CASE WHEN lab_name = 'monocyte_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS mono,
	-- Basophils
	min(
		CASE WHEN lab_name = 'basophil_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS baso,
	-- Bands
	min(
		CASE WHEN lab_name = 'stab_count' THEN
			lab_result_value
		ELSE
			NULL
		END) AS stab,
	-- atypicals, bands, not available
	-- Total **COAG**
	-- PT
	min(
		CASE WHEN lab_name = 'thrombin_time' THEN
			lab_result_value
		ELSE
			NULL
		END) AS PT,
	-- PTT
	min(
		CASE WHEN lab_name = 'partial_thromboplastin_time' THEN
			lab_result_value
		ELSE
			NULL
		END) AS PTT,
	-- Fibrinogen
	min(
		CASE WHEN lab_name = 'fibrinogen' THEN
			lab_result_value
		ELSE
			NULL
		END) AS fibrinogen,
	-- DDimer
	min(
		CASE WHEN lab_name = 'd_dimer' THEN
			lab_result_value
		ELSE
			NULL
		END) AS d_dimer,
	-- Total **Enzymes**
	-- ALT
	min(
		CASE WHEN lab_name = 'alanine_aminotransferase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS alt,
	-- AST
	min(
		CASE WHEN lab_name = 'ast' THEN
			lab_result_value
		ELSE
			NULL
		END) AS ast,
	-- PALC
	min(
		CASE WHEN lab_name = 'alkaline_phosphatase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS palc,
	-- GGT
	min(
		CASE WHEN lab_name = 'gamma_glutamyl_transferase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS ggt,
	-- Amylase
	min(
		CASE WHEN lab_name = 'amylase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS amylase,
	-- Lipase
	min(
		CASE WHEN lab_name = 'lipase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS lipase,
	-- Bili_total
	min(
		CASE WHEN lab_name = 'total_bilirubin' THEN
			lab_result_value
		ELSE
			NULL
		END) AS bili_tot,
	-- Bili_direct
	min(
		CASE WHEN lab_name = 'direct_bilirubin' THEN
			lab_result_value
		ELSE
			NULL
		END) AS bili_direct,
	-- Bili_indirect
	min(
		CASE WHEN lab_name = 'indirect_bilirubin' THEN
			lab_result_value
		ELSE
			NULL
		END) AS bili_indirect,
	-- Lipase
	min(
		CASE WHEN lab_name = 'lipase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS lipase,
	-- CK
	min(
		CASE WHEN lab_name = 'creatine_kinase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS ck,
	-- CK-MB
	min(
		CASE WHEN lab_name = 'ck_mb' THEN
			lab_result_value
		ELSE
			NULL
		END) AS ckmb,
	-- LDH
	min(
		CASE WHEN lab_name = 'lactate_dehydrogenase' THEN
			lab_result_value
		ELSE
			NULL
		END) AS ldh,
	-- TROPOS
	min(
		CASE WHEN lab_name = 'hs_troponin_t' THEN
			lab_result_value
		ELSE
			NULL
		END) AS tropot,
	-- Lactate
	min(
		CASE WHEN lab_name = 'lactic_acid' THEN
			lab_result_value
		ELSE
			NULL
		END) AS lactate,
	-- Oxygenation
	-- O2 sat
	min(
		CASE WHEN lab_name = 'o2_sat' AND lab_sample_type = 'venous_blood' THEN
			lab_result_value
		ELSE
			NULL
		END) AS svo2sat_min,
	-- PAO2
	min(
		CASE WHEN lab_name = 'po2' AND lab_sample_type = 'arterial_blood' THEN
			lab_result_value
		ELSE
			NULL
		END) AS pao2,
	-- PVO2
	min(
		CASE WHEN lab_name = 'po2' AND lab_sample_type = 'venous_blood' THEN
			lab_result_value
		ELSE
			NULL
		END) AS pvo2,
	-- PACO2
	min(
		CASE WHEN lab_name = 'pco2' AND lab_sample_type = 'venous_blood' THEN
			lab_result_value
		ELSE
			NULL
		END) AS paco2,
	-- PVCO2
	min(
		CASE WHEN lab_name = 'pco2' AND lab_sample_type = 'venous_blood' THEN
			lab_result_value
		ELSE
			NULL
		END) AS pvco2,
	-- Total **Other**
	-- TSH
	min(
		CASE WHEN lab_name = 'thyroid_stimulating_hormone' THEN
			lab_result_value
		ELSE
			NULL
		END) AS tsh,
	-- Vitamin D
	min(
		CASE WHEN lab_name = '25_oh_vitamin_d' THEN
			lab_result_value
		ELSE
			NULL
		END) AS vitd,
	-- CRP
	min(
		CASE WHEN lab_name = 'c_reactive_protein' THEN
			lab_result_value
		ELSE
			NULL
		END) AS crp,
	-- Ferritin
	min(
		CASE WHEN lab_name = 'ferritin' THEN
			lab_result_value
		ELSE
			NULL
		END) AS ferritin,
	-- BNP
	min(
		CASE WHEN lab_name = 'nt_pro_bnp' THEN
			lab_result_value
		ELSE
			NULL
		END) AS bnp
	-- AnionGAP_calculated
	-- sodium_mean - bicarbonate_mean - chloride_mean AS anion_gap_calc,
FROM
	labs_sample_first
GROUP BY
	patient_site_uid