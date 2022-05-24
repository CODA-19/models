SELECT
	diagnosis_data.patient_site_uid,
	datetime (diagnosis_data.diagnosis_time) death_time
FROM
	diagnosis_data
	INNER JOIN patient_data ON diagnosis_data.patient_site_uid = patient_data.patient_site_uid
WHERE
	diagnosis_data.diagnosis_type = 'death'
	AND patient_data.patient_covid_status = 'positive'