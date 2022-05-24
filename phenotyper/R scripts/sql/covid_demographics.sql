SELECT
	patient_site_uid,
	datetime("2021-01-01") - datetime(patient_birth_date) AS patient_age,
	patient_sex
FROM
	patient_data
WHERE
	patient_data.patient_covid_status = 'positive'