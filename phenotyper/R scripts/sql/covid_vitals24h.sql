WITH vs_sample_first AS (
WITH vs_sample AS (
-- In this script we are taking into consideration the ENTIRE COVID EPISODE AS ONE
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
	observation_name,
	observation_time,
	observation_value
FROM
	covidepisodes
	INNER JOIN observation_DATA ON covidepisodes.patient_site_uid = observation_data.patient_site_uid
	-- Can modulate time here to make sure the lab occured at a specific time from the onset of hospitalization i.e. first 24 hours
WHERE
	datetime (observation_time) BETWEEN datetime (covidepisodes.episode_start_time, '-25 hour') AND datetime (covidepisodes.episode_start_time, '+25 hour')
)
SELECT
	patient_site_uid,
	observation_name,
	FIRST_VALUE(observation_value) OVER (
	PARTITION BY observation_name, patient_site_uid
	ORDER BY datetime(observation_time)) AS observation_value
FROM 
	vs_sample
GROUP BY
	patient_site_uid, observation_name
	)
SELECT
	patient_site_uid,
	-- weight
	min( CASE WHEN observation_name = 'weight' THEN
		observation_value
	ELSE
		NULL
	END) AS weight,
	-- systolic_blood_pressure
	min( CASE WHEN observation_name = 'systolic_blood_pressure' THEN
		observation_value
	ELSE
		NULL
	END) AS sbp, 
	-- weight
	min( CASE WHEN observation_name = 'diastolic_blood_pressure' THEN
		observation_value
	ELSE
		NULL
	END) AS dbp,
	-- temperature
	min( CASE WHEN observation_name = 'temperature' THEN
		observation_value
	ELSE
		NULL
	END) AS temp, 
	-- so2
	min( CASE WHEN observation_name = 'oxygen_saturation' THEN
		observation_value
	ELSE
		NULL
	END) AS so2,
	-- respiratory_rate
	min( CASE WHEN observation_name = 'respiratory_rate' THEN
		observation_value
	ELSE
		NULL
	END) AS rr,
	-- oxygen flow
	min( CASE WHEN observation_name = 'oxygen_flow_rate' THEN
		observation_value
	ELSE
		NULL
	END) AS flow,
	-- fi02
	min( CASE WHEN observation_name = 'fraction_inspired_oxygen' THEN
		observation_value
	ELSE
		NULL
	END) AS fio2
FROM
	vs_sample_first
GROUP BY
	patient_site_uid
