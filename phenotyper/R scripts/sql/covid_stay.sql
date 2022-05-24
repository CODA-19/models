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
	datetime (patient_data.pcr_sample_time) BETWEEN datetime (mergedepisodes.episode_start_time, '-7 day') -- we consider a covid episode when positive test was done in the 7 days preceding the admission
	AND datetime (mergedepisodes.episode_end_time)
	AND patient_data.patient_covid_status = 'positive'