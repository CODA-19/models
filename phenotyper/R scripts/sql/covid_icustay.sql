WITH premergedicuepisode_flag AS (
WITH premergedicuepisodes AS (
WITH globalepisodes AS (
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
	mergedepisodes.episode_end_time,
	stayid
FROM
	mergedepisodes
	INNER JOIN patient_data ON mergedepisodes.patient_site_uid = patient_data.patient_site_uid
WHERE
	datetime (patient_data.pcr_sample_time) BETWEEN datetime (mergedepisodes.episode_start_time,
'-8 day') -- we consider a covid episode when positive test was done in the 8 days preceding the admission
	AND datetime (mergedepisodes.episode_end_time)
AND patient_data.patient_covid_status = 'positive'
) -- Grabbing all hospital episodes and secondarly identifying which one of those contained ICU episodes
SELECT
	globalepisodes.patient_site_uid,
	globalepisodes.episode_start_time hospit_start_time,
	globalepisodes.episode_end_time hospit_end_time,
	episode_data.episode_start_time episode_start_time,
	episode_data.episode_end_time episode_end_time,
	episode_data.episode_unit_type
FROM
	globalepisodes
	INNER JOIN episode_data ON globalepisodes.patient_site_uid = episode_data.patient_site_uid
WHERE
	datetime (episode_data.episode_start_time) BETWEEN datetime (globalepisodes.episode_start_time)
AND datetime (globalepisodes.episode_end_time)
AND episode_data.episode_unit_type = 'intensive_care_unit'
)
SELECT
	patient_site_uid, hospit_start_time, hospit_end_time, episode_start_time, episode_end_time,episode_unit_type, SUM(flag) OVER (PARTITION BY patient_site_uid ORDER BY episode_start_time) stayid
	FROM (
		SELECT
			*,
			strftime ('%s',
				premergedicuepisodes.episode_start_time) - strftime ('%s',
				LAG(premergedicuepisodes.episode_end_time, 1, datetime (premergedicuepisodes.episode_start_time, '-1 hour')) OVER (PARTITION BY premergedicuepisodes.patient_site_uid ORDER BY premergedicuepisodes.episode_start_time)) > 10 * 3600 flag -- this represents the delay in seconds at which an episode is considered having occured during another hospital stay
			-- Here decided to keep 10 hours - based on some observations made in the dataset
		FROM
			premergedicuepisodes))
SELECT
	patient_site_uid,
	hospit_start_time,
	hospit_end_time,
	min(episode_start_time) episode_start_time,
	max(episode_end_time) episode_end_time,
	episode_unit_type
FROM
	premergedicuepisode_flag
GROUP BY
	patient_site_uid, stayid
