# Plan
1. Collect and transform analytics data into bandwidth-timing data sets
2. Create models to predict future bandwidth from past values
  a. Linear
  b. RNN
3. Consider experiments to train a model based on session-scoring

# Bandwidth Predictor
## Dataset
Bandwidth measurement values from `video_engagement` and `qos` analytics data.

- Training set: all bandwidth values, minus the last sample
- Validation set: the same bandwidth values, including the last sample

```sql
SELECT
  session,
  ARRAY_TO_STRING(ARRAY_AGG(CONCAT(client_time, ": ", qos_data)), ", ") AS bitrate_samples
FROM
  `rising-ocean-426.scratch_pad.2017_04_30_videojs_sample`
WHERE
  qos_data IS NOT NULL
  AND event = 'qos.bitrates.bitrate'
GROUP BY
  session
```

## Parameters
_Input_: vectors of timestamp-bandwidth pairs
_Output_: a bandwidth value
