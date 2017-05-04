# Plan
1. Collect and transform analytics data into bandwidth-timing data sets
2. Create models to predict future bandwidth from past values
  a. Linear
  b. RNN
3. Incorporate bandwidth sample timing information into input
4. Consider experiments to train a model based on session-scoring

# Bandwidth Predictor
## Dataset
Bandwidth measurement values from `video_engagement`

- Training set: all bandwidth values, minus the last sample
- Validation set: the same bandwidth values, including the last sample

```sql
SELECT
  ARRAY_TO_STRING(ARRAY_AGG(measured_bps), " ") AS bitrate_samples
FROM
  `brightcove_player_analytics_table`
WHERE
  measured_bps IS NOT NULL
GROUP BY
  session
```

## Parameters
_Input_: vectors of bandwidth samples
_Output_: a bandwidth value
