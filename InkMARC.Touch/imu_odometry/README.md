# IMU Odometry Baseline

This folder contains an offline teacher/student inertial odometry baseline built around short sensor windows.

Files:
- `train.py`: training entrypoint, CLI, checkpoints, metrics, plots
- `cross_validate.py`: session-level cross-validation runner over whole CSV sessions
- `dataset.py`: session-aware CSV loader and sliding-window dataset
- `model.py`: `TemporalCnnRegressor` baseline and optional `CnnGruRegressor`
- `losses.py`: regression loss selection
- `evaluate.py`: checkpoint evaluation and prediction export
- `config.py`: defaults, CLI config, feature groups, and column aliases
- `utils.py`: seeds, JSON/CSV helpers, checkpoints, normalization, plotting

Default assumptions:
- Each CSV file is one session.
- Rows are sorted by a timestamp column.
- Inputs are time windows of sensor features.
- Targets are delta motion over the window using the first and last valid teacher rows.

Expected teacher columns:
- `plane_x_m`
- `plane_y_m`
- `plane_z_m`
- `tag_pose_valid` by default

Expected base input columns:
- `accel_x`, `accel_y`, `accel_z`
- `gyro_x`, `gyro_y`, `gyro_z`
- `rot_x`, `rot_y`, `rot_z`, `rot_w`

Optional aliases already supported:
- `grav_x/y/z` for gravity
- `lin_x/y/z` for linear acceleration
- `mag_x/y/z` for magnetometer

If your log schema differs, update `COLUMN_ALIASES` in `config.py`.

Cross-validation:
- Use session-level cross-validation only. Do not split overlapping windows randomly across train/val/test.
- `cross_validate.py` rotates whole CSV files through train/val/test folds and aggregates the saved metrics.

Example:
```bash
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python cross_validate.py \
  --csv-files /home/t/tburnah/inkmarc/dtm_dtm_1774674722341.csv /home/t/tburnah/inkmarc/dtm_dtm_1774675309496.csv /home/t/tburnah/inkmarc/dtm_dtm_1774840640921.csv /home/t/tburnah/inkmarc/dtm_dtm_1774840677658.csv /home/t/tburnah/inkmarc/dtm_dtm_1774840801448.csv \
  --mode rotation \
  --output-dir /home/t/tburnah/TouchModel/runs/imu_odometry_cv \
  -- \
  --feature-groups base gravity nav_frame \
  --window-size 60 \
  --stride 10 \
  --loss-weight-mode target_magnitude \
  --loss-weight-reference-m 0.05 \
  --min-target-magnitude-m 0.10 \
  --monitor-metric vector_mae
```

Outputs:
- `crossval_folds.csv`: one row per fold with flattened train/val/test metrics
- `crossval_summary.json`: aggregated mean/std/min/max across folds
