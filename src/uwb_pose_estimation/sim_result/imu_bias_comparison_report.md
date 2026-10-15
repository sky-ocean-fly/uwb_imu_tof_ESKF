# IMU bias-state ablation (`simulation_both.launch`)

## Setup

- Simulation duration: 60 s; random seed: `20260714`; 3001 synchronized pose samples per run.
- Both runs use the same generator, initial-pose perturbation, IMU/UWB/ToF noise, fake-height setting, gates, and logger.
- Injected constant IMU B gyro bias: `[0.020, -0.015, 0.010]` rad/s.
- Injected constant IMU B accelerometer bias: `[0.20, -0.10, 0.15]` m/s^2.
- IMU A injected bias is zero.
- With-bias run: `uwb_imu_tof_eskf_fake_height_add_imu_bias_node` (21D error state, online bias estimation enabled).
- Without-bias run: `uwb_imu_tof_eskf_fake_height_node` (9D error state).
- Both targets were force-rebuilt before the reported runs. This was necessary because the original incremental build contained a stale no-bias object file.

## Pose accuracy

| Window | Metric | With bias states | Without bias states | Reduction |
|---|---:|---:|---:|---:|
| 0--60 s | Position RMSE | 0.034631 m | 0.321430 m | 89.23% |
| 0--60 s | Attitude RMSE | 0.971128 deg | 8.128278 deg | 88.05% |
| 5--60 s | Position RMSE | 0.034347 m | 0.333155 m | 89.69% |
| 5--60 s | Attitude RMSE | 0.711952 deg | 8.434200 deg | 91.56% |
| 50--60 s | Position RMSE | 0.024445 m | 0.316559 m | 92.28% |
| 50--60 s | Attitude RMSE | 0.394765 deg | 12.264576 deg | 96.78% |

At 60 s, the with-bias estimator has 0.019694 m position error and 0.665839 deg attitude error. The no-bias estimator has 0.314228 m and 13.301998 deg, respectively.

For the steady 5--60 s interval, the with-bias position-axis RMSE is `[0.023066, 0.021204, 0.014075]` m, compared with `[0.189272, 0.257390, 0.094442]` m without bias states. Its roll/pitch/yaw RMSE is `[0.375869, 0.277620, 0.538387]` deg, compared with `[2.393220, 3.153172, 7.414861]` deg.

The identical large initial maxima (0.353553 m and 10.246951 deg for the with-bias run) come from the common injected initial-pose error. The post-transient and last-10-second metrics are therefore more informative about bias compensation.

## Bias estimates

Mean estimates over 50--60 s:

| State | Injected truth | Estimated mean | Error norm |
|---|---:|---:|---:|
| Gyro A (rad/s) | `[0, 0, 0]` | `[-0.000684, 0.002103, -0.000562]` | 0.002282 |
| Gyro B (rad/s) | `[0.020, -0.015, 0.010]` | `[0.019625, -0.012795, 0.009978]` | 0.002237 |
| Accel A (m/s^2) | `[0, 0, 0]` | `[-0.041710, -0.014924, -0.074212]` | 0.086429 |
| Accel B (m/s^2) | `[0.20, -0.10, 0.15]` | `[0.155734, -0.101772, 0.071429]` | 0.090200 |

The gyro states converge close to the injected values. The accelerometer states compensate the motion enough to improve the pose strongly, but the individual A/B accelerometer biases do not converge to their separate ground truths, especially the z components. This points to coupling/limited observability between the two accelerometer-bias vectors in this trajectory and measurement geometry.

## Conclusion

Adding IMU bias states is effective for pose estimation in this fixed-bias simulation: it prevents the large position and yaw drift seen in the 9D filter and reduces steady-state pose RMSE by about 90% or more. However, the experiment does not show that all twelve individual bias components are accurately identifiable. Further validation should use multiple random seeds and trajectories, sweep bias magnitudes, and include a zero-injected-bias control to check whether the larger state hurts performance when no bias is present.

Note: although `sim_data_generator.yaml` contains `uwb_bias: 0.3`, the current generator source sets the per-range UWB bias contribution to zero before publishing. This comparison therefore tests IMU bias compensation under UWB random noise, not simultaneous fixed UWB bias compensation.
