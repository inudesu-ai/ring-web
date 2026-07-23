# Paper-informed design notes

This simulation is intentionally more difficult than the original sine-wave
smoke test. It follows the parts of published methods that are applicable
without a full motion-capture skeleton.

## Physics-informed IMU generation

WIMUSim models Body, Dynamics, Placement, and Hardware parameters. The local
ring simulator implements:

- **Dynamics:** smooth orientation trajectories and world-frame linear
  acceleration, converted to body angular velocity and specific force.
- **Placement:** subject-specific sensor-to-hand rotation plus session
  perturbation.
- **Hardware:** per-axis gain, static bias, bias random walk, bandwidth,
  measurement noise, clock error, jitter, range clipping, and int16
  quantisation.

It does not claim the full skeletal Body model or motion-capture parameter
identification of WIMUSim. Real ring captures remain necessary for sim-to-real
calibration.

References:

- [WIMUSim: simulating realistic variabilities in wearable IMUs for HAR](https://doi.org/10.3389/fcomp.2025.1514933)
- [Physically Plausible Data Augmentations for Wearable IMU-based HAR](https://arxiv.org/abs/2508.13284)

## Augmentation and generalisation

Um et al. evaluated jitter, scaling, rotation, permutation, magnitude warping,
and time warping for wearable sensors. Their central constraint is preserved
here: an augmentation must remain label-preserving. Arbitrary post-hoc warps
can turn one physical gesture into another, so this simulator changes movement,
placement, and hardware parameters before sensor projection instead.

- [Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring](https://arxiv.org/abs/1706.00527)
- [Sampling-rate-robust smartphone HAR](https://arxiv.org/abs/2101.00812)

The split is by whole simulated subject, not random windows:

```text
70 train subjects / 15 validation subjects / 15 shifted test subjects
```

This prevents adjacent overlapping windows from the same person appearing in
multiple partitions. The test subjects have 1.6x placement/noise/bias severity.

## Temporal classifiers

The MLP receives per-step raw six-axis values, first differences, and
accelerometer/gyroscope magnitudes. It is kept deliberately small for an edge
computer and uses class balancing, noise injection, dropout, L2 regularisation,
Adam, validation-only early stopping, and three capacity candidates.

The HMM uses diagonal Gaussian emissions trained with Baum-Welch EM. A
left-to-right topology is suitable for pre-segmented gestures that all start at
the same phase. Continuous sliding windows have arbitrary phase, so the final
simulation uses a seven-state **ergodic** topology. This change removed the
first run's underfitting.

- [Original Baum-Welch maximisation paper](https://doi.org/10.1214/aoms/1177697196)
- [DeepConvLSTM for multimodal wearable activity recognition](https://doi.org/10.3390/s16010115)

## Six-axis drift control

The benchmark compares:

- raw gyro quaternion integration;
- stationary startup bias calibration;
- Mahony explicit nonlinear complementary filtering;
- Madgwick six-axis gradient descent;
- the official `vqf` implementation in magnetometer-free 6D mode;
- the official `imufusion` implementation and runtime gyro bias estimator;
- attitude-aided acceleration integration with and without ZUPT.

References and code:

- [Mahony et al., Nonlinear Complementary Filters on SO(3)](https://doi.org/10.1109/TAC.2008.923738)
- [Madgwick's IMU orientation filter report](https://courses.cs.washington.edu/courses/cse466/14au/labs/l4/madgwick_internal_report.pdf)
- [VQF paper](https://arxiv.org/abs/2203.17024) and [official implementation](https://github.com/dlaidig/vqf)
- [xioTechnologies Fusion](https://github.com/xioTechnologies/Fusion)
- [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)

Gravity observes roll and pitch but not absolute yaw. No six-axis-only result in
this repository should be interpreted as absolute heading. Position is even
less observable: ZUPT constrains drift when rest intervals exist, but external
position/velocity observations are required for general long-duration
tracking.
