# Six-axis IMU simulation results

## One-million-row dataset

Independent decompression and line counting confirmed exactly:

```text
1,000,000 sample rows
1,000 sessions
100 simulated subjects
10 balanced classes, 100,000 rows per class
700,000 train / 150,000 validation / 150,000 test rows
```

The compressed dataset is 23 MiB locally in `ml/data/simulated-1m/` and is not
committed. Its seed, per-shard SHA-256 values, counts, and generation parameters
are recorded in
[`million-1m/dataset_manifest.json`](million-1m/dataset_manifest.json).

The 1.6-second windows use a 0.8-second stride and are resampled to 32 steps:

| Partition | Subjects | Windows |
| --- | ---: | ---: |
| Train | 70 | 34,300 |
| Validation | 15 | 7,350 |
| Shifted test | 15 | 7,350 |

Subject overlap across partitions is exactly zero.

## Gesture classification

### Small MLP

Three sizes were trained and selected using validation data only. The selected
model is a 48-hidden-unit MLP with dropout `0.24`, L2 `8e-4`, sensor noise
injection `0.01`, and early stopping at 39 epochs.

| Split | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Train | 0.8138 | 0.8010 |
| Validation | 0.8046 | 0.7966 |
| Shifted test | 0.7903 | 0.7810 |

- Train-validation Macro-F1 gap: **0.0044**
- Validation-test domain-shift gap: **0.0156**
- Diagnosis: **balanced** — neither underfit nor overfit

### HMM + Baum-Welch EM

The first five-state left-to-right HMM underfit because continuous windows begin
at arbitrary gesture phases. It was replaced with a seven-state ergodic HMM,
raw+difference+magnitude emissions, Dirichlet-smoothed transitions, and 700
balanced EM windows per class.

| Split | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Train | 0.8130 | 0.8125 |
| Validation | 0.7750 | 0.7740 |
| Shifted test | 0.7308 | 0.7200 |

- Train-validation Macro-F1 gap: **0.0385**
- Validation-test domain-shift gap: **0.0540**
- Diagnosis: **balanced** — neither underfit nor overfit

The main remaining confusion is between intentionally symmetric pairs:
`up/down` and `rotate_back/rotate_front`. The complete matrices are in
[`million-1m/test_confusion_matrices.png`](million-1m/test_confusion_matrices.png).

## Six-axis drift benchmark

The benchmark lasts 180 seconds at 100 Hz and includes periodic motion/rest,
known gyro/accelerometer bias, random walk, and noise.

| Orientation algorithm | Tilt RMSE | Yaw RMSE |
| --- | ---: | ---: |
| Raw gyro integration | 52.752° | 77.670° |
| Startup gyro calibration | 3.429° | 7.927° |
| Mahony 6D | 2.729° | 100.547° |
| Madgwick 6D | 2.600° | 67.822° |
| VQF 6D | **0.930°** | **0.594°** |
| Fusion 6D + runtime bias | 1.389° | 6.938° |

The low VQF yaw error occurs because repeated rest periods allow bias estimation
and the simulated yaw returns to its starting direction. It does **not** make
yaw observable in a general six-axis system.

| Position method | Final error |
| --- | ---: |
| Raw double integration | 6,149.58 m |
| Fused orientation, no ZUPT | 2,399.53 m |
| Fused orientation + ZUPT | **10.75 m** |

ZUPT removes most numerical explosion, but 10.75 m after three minutes is still
not absolute positioning. Camera/VIO, UWB, GNSS, wheel/leg contact constraints,
or another external observation is required for dependable long-duration
position.

Detailed data:

- [`drift/drift_metrics.json`](drift/drift_metrics.json)
- [`drift/drift_benchmark.png`](drift/drift_benchmark.png)
- [`million-1m/experiment_report.json`](million-1m/experiment_report.json)
- [`million-1m/mlp_learning_curves.png`](million-1m/mlp_learning_curves.png)
- [`million-1m/models/gesture-mlp-sim-1m.npz`](million-1m/models/gesture-mlp-sim-1m.npz)
- [`million-1m/models/gesture-hmm-sim-1m.npz`](million-1m/models/gesture-hmm-sim-1m.npz)

## Morning smoke/live test

The committed models are simulation-trained baselines. Verify loading and all
local tests first:

```bash
source .venv/bin/activate
python -m unittest discover -s sdk/tests -v
python -m unittest discover -s ml/tests -v
```

Then connect the ring over BLE and try the selected MLP:

```bash
python ml/realtime_infer.py \
  --address YOUR_RING_ADDRESS \
  --model ml/results/million-1m/models/gesture-mlp-sim-1m.npz \
  --threshold 0.65
```

Real-ring accuracy must be measured with subject-held-out captured sessions;
the simulated model is a pretraining and integration baseline, not a substitute
for the device recordings.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/simulation/requirements.txt

python ml/simulation/generate_imu_1m.py \
  --output ml/data/simulated-1m \
  --records 1000000 \
  --subjects 100 \
  --seed 20260723

python ml/simulation/run_million_experiment.py \
  --data "ml/data/simulated-1m/*.jsonl.gz" \
  --manifest ml/data/simulated-1m/manifest.json \
  --output ml/results/million-1m

python ml/simulation/drift_benchmark.py \
  --output ml/results/drift \
  --duration 180 \
  --sample-rate 100 \
  --seed 20260723
```
