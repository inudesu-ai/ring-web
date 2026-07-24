# ring-web

Web/API starter for `inudesu.xyz` and the `inudesu-ai` organization.

## Ring Sound SDK

The repository vendors the Python **Ring Sound SDK 0.4.1** in [`sdk/`](sdk/).
It supports BLE discovery and connection, system information, audio transfer,
real-time six-axis IMU data, and built-in HMM gesture events.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r sdk/requirements.txt

# Scan nearby BLE devices.
python sdk/ring_sound.py scan

# Read ring system information.
python sdk/ring_sound.py info --address YOUR_RING_ADDRESS
```

The ring must already be in gesture mode before real-time IMU reporting can be
started. Capture labeled IMU samples for the upcoming classifier:

```bash
python sdk/examples/capture_imu_jsonl.py \
  --address YOUR_RING_ADDRESS \
  --label wave \
  --duration 30 \
  --output sdk/captures/wave-001.jsonl
```

Monitor the firmware's built-in HMM gesture output:

```bash
python sdk/examples/hmm_gesture_monitor.py \
  --address YOUR_RING_ADDRESS
```

See [`sdk/INTEGRATION.md`](sdk/INTEGRATION.md) for the import manifest and
migration notes, and [`sdk/ring_sound_use.md`](sdk/ring_sound_use.md) for the
complete API guide.

## Gesture models

The [`ml/`](ml/) pipeline includes:

- a configurable left-to-right/ergodic diagonal Gaussian HMM trained with
  Baum-Welch EM;
- a small NumPy MLP trained with Adam, class balancing, dropout, and early
  stopping;
- a 10k-parameter depthwise temporal CNN trained with placement/noise
  augmentation and exported to portable NumPy inference;
- subject/session-aware train/validation/test splitting;
- portable `.npz` model files;
- live BLE inference and authenticated publishing to the Web dashboard.

Train both baselines after collecting at least two sessions per class
(8-10 sessions per class is the practical target):

```bash
python -m pip install -r ml/requirements.txt

python ml/train_hmm.py \
  --data "ml/data/*.jsonl" \
  --output ml/models/gesture-hmm.npz

python ml/train_mlp.py \
  --data "ml/data/*.jsonl" \
  --output ml/models/gesture-mlp.npz

python -m pip install -r ml/requirements-train.txt
python ml/train_temporal_cnn.py \
  --data "ml/data/*.jsonl" \
  --output ml/models/gesture-temporal-cnn.npz \
  --report ml/results/temporal-cnn-report.json
```

Run real-time inference on the laptop or edge computer connected to the ring:

```bash
export RING_BRIDGE_TOKEN="same-token-as-the-API-server"

python ml/realtime_infer.py \
  --name ring \
  --model ml/models/gesture-temporal-cnn.npz \
  --robot-commands \
  --publish-url https://api.inudesu.xyz/v1/gesture
```

`--name ring` is recommended on macOS because CoreBluetooth UUIDs can rotate.
The bridge now also publishes 10 Hz quaternion, Euler-angle, acceleration,
gyroscope, and short-term relative-motion telemetry to `/v1/telemetry`.

### ZUPT 3-D trajectory and directional fusion

The live bridge performs displacement calculation at the full IMU sample rate
instead of integrating in the browser:

1. estimate a body-to-world quaternion and runtime gyro bias;
2. rotate acceleration into the startup world frame and remove gravity;
3. learn stationary acceleration bias/noise and apply an adaptive deadband;
4. use trapezoidal integration for velocity and position;
5. reject rotation-only lever-arm acceleration;
6. force velocity to zero at confirmed rest (ZUPT).

`telemetry.motion` contains position, velocity, path length, segment state,
adaptive threshold and ZUPT status. A trajectory-aided classifier uses the
dominant displacement axis to correct ambiguous `left/right/up/down` MLP
outputs. Up/down is gravity-referenced; horizontal heading remains relative
because the ring is a six-axis device without a magnetometer.

The physical recognizer also exposes two first-class states:

- `zupt-stationary` forces `idle` only after a calibrated, low-fluctuation
  rest window;
- `zupt-circle` verifies a loop using best-fit-plane PCA, angular sweep,
  closure, roundness, radial consistency, and planar energy before overriding
  the MLP. Circle geometry is included in `gesture.circle_metrics`.
- `zupt-depth` adds physical-only `forward/backward` translation when the
  world-Y displacement clearly dominates lateral and vertical movement.

This trajectory is intended for short `rest → move → rest` gestures. General
long-duration IMU-only position tracking remains unobservable without an
external position or velocity reference.

Full training and collection instructions are in [`ml/README.md`](ml/README.md).
The completed one-million-row simulation and six-axis drift benchmark are
summarized in [`ml/results/RESULTS.md`](ml/results/RESULTS.md).
The current literature review, episodic one-million-row experiment, real-idle
check, and mechanical-dog command architecture are documented in
[`ml/research/MAINSTREAM_GESTURE_RECOGNITION.md`](ml/research/MAINSTREAM_GESTURE_RECOGNITION.md).

## Live gesture API

The local inference bridge publishes predictions to:

```text
POST /v1/gesture
Authorization: Bearer $RING_BRIDGE_TOKEN
```

The server validates and broadcasts accepted predictions to browser viewers:

```text
GET /v1/gesture/latest
POST /v1/telemetry
GET /v1/telemetry/latest
WebSocket /ws
```

The WebSocket drives the frontend's gesture probabilities, 3D ring pose,
roll/pitch/yaw readout, six-axis plots, and recognition history.

Create an API token on the EC2 instance:

```bash
cd /opt/ring-web/api
printf 'RING_BRIDGE_TOKEN=%s\n' "$(openssl rand -hex 32)" > .env
pm2 restart ring-api --update-env
```

Copy the same value into `RING_BRIDGE_TOKEN` on the local ring computer. Never
put this producer token in frontend JavaScript.

## DNS

Cloudflare records should point to the AWS EC2 public IPv4 address:

```text
A    @      3.239.192.34    Proxied
A    www    3.239.192.34    Proxied
A    api    3.239.192.34    Proxied
```

## AWS EC2 security group

Open inbound ports:

```text
TCP 22   SSH
TCP 80   HTTP
TCP 443  HTTPS
```

## Quick deploy on Ubuntu EC2

SSH into the instance, then run:

```bash
curl -fsSL https://raw.githubusercontent.com/inudesu-ai/ring-web/main/scripts/install-ec2-ubuntu.sh | bash
```

This installs Nginx, Node.js, PM2, clones this repo, serves the static site, and starts the API.
On the first run it also generates the gesture producer token in
`/opt/ring-web/api/.env`.

## Manual deploy

```bash
sudo apt update
sudo apt install -y nginx git curl ca-certificates
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g pm2

sudo git clone https://github.com/inudesu-ai/ring-web.git /opt/ring-web
sudo mkdir -p /var/www/inudesu.xyz
sudo rsync -a /opt/ring-web/public/ /var/www/inudesu.xyz/public/

cd /opt/ring-web/api
sudo npm install --omit=dev
pm2 start server.js --name ring-api
pm2 save

sudo cp /opt/ring-web/deploy/nginx/*.conf /etc/nginx/conf.d/
sudo nginx -t
sudo systemctl reload nginx
```

## Test

```bash
curl http://inudesu.xyz/health
curl http://api.inudesu.xyz/health
curl http://api.inudesu.xyz/hello
curl http://api.inudesu.xyz/v1/gesture/latest
```

## HTTPS

If Cloudflare proxy is enabled, set Cloudflare SSL/TLS mode to **Full**.

For direct origin HTTPS on EC2:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d inudesu.xyz -d www.inudesu.xyz -d api.inudesu.xyz
```
