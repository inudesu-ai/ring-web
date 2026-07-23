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
```

## HTTPS

If Cloudflare proxy is enabled, set Cloudflare SSL/TLS mode to **Full**.

For direct origin HTTPS on EC2:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d inudesu.xyz -d www.inudesu.xyz -d api.inudesu.xyz
```
