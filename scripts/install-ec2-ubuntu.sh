#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${DOMAIN:-inudesu.xyz}"
REPO_URL="${REPO_URL:-https://github.com/inudesu-ai/ring-web.git}"
APP_DIR="/opt/ring-web"
WEB_ROOT="/var/www/${DOMAIN}"

sudo apt update
sudo apt install -y nginx git curl ca-certificates openssl

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt install -y nodejs
fi
sudo npm install -g pm2

if [ -d "$APP_DIR/.git" ]; then
  sudo git -C "$APP_DIR" pull
else
  sudo git clone "$REPO_URL" "$APP_DIR"
fi

sudo mkdir -p "$WEB_ROOT"
sudo rsync -a --delete "$APP_DIR/public/" "$WEB_ROOT/public/"

cd "$APP_DIR/api"
sudo npm install --omit=dev
if [ ! -f .env ]; then
  PRODUCER_TOKEN="$(openssl rand -hex 32)"
  sudo install -m 600 /dev/null .env
  printf 'PORT=3000\nHOST=127.0.0.1\nRING_BRIDGE_TOKEN=%s\n' "$PRODUCER_TOKEN" \
    | sudo tee .env >/dev/null
  sudo chown "$USER" .env
  echo "Created the ring bridge producer token in $APP_DIR/api/.env"
fi
if pm2 describe ring-api >/dev/null 2>&1; then
  pm2 restart ring-api --update-env
else
  pm2 start server.js --name ring-api
fi
pm2 save
pm2 startup systemd -u "$USER" --hp "$HOME" || true

sudo cp "$APP_DIR/deploy/nginx/inudesu.xyz.conf" /etc/nginx/conf.d/inudesu.xyz.conf
sudo cp "$APP_DIR/deploy/nginx/api.inudesu.xyz.conf" /etc/nginx/conf.d/api.inudesu.xyz.conf
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl reload nginx

echo "Done. Open http://${DOMAIN} and http://api.${DOMAIN}/health"
echo "Read the producer token with: grep RING_BRIDGE_TOKEN $APP_DIR/api/.env"
