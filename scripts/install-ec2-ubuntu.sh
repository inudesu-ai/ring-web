#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${DOMAIN:-inudesu.xyz}"
REPO_URL="${REPO_URL:-https://github.com/inudesu-ai/ring-web.git}"
APP_DIR="/opt/ring-web"
WEB_ROOT="/var/www/${DOMAIN}"

sudo apt update
sudo apt install -y nginx git curl ca-certificates

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
pm2 start server.js --name ring-api || pm2 restart ring-api
pm2 save
pm2 startup systemd -u "$USER" --hp "$HOME" || true

sudo cp "$APP_DIR/deploy/nginx/inudesu.xyz.conf" /etc/nginx/conf.d/inudesu.xyz.conf
sudo cp "$APP_DIR/deploy/nginx/api.inudesu.xyz.conf" /etc/nginx/conf.d/api.inudesu.xyz.conf
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl reload nginx

echo "Done. Open http://${DOMAIN} and http://api.${DOMAIN}/health"
