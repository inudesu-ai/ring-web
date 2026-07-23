# ring-web

Web/API starter for `inudesu.xyz` and the `inudesu-ai` organization.

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
