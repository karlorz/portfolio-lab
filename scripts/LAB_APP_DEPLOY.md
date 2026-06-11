# Lab App Deploy

Canonical URL:

```text
https://lab.karldigi.dev
```

DNS for `lab.karldigi.dev` is already completed outside this repo. The project deploy path only assigns the app to the hostname through Caddy and starts the tasker API through systemd.

## Model

The lab app follows the same operational shape as the Proxmox LXC internal update pattern:

- The app runs directly inside the LXC/Linux guest.
- `scripts/deploy-lab-app.sh` updates the checkout, builds the frontend, publishes `dist/`, installs/restarts the tasker systemd service, and writes a managed Caddy site block.
- `/usr/local/bin/portfolio-lab-update` is installed as the durable in-container update command.
- Docker remains available for separate container experiments, but it is not the default lab deployment path.

## Deploy

Run from inside the Portfolio Lab LXC/container host:

```bash
scripts/deploy-lab-app.sh
```

Preview the exact host changes without writing system files:

```bash
scripts/deploy-lab-app.sh --dry-run
```

After the first successful deploy, future updates can be run from anywhere in the guest:

```bash
portfolio-lab-update
```

or through Make:

```bash
make deploy-lab-app
```

## Caddy

The script writes a managed block to `/etc/caddy/Caddyfile` by default. It validates before reload and restores the backup if validation fails.

The generated site block has this shape:

```caddyfile
lab.karldigi.dev {
	encode gzip

	handle /api/* {
		reverse_proxy 127.0.0.1:8000
	}

	handle /data/* {
		root * /var/www/portfolio-lab
		file_server
	}

	handle {
		root * /var/www/portfolio-lab
		try_files {path} /index.html
		file_server
	}
}
```

Print the block without writing it:

```bash
scripts/deploy-lab-app.sh --print-caddy
```

## Config

Defaults live in `config/lab-app.env`:

```text
PORTFOLIO_LAB_SITE_ADDRESS=lab.karldigi.dev
PORTFOLIO_LAB_APP_DIR=/root/projects/portfolio-lab
PORTFOLIO_LAB_WEB_ROOT=/var/www/portfolio-lab
PORTFOLIO_LAB_PUBLIC_ROOT=/var/www/portfolio-lab
PUBLIC_DATA_DIR=/var/www/portfolio-lab/data
TASKER_HOST=127.0.0.1
TASKER_PORT=8000
TASKER_SERVICE_NAME=portfolio-lab-tasker
CADDY_CONFIG=/etc/caddy/Caddyfile
CADDY_SERVICE_NAME=caddy
UPDATE_COMMAND_PATH=/usr/local/bin/portfolio-lab-update
```

## Smoke Checks

```bash
systemctl status portfolio-lab-tasker --no-pager -l
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
curl -fsS http://127.0.0.1:8000/api/tasker/status
curl -I https://lab.karldigi.dev/
curl -fsS https://lab.karldigi.dev/api/tasker/status
curl -fsS https://lab.karldigi.dev/data/signals.json | head
```
