# Authenticated full-console web access

This deployment publishes the complete Lumen operator console through one
Cloudflare Tunnel and protects every route with Cloudflare Access. It never
uploads Lumen recordings, runtime databases, Spotify tokens, learned
preferences, or the Cloudflare tunnel token. Do not forward TCP port 4042 on
the home router.

The remote hostname has full operational authority: Live/Monitor, rehearsal,
DMX and fixture settings, Spotify control, timeline review, training, Lumen
Link, and service shutdown. Permit only the owner's exact identity and require
MFA.

## 1. Prepare the hostname and Access gate

1. Register the chosen domain and add it as an active Cloudflare zone.
2. In **Zero Trust → Access controls → Applications**, create a
   **Self-hosted and private** application with a public hostname such as
   `console.lumen-engine.com`.
3. Add one **Allow** policy containing the owner's exact email address. Never
   use `Everyone`, an unrestricted email domain, or One-time PIN alone.
4. Require the chosen identity provider and MFA. Use a finite session duration
   such as eight hours.
5. Create this Access application before publishing the tunnel route.

## 2. Create the tunnel in Cloudflare

1. Go to **Networking → Tunnels** and create a remotely managed tunnel named
   `lumen-console`.
2. Add a **Published application** route for the same hostname.
3. Set the service URL to `http://127.0.0.1:4042`.
4. Enable **Protect with Access** so `cloudflared` validates the Access token at
   the origin boundary.
5. On the tunnel Overview page, choose **Add a replica** and copy the displayed
   command into a temporary local text editor. Keep only the long value after
   `--token`; never paste that token into chat, a Git file, or shell history.

## 3. Stage and activate the Lumen PC

The installer pins and verifies the official Linux amd64 `cloudflared` binary.
Staging writes disabled user-service definitions and does not alter the running
console.

```bash
cd /home/the-system/Desktop/lumenengine
./scripts/lumen-web-access install-client --apply
./scripts/lumen-web-access stage console.lumen-engine.com
./scripts/lumen-web-access activate --apply
```

`activate` privately prompts for only the tunnel token. It then moves Lumen
from the LAN-wide listener to `127.0.0.1:4042`, enables restartable user
services for Lumen and `cloudflared`, and leaves the normal desktop shortcut
working.

The current appliance uses a logged-in GNOME user service and does not enable
systemd lingering. Therefore the Lumen account must remain logged in after a
reboot, as it does for normal garage operation.

## 4. Prove the boundary

```bash
./scripts/lumen-web-access status
./scripts/lumen-web-access verify
```

Verification fails unless Lumen listens only on loopback, both services are
running, the local API responds, and an unauthenticated HTTPS request is sent
to the Cloudflare Access login flow. Then open the hostname in a private browser
window, confirm that login is required, authenticate as the owner, and exercise
the console in Monitor mode before using physical output.

If the gate is ever questionable, stop public access immediately:

```bash
./scripts/lumen-web-access disable --apply
```

This preserves the local Lumen service and stored tunnel token while disabling
the public connector.
