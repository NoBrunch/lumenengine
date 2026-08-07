# Lumen Link: Threadripper/WSL deployment

Lumen Link v1 moves **offline EDMFormer song analysis** from the dedicated
Lumen PC to the 128 GiB Threadripper. Its contract is ready to add other
offline work after each result importer is implemented and validated. Lumen
remains the authority for line-in timing, feedback, song memory, choreography
and DMX. If the cable is unplugged or Windows restarts, Live continues and
remote jobs remain queued.

The operator dashboard stays in Lumen's KDE-inspired interface. It shows the
node connection, CPU, memory and disk telemetry, current song/job, stage,
progress, completed/queued work and the last verified result. The WSL side is
deliberately headless.

```text
Lumen PC                                      Windows + WSL Threadripper
192.168.50.2/24                               192.168.50.1/24

canonical DB ── immutable HMAC job bundle ──> content-addressed job store
Live + DMX    <── checksummed result bundle ── EDMFormer (first capability)
dashboard    <──────── status/progress ─────── dependency-free HTTP :8765
```

Neither machine shares a writable SQLite database. Git contains source,
schemas and examples only. It must never contain recordings, job bundles,
tokens, the HMAC secret, runtime databases, models or learned preferences.

## Before beginning

You need:

- Windows 11 22H2 or newer on the Threadripper.
- Ubuntu installed under WSL2.
- One unused Ethernet port on each PC and a Cat5e/Cat6 cable.
- The name of the dedicated Windows Ethernet adapter, such as `Ethernet 2`.
- Internet access on the Threadripper's normal network while provisioning.

The direct cable carries no gateway and no DNS. Windows keeps its existing
internet connection. The Lumen PC keeps Wi-Fi for internet access.

Do not place the repository under `/mnt/c`. Models and song objects perform
far better in WSL's native Linux filesystem, at `~/lumenengine`.

Keep the three command environments distinct:

| Where you are | Prompt usually looks like | What belongs there |
|---|---|---|
| Threadripper Ubuntu WSL | `name@threadripper:~/lumenengine$` | Clone, dependencies, worker service and logs |
| Windows administrator PowerShell | `PS C:\Windows\System32>` | Dedicated NIC, firewall, WSL mode and persistence |
| Lumen Ubuntu PC | `the-system@the-system:~/Desktop/lumenengine$` | `enp1s0`, private pairing, dashboard and canonical data |

Do not run a Windows networking command in Ubuntu, or a Lumen-PC command in
Threadripper WSL. Each procedure below names its required machine again.

## 1. Install and start Ubuntu WSL

Open **PowerShell as Administrator** on Windows and run:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if prompted. Open **Ubuntu** from the Start menu. The first
launch asks for a new Linux username and password. This is separate from the
Windows account. The password will not appear while typing; that is normal.

Check WSL:

```powershell
wsl --list --verbose
wsl --version
```

The Ubuntu row must say version `2`.

### Give WSL useful Threadripper resources

In Windows, create or edit `%UserProfile%\.wslconfig`. Preserve unrelated
settings if that file already exists. A sensible starting point for this
128 GiB machine is:

```ini
[wsl2]
memory=112GB
processors=48
swap=32GB
networkingMode=mirrored
```

This leaves Windows approximately 16 GiB and 16 logical processors. It is not
a neural-model limit; it only prevents WSL from crowding out Windows itself.
Apply it from PowerShell:

```powershell
wsl --shutdown
```

Open Ubuntu again and verify:

```bash
nproc
free -h
```

If `networkingMode=mirrored` is unsupported or unreliable, remove that line,
run `wsl --shutdown`, and use the NAT/port-proxy path in section 4. Both modes
are supported by the deployment script.

### Enable systemd

Current Ubuntu WSL releases normally enable systemd. Check inside Ubuntu:

```bash
systemctl --user status
```

If it says systemd is not running, edit `/etc/wsl.conf`:

```bash
sudo nano /etc/wsl.conf
```

Enter exactly:

```ini
[boot]
systemd=true
```

Save with `Ctrl+O`, press Enter, then exit with `Ctrl+X`. Back in Windows
PowerShell run `wsl --shutdown`, reopen Ubuntu, and repeat the status check.

## 2. Clone the public source, not private state

Inside Ubuntu:

```bash
cd ~
git clone https://github.com/NoBrunch/lumenengine.git
cd ~/lumenengine
git config pull.ff only
git status --short
```

The final command should print nothing. The clone contains no Lumen recordings
or credentials.

## 3. Provision the compute node

First inspect the read-only status and dry run:

```bash
cd ~/lumenengine
./scripts/lumen-link-wsl status
./scripts/lumen-link-wsl install
```

The second command prints intended operations but does not use `sudo`, access
the network or change files. When it looks correct:

```bash
./scripts/lumen-link-wsl install --apply --cpu-threads 24
```

Enter the Linux password when `sudo` asks. Provisioning installs:

- Git/LFS, FFmpeg, libsndfile, rsync, OpenSSH and build tools.
- Python 3.12.8 through a dedicated, release-pinned pyenv checkout for the
  Lumen core and isolated EDMFormer
  environment. Provisioning never relies on whichever Python Ubuntu happens
  to put on `PATH`.
- Python 3.10.20 through pyenv for the isolated SongFormer environment.
- The exact frozen CPU PyTorch 2.4, MuQ, MusicFM, EDMFormer and SongFormer
  dependencies already pinned in `config/research/`. The first Lumen Link
  release executes EDMFormer remotely. SongFormer and student training are
  provisioned but visibly gated until their immutable result importers exist.
- Verified model assets and source revisions from `research-lock.json`.
- A local 256-bit HMAC secret with mode `600`.
- A user service at `~/.config/systemd/user/lumen-link-worker.service`.

Model downloads are several gigabytes and may take time. Provisioning is
resumable: rerun the same command after an interruption. It does not download
commercial song audio, the 262 GiB SongFormDB feature repository, or Lumen's
private database.

If an interrupted earlier attempt created an environment with the wrong
Python, the installer moves that generated environment to an
`.incompatible-<version>-<timestamp>` backup before rebuilding it. It never
silently mixes Python ABIs. `status` reports all four relevant interpreters;
`verify` requires core/EDMFormer 3.12.8 and SongFormer 3.10.20.
The pyenv tool itself is pinned to release `v2.6.27` under
`~/.local/share/lumen-link/pyenv`, separate from any personal pyenv install.

Keep the worker alive across WSL sessions:

```bash
sudo loginctl enable-linger "$USER"
./scripts/lumen-link-wsl verify
./scripts/lumen-link-wsl start
./scripts/lumen-link-wsl status
```

Normal controls are:

```bash
./scripts/lumen-link-wsl status
./scripts/lumen-link-wsl logs
./scripts/lumen-link-wsl stop
./scripts/lumen-link-wsl start
./scripts/lumen-link-wsl restart
```

`verify` is intentionally thorough and loads the model heads. `status` is the
quick, non-mutating everyday check.

## 4. Configure the dedicated Windows Ethernet port

Connect the cable. Open PowerShell as Administrator and find the unused port:

```powershell
Get-NetAdapter
```

Reference the script directly through WSL. Replace `<linux-user>` and the
adapter name:

```powershell
$LinkScript = "\\wsl.localhost\Ubuntu\home\<linux-user>\lumenengine\scripts\lumen-link-windows.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $LinkScript
powershell -NoProfile -ExecutionPolicy Bypass -File $LinkScript -Apply -InterfaceAlias "Ethernet 2" -Mode Mirrored
```

This adds `192.168.50.1/24` to only that adapter. It assigns no gateway or DNS.
It creates a Windows inbound rule permitting TCP 8765 only on
`192.168.50.1` and only from `192.168.50.2`. It refuses to reconfigure an
adapter that already has another IPv4 address or a default route, which helps
catch selection of the wrong port.

### NAT fallback

If mirrored networking is unavailable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $LinkScript -Apply -InterfaceAlias "Ethernet 2" -Mode Nat
```

NAT mode forwards `192.168.50.1:8765` to WSL's changing private address. The
script installs a Windows scheduled task named **Lumen Link WSL Network
Refresh** at logon so the forwarding target is corrected after restarts. The
worker binds `0.0.0.0` inside WSL because the Windows-owned
`192.168.50.1` address does not exist inside the NAT namespace. The dedicated
interface and restricted Windows firewall remain the exposure boundary.

Mirrored mode may also require the Windows Hyper-V firewall rule created by
the script. Do not add a broad `Any`-remote rule to work around a connection
problem. In either mode, the script registers the **Lumen Link WSL Network
Refresh** logon task. NAT uses it to refresh the changing WSL forwarding
address; mirrored mode uses it only to start Ubuntu so the enabled systemd user
service comes up after Windows login.

## 5. Configure the Lumen PC side of the cable

On the Lumen PC, identify the wired connection:

```bash
nmcli device status
nmcli connection show
```

The known wired device is `enp1s0`. Inspect Lumen's safe setup dry run:

```bash
cd "/home/the-system/Desktop/lumenengine"
./scripts/lumen-link status
./scripts/lumen-link network
```

The dry run must show `192.168.50.2/24`, `ipv4.never-default yes`, an empty
gateway and empty DNS. Apply it only after checking the interface name:

```bash
./scripts/lumen-link network --apply --interface enp1s0
```

Verify that the normal default route did not move to Ethernet:

```bash
ip -4 address show enp1s0
ip route
ping -c 3 192.168.50.1
```

The default route should still use Wi-Fi. Do not add a gateway to the
`192.168.50.0/24` link.

## 6. Pair HMAC authentication

The WSL installer creates:

```text
~/.config/lumen-link/shared-secret
```

It is mode `600`, excluded from Git and never shown by a status command. Copy
its one-line value once through a local clipboard or removable media. On the
Lumen PC, use the hidden prompt so the value does not enter shell history:

```bash
./scripts/lumen-link pair --apply
```

Alternatively, put it in a local/removable-media file and use:

```bash
./scripts/lumen-link pair --apply --secret-source /path/to/local/secret-file
```

Delete the transfer copy afterward. Do not paste the value into chat, an
issue, Git, or a shell command. The local script writes
`state/lumen-link/config.json` and `secret` with mode `600`; that state path is
ignored by Git. It initially leaves offload disabled.

Lumen uses that secret to authenticate the request body, timestamp and nonce.
The Windows interface/firewall accepts only the direct-link Lumen address.
Checksums protect every immutable uploaded object and downloaded result
independently of transport.

## 7. Optional SSH bootstrap and diagnostics

SSH does not carry runtime jobs. It is optional for source updates and remote
diagnostics. Generate a dedicated key on the Lumen PC:

```bash
install -d -m 700 state/lumen-link/keys
ssh-keygen -t ed25519 -f state/lumen-link/keys/threadripper_ed25519 -C lumen-link
```

Copy only the `.pub` file to WSL, then inside WSL run:

```bash
./scripts/lumen-link-wsl install-ssh --apply \
  --lumen-public-key-file /path/to/threadripper_ed25519.pub
```

For WSL NAT, repeat the Windows network command with
`-EnableSshBootstrap`; it opens TCP 9022 only from `192.168.50.2`. Test from
Lumen:

```bash
ssh -p 9022 -i state/lumen-link/keys/threadripper_ed25519 \
  <linux-user>@192.168.50.1
```

Never copy the private key to Windows or Git. Password login is unnecessary.
The SSH firewall/forward can be removed after bootstrap if remote diagnostics
are not wanted.

## 8. Dashboard and first end-to-end check

Open the standalone **Lumen Link** page in Lumen's main navigation. The
expected progression is:

```text
Disconnected → Authenticating → Ready → Uploading → Analyzing
             → Verifying result → Imported
```

Before sending a real song, use **Test connection**. You can perform the same
authenticated health check at the Lumen terminal with
`./scripts/lumen-link test`. A successful test authenticates the worker and
compares its code, teacher, model, preprocessing and ontology contract with
the Lumen PC.

There is intentionally no per-song offload selector in v1. The coordinator
chooses the next eligible automatic EDMFormer job and routes only one job at a
time. Use that behavior for the first canary:

1. Press **Enable link**. Wait for the first job to appear as uploading or
   running.
2. Press **Disable link** while that first job is active. Other queued
   automatic jobs return to local eligibility; the already-active remote job
   is allowed to finish and import its verified result.
3. Inspect that job's result, event history and imported timeline.
4. If the canary is correct, press **Enable link** again for sequential bulk
   processing. **Pause dispatch** temporarily stops new routing; **Disable
   link** returns unstarted queued work to automatic/local eligibility. An
   already-active remote process is never discarded mid-result.

During the canary, confirm:

- the dashboard updates without blocking Spotify or the live audio display;
- unplugging Ethernet leaves the job queued and does not affect Live;
- reconnecting resumes without creating a duplicate job;
- the result checksum and code/model revisions verify before import;
- only `teacher.edmformer` is shown as available; SongFormer and student
  training remain clearly gated;
- a future remote candidate cannot become active until held-out validation
  passes.

Those cable/restart cases are physical acceptance work, not claims that this
deployment package has already exercised the two computers.

## Updating Lumen Link

Stop only the offline worker, then update the public source inside WSL:

```bash
cd ~/lumenengine
./scripts/lumen-link-wsl stop
git pull --ff-only
./scripts/lumen-link-wsl configure --apply
./scripts/lumen-link-wsl verify
./scripts/lumen-link-wsl start
```

Do not run `git clean` against the state tree or copy Lumen's SQLite database
to the compute node. Existing content-addressed objects are reusable after a
code update only when their recorded schema, preprocessing, ontology and model
revisions still match.

## Troubleshooting

- **WSL command says systemd is unavailable:** enable it in `/etc/wsl.conf`,
  run `wsl --shutdown` in Windows, and reopen Ubuntu.
- **The worker works until Windows restarts:** use the NAT apply command again
  and confirm the **Lumen Link WSL Network Refresh** scheduled task exists, or
  repair mirrored networking.
- **Connection refused:** check `./scripts/lumen-link-wsl status`, then
  `logs`; confirm TCP 8765 with `Test-NetConnection 192.168.50.1 -Port 8765`
  from the appropriate side.
- **Authentication rejected:** pair the secret again. Never weaken or disable
  HMAC to diagnose networking.
- **Model verification reports missing assets:** rerun
  `./scripts/lumen-link-wsl provision --apply`; downloads are checksummed and
  resumable.
- **Poor disk performance:** confirm `pwd` begins with `/home/`, not `/mnt/c/`.
- **Windows loses internet:** the direct adapter was misconfigured. Remove any
  gateway or DNS from it; Lumen Link needs only its `/24` address.

See [Threadripper compute-node link](threadripper-compute-node.md) for the
authority boundary and full acceptance criteria.
