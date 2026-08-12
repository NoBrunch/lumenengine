# Lumen Link: Threadripper/WSL deployment

Lumen Link moves the heavy offline research pipeline from the dedicated Lumen
PC to the 128 GiB Threadripper: full-song EDMFormer and SongFormer analysis,
student-model training, held-out evaluation, and return of verified result
artifacts. Lumen remains the sole authority for line-in timing, feedback, song
memory, model activation, choreography and DMX. If the cable is unplugged or
Windows restarts, Live continues and remote jobs remain durable.

Lumen provides its coordinator console, and the Threadripper also hosts a
KDE-inspired, read-only compute dashboard. It shows truthful Lumen-contact
state, worker uptime, memory, parallel slots, job stage, progress, queue and
completion counts. Windows receives a **Lumen Link Dashboard** desktop
shortcut, while the WSL worker remains headless.

```text
Lumen PC                                      Windows + WSL Threadripper
192.168.50.2/24                               192.168.50.1/24

canonical DB ── immutable HMAC job bundle ──> content-addressed job store
Live + DMX    <── checksummed result bundle ── teachers / train / evaluation
dashboard    <──────── status/progress ─────── dependency-free HTTP :8765
```

Neither machine shares a writable SQLite database. Git contains source,
schemas and examples only. It must never contain recordings, job bundles,
tokens, the HMAC secret, runtime databases, models or learned preferences.

The private link does transfer the minimum immutable inputs required by each
job. Teacher jobs transfer their captured song WAV plus a versioned manifest.
Student training transfers selected, checksummed training examples and their
required local audio objects—not the writable Lumen database. Returned teacher
timelines, candidate models, and held-out evaluation reports are also private
runtime artifacts. They stay under ignored `state/` trees on both computers
and must never be staged with Git, attached to an issue, or pasted into a
Codex conversation.

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
./scripts/lumen-link-wsl install --apply --cpu-threads 48
```

Enter the Linux password when `sudo` asks. Provisioning installs:

- Git/LFS, FFmpeg, libsndfile, rsync, OpenSSH and build tools.
- Python 3.12.8 through a dedicated, release-pinned pyenv checkout for the
  Lumen core and isolated EDMFormer
  environment. Provisioning never relies on whichever Python Ubuntu happens
  to put on `PATH`.
- Python 3.10.20 through pyenv for the isolated SongFormer environment.
- The exact frozen CPU PyTorch 2.4, MuQ, MusicFM, EDMFormer and SongFormer
  dependencies already pinned in `config/research/`.
- Lumen's dependency-light core environment for student training, held-out
  validation, artifact packaging, and the authenticated compute service.
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
./scripts/lumen-link-wsl capabilities
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
quick, non-mutating everyday check. `capabilities` authenticates to the
running local worker and must report all three job types as `READY`:

```text
teacher.edmformer
teacher.songformer
student.train
```

`logs` follows new service messages continuously. Press `Ctrl+C` when you are
finished reading them; that stops only the log viewer, not the worker.

`student.train` includes held-out validation. Its return bundle contains the
candidate model and evaluation report; the Threadripper cannot activate that
model. Lumen verifies and imports the artifacts locally, and its existing
held-out activation rules remain authoritative.

The worker processes up to six teacher jobs concurrently. Each EDMFormer job
is limited to its validated eight-thread runner maximum, filling the 48-thread
WSL allocation when six are active. SongFormer jobs divide the same allocation
evenly. Student training still runs alone and may use the full allocation.
Its per-recording causal feature preparation uses at most 24 isolated workers,
publishes completed/total recording progress, and reuses immutable feature
caches after retries or later training snapshots. The worker accounts for the
entire student process group when enforcing the memory ceiling.
The per-job memory ceiling is 96 GiB on this 128 GiB computer; exceeding it
fails that disposable job instead of exhausting Windows and WSL.

The Lumen PC performs its exact training-readiness audit only after active Link
compute and result imports drain. Imports coalesce into one audit, which merges
one recording at a time in a low-priority, 3 GiB-address-space subprocess. A
readiness regression therefore fails the derived status refresh instead of
pressuring the 16 GiB desktop session or delaying Link telemetry.

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

The second line—the one containing `-Apply`—is the elevated Windows Link
setup command. Rerunning that same line is safe and refreshes the firewall,
desktop shortcut, scheduled watchdog, and current WSL forwarding.

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
`./scripts/lumen-link test`. That terminal check authenticates the worker and
requires all three job capabilities. The dashboard additionally compares each
job's code, teacher/model, preprocessing and ontology contract with the Lumen
PC before allowing it to be routed.

There is intentionally no per-song offload selector. The coordinator chooses
eligible automatic jobs and keeps up to six teacher slots supplied. A
student-training job waits for the teacher jobs and then runs alone so it may
use the full Threadripper. Use one job as the first canary:

1. Press **Enable link**. Wait for the first job to appear as uploading or
   running.
2. Press **Disable link** while that first job is active. Other queued
   automatic jobs return to local eligibility; the already-active remote job
   is allowed to finish and import its verified result.
3. Inspect that job's result and event history. For a teacher job, inspect the
   imported timeline. For `student.train`, inspect the candidate and held-out
   evaluation; activation remains a separate local decision.
4. If the canary is correct, press **Enable link** again for parallel bulk
   processing. **Pause dispatch** temporarily stops new routing; **Disable
   link** returns unstarted queued work to automatic/local eligibility. An
   already-active remote process is never discarded mid-result.

During the canary, confirm:

- the dashboard updates without blocking Spotify or the live audio display;
- unplugging Ethernet leaves the job queued and does not affect Live;
- reconnecting resumes without creating a duplicate job;
- the result checksum and code/model revisions verify before import;
- `teacher.edmformer`, `teacher.songformer`, and `student.train` are all shown
  as available;
- returned teacher timelines, candidate models and evaluation reports verify
  their immutable object hashes and job contract before local import;
- a remote candidate cannot become active merely because the remote job
  completed; Lumen's held-out validation and activation rules still apply.

Those cable/restart cases are physical acceptance work, not claims that this
deployment package has already exercised the two computers.

### Threadripper-local dashboard

After applying the Windows setup, open **Lumen Link Dashboard** from the
Threadripper's Windows desktop, or browse locally to:

```text
http://127.0.0.1:8765/dashboard
```

**WORKER ONLINE** means the WSL service and dashboard are responding.
**LUMEN CONNECTED** means an authenticated request arrived from the Lumen PC
within ten seconds; **LUMEN WAITING** means the worker is available but has not
heard from the coordinator recently. The page never exposes song identity,
recording identity, manifests, tokens or the pairing secret.

The installed Windows scheduled watchdog starts WSL and, once per Windows
login, checks Git, applies the current worker configuration, verifies the
pinned environments and models, and starts the worker. If the Git remote is
temporarily unavailable, it verifies and starts the current local revision.
Failed verification leaves the worker stopped and retries after five minutes.
After successful startup, the watchdog repairs NAT forwarding and service
state every 30 seconds. The WSL service itself uses `Restart=always`.

The setup installs a real **Lumen Link Dashboard** shortcut with a Lumen icon
and removes the obsolete `.url` shortcut. Its target is always the
Threadripper-local address `http://127.0.0.1:8765/dashboard`; the dedicated
`192.168.50.1` address is only for traffic arriving from the Lumen computer.

## Updating Lumen Link

The scheduled startup task normally performs this automatically. To update
immediately without restarting Windows, run inside WSL:

```bash
cd ~/lumenengine
./scripts/lumen-link-wsl stop
git pull --ff-only
./scripts/lumen-link-wsl configure --apply
./scripts/lumen-link-wsl verify
./scripts/lumen-link-wsl start
```

After pulling a version that adds or changes Windows integration, refresh the
scheduled task and shortcut once from elevated PowerShell without touching the
already configured Ethernet adapter:

```powershell
& "\\wsl.localhost\Ubuntu\home\lumen\lumenengine\scripts\lumen-link-windows.ps1" -Apply -RefreshOnly
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
