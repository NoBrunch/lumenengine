# Lumen Link: Codex handoff on the Threadripper

This is the exact handoff for continuing deployment with Codex inside Ubuntu
WSL. It contains no secret, token, recording or private database.

Command ownership is strict:

- Threadripper Ubuntu WSL: repository, Python/model provisioning and service.
- Elevated Windows PowerShell: direct NIC, firewall and WSL persistence.
- Lumen Ubuntu PC: `enp1s0`, secret pairing, dashboard test and canonical jobs.

Codex must always say which of those three terminals a command belongs in.

## Start Codex in the correct place

Open the Ubuntu app from Windows, then run:

```bash
cd ~/lumenengine
git pull --ff-only
git status --short
codex
```

If `git status --short` prints unexpected files, stop and inspect them before
pulling or installing. Runtime state belongs outside Git or in ignored paths.

Give Codex this prompt:

> Read AGENTS.md, docs/threadripper-compute-node.md,
> docs/lumen-link-wsl-deployment.md, and this handoff completely. This is the
> Threadripper WSL compute node for Lumen Link. Check the current branch and
> working tree, run `./scripts/lumen-link-wsl status` and `preflight`, then
> guide me through deployment. Pause before Windows administrator or Linux
> sudo steps so I can supply authority locally. Never request, print, commit,
> upload, or paste the HMAC secret, recordings, tokens, model artifacts,
> learned preferences, job bundles, or runtime databases. Keep the repository
> and all model/job data in the Linux filesystem under `~/lumenengine`, never
> `/mnt/c`. Use the pinned research dependencies and verify every asset. Do
> not alter the Lumen PC's live database and do not make the compute node a
> live/DMX authority.

## Checkpoint 1: read-only discovery

Codex should run only:

```bash
pwd
git status --short
git log -1 --oneline
./scripts/lumen-link-wsl status
./scripts/lumen-link-wsl preflight
./scripts/lumen-link-wsl install
```

The last command is a dry run. No network, sudo or files are changed.

## Checkpoint 2: Linux sudo and downloads

After reviewing the dry run, tell Codex:

> I am ready for the documented WSL installation. Run the apply command and
> pause while I enter sudo locally. Do not display credentials.

The apply command is:

```bash
./scripts/lumen-link-wsl install --apply --cpu-threads 24
```

Codex should then run:

```bash
sudo loginctl enable-linger "$USER"
./scripts/lumen-link-wsl verify
```

The status and verification output must report Python 3.12.8 for the pinned
core interpreter, core virtual environment and EDMFormer environment, plus
Python 3.10.20 for SongFormer. A system `python3` version is not sufficient
evidence.

## Checkpoint 3: Windows administrator networking

Codex cannot silently grant itself Windows administrator privileges. In an
elevated Windows PowerShell, first discover the dedicated adapter:

```powershell
Get-NetAdapter
```

Then follow section 4 of `docs/lumen-link-wsl-deployment.md`. Run the Windows
script once without `-Apply`, inspect its report, and only then run it with
the correct `-InterfaceAlias` and chosen `Mirrored` or `Nat` mode.

Tell Codex when that step is complete. It should verify that:

- Windows owns `192.168.50.1/24` on only the dedicated port;
- no gateway or DNS was assigned there;
- TCP 8765 is accepted only from `192.168.50.2`;
- the logon startup task exists in either mode, while port forwarding exists
  only in NAT mode.

## Checkpoint 4: local pairing

The secret is already stored at `~/.config/lumen-link/shared-secret` in WSL.
At the **Lumen Ubuntu PC** terminal first run
`./scripts/lumen-link network` and check the dry run, then apply
`./scripts/lumen-link network --apply --interface enp1s0`. Confirm the default
route still uses Wi-Fi. Next use `./scripts/lumen-link pair --apply` and paste
the secret only into the hidden local prompt, or use a removable-media file
with `--secret-source`. Do not give its value to Codex. Do not copy it using
Git. Codex may verify only file existence and mode `600`.

## Checkpoint 5: start and prove the worker

Codex may run:

```bash
./scripts/lumen-link-wsl start
./scripts/lumen-link-wsl status
./scripts/lumen-link-wsl logs
```

After the Lumen dashboard reports **Ready**, use **Test connection** before
sending work. The first release supports remote `teacher.edmformer`;
SongFormer, student training and held-out evaluation must remain shown as
gated.

For the canary, press **Enable link**, wait until its automatically chosen
single EDMFormer job is active, then press **Disable link**. The active job is
allowed to finish and import; other queued automatic jobs return to local
eligibility. Inspect its verified result. If accepted, press **Enable link**
again for sequential bulk processing. There is no per-song selector in v1.
Codex should inspect status, progress, result checksums and service logs
without opening, copying or summarizing song audio.

## Ready for the owner's physical acceptance test

The software-side deployment is ready for physical acceptance only when:

- installation and strict model verification pass;
- the Lumen dashboard authenticates over the direct cable;
- the first result passes schema, revision and checksum checks and imports
  exactly once;
- no private artifact appears in `git status` or the Git history.

Local/remote inference parity, Windows/WSL restart recovery, interrupted
transfer resume, and cable-loss isolation are the remaining physical checks.
Do not report any of them as proven until they have been exercised on the two
PCs.
