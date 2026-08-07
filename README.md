# Lumen Engine

Lumen Engine is a private, local-first system for making a lighting rig listen,
remember, aim, and perform expressively. It is being built for one dedicated
Ubuntu PC in a garage audio system. It is a separate project from Party Parrot.

The source repository is public. Recordings, credentials, tokens, learned
preferences, runtime databases, and downloaded research assets remain private,
local, and ignored by Git.

## Start Lumen

Normal operation does not use the command line. Double-click **Lumen Engine**
on the Ubuntu desktop. It starts the local operator service and opens the
desktop console in the default browser.

The application has two purpose-built surfaces:

- **Desktop console** at `http://127.0.0.1:4042/`: live performance state,
  room and fixture editing, beam solving, DMX patching, calibration, audio
  analysis, song memory, hardware status, logs, and keyboard controls.
- **Phone/tablet remote** at `http://<this-computer-ip>:4042/remote`: current
  song and gesture, performance influence, quick character presets, feedback,
  notes, and blackout. The System page displays the exact address for this PC.

The desktop console has three start modes:

- **Monitor** listens to line-in and runs the entire engine through virtual
  output.
- **Perform** listens to line-in and drives the FT232R/Open-DMX interface.
- **Demo** runs a built-in musical demonstration so the interface and resolver
  can be exercised without playing audio or opening the DMX adapter.

The command-line tools below remain available for development and diagnostics.

## What works

- Meter-based, Z-up room and fixture configuration
- Calibrated 3D target-to-pan/tilt solving
- Alternate mechanical solutions and continuity-aware path choice
- Fixture movement speed limiting
- Virtual DMX frame generation
- Direct FT232R/Open-DMX output migrated from Party Parrot
- Optional direct Art-Net packet output
- Read-only import of Party Parrot's active show database
- Exact active-garage profiles for the 11-channel RGBW movers and 19-channel
  rotating multi-effect
- NumPy spectrum analysis and an onset-autocorrelation tempo clock that rejects
  syncopated false beats and maintains an explicit four-beat phase grid
- ALSA PCM16 capture with a noise gate, transient envelope, beat pulse, beat
  phase, and four-beat bar clock
- Full-envelope, bar-synchronized performance trajectories plus short beat
  accents for mover position, brightness, color, emitters, lasers, ring
  programs, and strobe
- Tempo-aware compound control of the center 19-channel rotating multi-effect
- Interpretable energy, tension, motion, and intimacy state
- Explainable gesture selection
- Private SQLite song, analysis, routine, decision, and feedback memory
- Lossless local training capture: 48 kHz stereo WAV segments, exact
  audio-frame synchronization, ten-Hz semantic/DMX context, linked feedback,
  checksums, storage controls, and model-ready JSONL manifests
- Provider-neutral media identity
- Built-in Spotify Connect console with search, transport, queue, seeking,
  volume, device selection, album art, and now-playing metadata
- A complete simulation from musical observations to virtual DMX
- A dark, KDE-inspired technical desktop console with hotkeys
- A responsive phone/tablet influence and feedback remote
- Live room/beam visualization, target solving, patch and calibration editing
- Real PCM waveform, packet heartbeat, dBFS/RMS/peak/clipping proof, rhythm
  lock, expression meters, event log, and DMX heatmap
- One-click desktop launcher
- **Lumen Link** private-LAN offload for full-song EDMFormer and SongFormer
  work, student training, and held-out evaluation on the Threadripper WSL
  node, with authenticated resumable transfers, deterministic artifact import,
  resource telemetry, and a desktop/phone status dashboard

## Neural training collection

Monitor and Perform modes record the same line-in PCM used by the live
analyzer. The recording runs on a background writer and is divided into
one-minute lossless WAV files under `state/training/audio`. Every teaching
moment is linked to an exact audio frame, and ten semantic frames per second
preserve the observation, expression, routine, complete fixture output, media
identity, and operator settings. The feedback surfaces also provide structured
song-context and preferred-next-action labels without interrupting the active
routine.

Use **Audio laboratory → Neural training dataset** to enable collection, choose
the storage ceiling, inspect capture health, and build a local JSONL training
manifest after stopping the engine. Current heuristic output is retained only
as baseline context; it is not labeled as correct behavior.

The research layer normalizes EDM-98, Harmonix, CCMusic, and SALAMI onto
independent functional, energy, and content timelines. EDMFormer and the
CPU-bounded SongFormer runner label completed captures as isolated offline
teachers. Those labels are aligned to Lumen's causal ten-Hz features for a
small CPU student model. Lighting taste is learned separately as complete
beat-addressed choreography sequences, and feedback updates apply at the next
phrase boundary rather than interrupting the current motion.

Use **Analyze new recordings** in the Musical-structure research panel to run
both teachers as a resumable batch. The panel exposes recording progress,
estimated time, database-verified examples, song-separated held-out material,
label balance, process memory, recovered interrupted jobs, and exact failures.
EDMFormer uses bounded 30–60-second CPU windows plus an 8 GiB process-group
memory limit, so offline research cannot consume all memory on the target PC.
The limit is a cutoff rather than a throttle and is adjustable through
`LUMEN_OFFLINE_MAX_RSS_GIB` if a future local model demonstrates a legitimate
larger working set.
**Train and validate** builds a causal student
with 0.5–60 second context and a separate boundary output. Training first
writes a candidate; only a candidate that passes held-out song gates becomes
the active Live model. Maintenance commands are
`research-status`, `research-import-annotations`, `research-worker`, and
`research-train-student`.

The `live-demo` and `dmx-blackout` commands write directly to the FT232R cable.
The plain `demo` command remains virtual.

## Lumen Link

The standalone **Lumen Link** console page connects this computer to the
Threadripper over the dedicated `192.168.50.0/24` Ethernet link. Lumen keeps
the canonical database, line-in clock, feedback, Live engine, and DMX output;
the WSL node receives only immutable checksummed offline job objects. It runs
EDMFormer, SongFormer, and `student.train`; the training result includes a
candidate model and held-out evaluation report. Lumen verifies and imports
every returned artifact locally. Remote completion never activates a model or
grants the Threadripper access to the writable database, Live timing, or DMX.

The dashboard authenticates and checks the worker contract before routing one
eligible automatic job at a time. **Disable link** returns unstarted work to
local eligibility while allowing an already-active job to finish safely. The
two-PC physical acceptance test still needs to be performed after deployment;
the repository test suite does not claim cable, restart, or local/remote
inference parity by itself.

The complete beginner-oriented Windows, WSL, network, pairing, and recovery
procedure is in [Lumen Link deployment](docs/lumen-link-wsl-deployment.md).
The Windows-side Codex handoff is in
[Lumen Link Codex handoff](docs/lumen-link-codex-handoff.md).

## Run it without installing anything

```bash
cd /home/the-system/Desktop/lumenengine
./scripts/lumen demo
```

Import the latest active Party Parrot show and run it virtually:

```bash
./scripts/lumen import-party-parrot
./scripts/lumen demo --rig config/party-parrot-active.json
```

Inspect the Open-DMX connection discovered on this computer:

```bash
./scripts/lumen dmx-devices
```

Party Parrot currently identifies it as the FT232R/Open-DMX adapter at USB
`0403:6001`. Lumen uses the installed `libftdi1.so.2` directly and repeats a
full 512-channel frame at 40 Hz.

Write directly to that interface:

```bash
# Stop Party Parrot first; the USB device has one owner.
./scripts/lumen dmx-blackout
./scripts/lumen live-demo --rig config/party-parrot-active.json --duration 30
```

Run the current audio-reactive engine continuously from line-in:

```bash
./scripts/lumen run --rig config/party-parrot-active.json --device default
```

The optional tty diagnostic fallback is:

```bash
./scripts/lumen live-demo --driver tty --port /dev/ttyUSB0
```

It requires `pyserial`; the normal native path does not.

Solve a single room target:

```bash
./scripts/lumen target 0 0 1.2 --rig config/party-parrot-active.json
```

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Check the deployment machine:

```bash
./scripts/lumen doctor
./scripts/lumen audio-devices
```

Listen to line-in for ten seconds without generating DMX:

```bash
./scripts/lumen listen --device default --duration 10
```

## Spotify music console

Spotify is an optional account, library, playback-control, and identity source.
The line-in remains authoritative for musical timing and Lumen's own analysis,
so the engine also works for instruments and other non-Spotify sources.

1. Open **System → Spotify playback identity** in the desktop console.
2. Use its link to create a private app in the Spotify developer dashboard and
   select the Web API.
3. Add this exact redirect URI:

   `http://127.0.0.1:8765/callback`

4. Paste the client ID into Lumen and press **Connect Spotify**. A client secret
   is not required because Lumen uses PKCE.
5. Approve the private app once in the desktop browser, then open
   **Spotify console** in Lumen.

Use **Open Spotify player + device picker** for Spotify's full browsing
experience and to choose Chromecast Audio. Lumen follows the device that is
active in Spotify: normal transport and song-selection commands intentionally
omit a device ID. The optional transfer selector lists only devices Spotify
exposes through its Web API; Spotify documents that some device models are not
returned there.

Lumen reads the current artist, title, album, track length, live track position,
playing state, playlist context, and active-device name. Its library panel
lists the account's playlists, lets the operator play a playlist context, and
can start a selected playlist track without losing next/previous context.

The equivalent command-line login and inspection commands remain available for
diagnostics:

```bash
export LUMEN_SPOTIFY_CLIENT_ID='your-client-id'
./scripts/lumen spotify-login
./scripts/lumen spotify-now --remember
```

The token is stored at
`~/.local/state/lumenengine/spotify-token.json` with owner-only permissions. Do
not put credentials or tokens in this repository.

Record feedback using the Lumen song number printed by `spotify-now --remember`
or `demo`:

```bash
./scripts/lumen feedback 1 too_busy --position-ms 95000 \
  --note "I liked the convergence, but there was too much motion after it."
./scripts/lumen memory 1
```

Your own words are retained. The short label simply gives future preference
learning a stable category.

## Coordinates

Lumen's internal room coordinates are right-handed and measured in meters:

- X: room left to room right
- Y: front of room to back of room, with the floor center at zero
- Z: floor to ceiling

Fixture housing rotations use intrinsic XYZ Euler order and are stored in
degrees in the initial JSON configuration. Adapters for other systems are
responsible for converting their coordinate and rotation conventions.

`example-rig.json` remains a synthetic development room.
`party-parrot-active.json` is generated from the real active Party Parrot show.

## Output boundary

The engine keeps perception, choreography, fixture realization, and transport
as separate layers:

```text
audio and media identity
        ↓
musical observations
        ↓
expression and gestures
        ↓
spatial fixture realization
        ↓
DMX frame
        ↓
USB Open-DMX or Art-Net transport
        ↓
physical output
```

The Open-DMX adapter repeats the latest full 512-channel universe on a dedicated
40 Hz thread, matching Party Parrot's proven behavior. Only one program can own
the FT232R device at a time, so stop Party Parrot before running Lumen's direct
output commands.

See [Architecture](docs/architecture.md),
[Threadripper compute-node link](docs/threadripper-compute-node.md), and
[Roadmap](docs/roadmap.md) for the decisions and next milestones. The
[progress log](CHANGELOG.md) records restorable versions, and the
[backup and restore runbook](docs/backup-and-restore.md) explains how to rebuild
the machine and return safely to the last recorded operating state.
