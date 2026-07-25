# Lumen Engine

Lumen Engine is a private, local-first system for making a lighting rig listen,
remember, aim, and perform expressively. It is being built for one dedicated
Ubuntu PC in a garage audio system. It is a separate project from Party Parrot.

This repository is not connected to a public remote.

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

The `live-demo` and `dmx-blackout` commands write directly to the FT232R cable.
The plain `demo` command remains virtual.

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

See [Architecture](docs/architecture.md) and
[Roadmap](docs/roadmap.md) for the decisions and next milestones.
