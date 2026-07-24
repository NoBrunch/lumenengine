# Lumen Engine

Lumen Engine is a private, local-first system for making a lighting rig listen,
remember, aim, and perform expressively. It is being built for one dedicated
Ubuntu PC in a garage audio system. It is a separate project from Party Parrot.

This repository is not connected to a public remote.

## What works in the first foundation

- Meter-based, Z-up room and fixture configuration
- Calibrated 3D target-to-pan/tilt solving
- Alternate mechanical solutions and continuity-aware path choice
- Fixture movement speed limiting
- Virtual DMX frame generation
- Explicitly armed output safety gate
- Optional Art-Net packet output behind that gate
- Dependency-free ALSA PCM16 capture and first-pass audio features
- Interpretable energy, tension, motion, and intimacy state
- Explainable gesture selection
- Private SQLite song, analysis, routine, decision, and feedback memory
- Provider-neutral media identity
- Optional Spotify PKCE login and now-playing metadata
- A complete simulation from musical observations to virtual DMX

No physical DMX command is exposed yet. That is deliberate: fixture profiles and
calibration must be measured before hardware output is enabled.

## Run it without installing anything

```bash
cd /home/the-system/Desktop/lumenengine
./scripts/lumen demo
```

Solve a single room target:

```bash
./scripts/lumen target 0 2.5 1.2
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

The current sandbox used to build the project cannot see the host ALSA sound
card. `doctor` and `audio-devices` therefore report it as unavailable here. Run
the same commands from the PC's normal terminal to identify the real line-in
device.

## Spotify identity

Spotify is an optional identity source. The line-in remains authoritative for
musical timing and Lumen's own analysis.

1. Create a private app in the Spotify developer dashboard.
2. Add this exact redirect URI:

   `http://127.0.0.1:8765/callback`

3. Copy its client ID. A client secret is not required because Lumen uses PKCE.
4. Connect locally:

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
- Y: front of room to back of room
- Z: floor to ceiling

Fixture housing rotations use intrinsic XYZ Euler order and are stored in
degrees in the initial JSON configuration. Adapters for other systems are
responsible for converting their coordinate and rotation conventions.

The example fixture geometry and DMX channels are placeholders. Do not use them
to drive real equipment.

## Safety boundary

The engine is divided so these concerns cannot silently arm one another:

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
explicit safety gate
        ↓
physical output
```

Future physical output must include an operator-visible armed state, blackout,
watchdog heartbeat, bounded pan/tilt speed, validated DMX patches, and safe
startup/shutdown behavior.

See [Architecture](docs/architecture.md) and
[Roadmap](docs/roadmap.md) for the decisions and next milestones.
