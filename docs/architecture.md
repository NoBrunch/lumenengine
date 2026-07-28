# Architecture

## Design commitments

Lumen is local-first, understandable, and rig-aware. Expensive analysis or a
network outage must not stall time-critical lighting output. Raw audio is not
required to leave the PC.

The current modules are:

- `audio`: line-in capture and musical observations
- `beat`: Party Parrot's trigger tracker plus a spectrum-onset autocorrelation
  tempo clock for stable full-mix BPM and phase
- `media`: optional recording identity providers
- `memory`: private song and performance knowledge
- `expression`: interpretable expressive state and gesture policy
- `spatial`: calibrated 3D inverse kinematics
- `dmx`: frame realization and isolated outputs
- `usb_dmx`: native libftdi and tty Open-DMX transports
- `profiles`: fixture capabilities and channel layouts
- `party_parrot`: read-only show database importer
- `runtime`: one-way coordination of the above
- `config`: validated room, fixture, calibration, and patch data

## Timing domains

Spotify or another metadata provider can identify a recording and estimate
playback position. It is never the precise lighting clock. Audio samples from
line-in establish beat and onset timing.

Runtime work will eventually be separated into bounded loops:

1. DMX output
2. Audio capture and time stamping
3. Live feature extraction
4. Gesture planning and spatial realization
5. User interface and 3D preview
6. Background song analysis and preference learning

Communication between loops should use immutable snapshots and bounded queues.

## Spatial convention

The canonical mathematical beam direction for mechanical angles is:

```text
x = cos(tilt) cos(pan)
y = cos(tilt) sin(pan)
z = sin(tilt)
```

Fixture calibration supplies mechanical offsets, axis directions, limits, DMX
inversion, and speed limits. Housing rotation maps that fixture-local direction
into the room.

Party Parrot imports anchor each fixture's saved home DMX values to room center,
then preserve its saved room endpoints as the reachable mechanical/DMX window.
Further measurements can refine that bootstrap calibration without changing the
solver.

## Learning boundary

The authored expression policy is the baseline and always remains available.
Learning should initially adjust understandable preferences:

- motion density by section
- brightness restraint
- gesture acceptance or rejection
- color-family preference
- desired contrast between sections
- repetition tolerance

Feedback context is captured with every operator action: song identity,
playback position, active gesture, inferred section, expression values, BPM,
scope, and fixture/group target. The runtime aggregates this memory at overall,
fixture, song, artist, and song-section levels. Semantic routines store these
moments and are resolved into the current rig rather than replaying old DMX.

It should learn semantic decisions, not raw DMX bytes. Geometry, constraints,
output encoding, and hardware transport remain deterministic.
