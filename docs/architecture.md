# Architecture

## Design commitments

Lumen is local-first, understandable, and rig-aware. Expensive analysis or a
network outage must not stall time-critical lighting output. Raw audio is not
required to leave the PC.

The current modules are:

- `audio`: line-in capture and musical observations
- `media`: optional recording identity providers
- `memory`: private song and performance knowledge
- `expression`: interpretable expressive state and gesture policy
- `spatial`: calibrated 3D inverse kinematics
- `dmx`: frame realization and isolated outputs
- `runtime`: one-way coordination of the above
- `config`: validated room, fixture, calibration, and patch data

## Timing domains

Spotify or another metadata provider can identify a recording and estimate
playback position. It is never the precise lighting clock. Audio samples from
line-in establish beat and onset timing.

Runtime work will eventually be separated into bounded loops:

1. DMX output and watchdog
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
into the room. Model-specific calibration is required before physical use.

## Learning boundary

The authored expression policy is the baseline and always remains available.
Learning should initially adjust understandable preferences:

- motion density by section
- brightness restraint
- gesture acceptance or rejection
- color-family preference
- desired contrast between sections
- repetition tolerance

It should learn semantic decisions, not raw DMX bytes. Geometry, constraints,
output encoding, and safety remain deterministic.

