# Architecture

## Design commitments

Lumen is local-first, understandable, and rig-aware. Expensive analysis or a
network outage must not stall time-critical lighting output. Raw audio is not
required to leave the PC.

The current modules are:

- `audio`: line-in capture, sample-clock timing, musical observations, and a
  hysteretic section tracker
- `beat`: Party Parrot's trigger tracker plus a spectrum-onset autocorrelation
  tempo clock for stable full-mix BPM and phase
- `media`: optional recording identity providers
- `memory`: private song and performance knowledge
- `expression`: interpretable expressive state and gesture policy
- `runtime`: phrase-level routine planner that holds a named motif across a
  musical bar and applies contextual learned preferences
- `spatial`: calibrated 3D inverse kinematics
- `dmx`: frame realization and isolated outputs
- `usb_dmx`: native libftdi and tty Open-DMX transports
- `profiles`: fixture capabilities and channel layouts
- `party_parrot`: read-only show database importer
- `runtime`: coordination of the above, including beat/bar routine selection
- `config`: validated room, fixture, calibration, and patch data

## Timing domains

Spotify or another metadata provider can identify a recording and estimate
playback position. It is never the precise lighting clock. Audio samples from
line-in establish beat and onset timing.

Runtime work will eventually be separated into bounded loops:

1. DMX output
2. Audio capture and time stamping
3. Live feature extraction
4. Two-bar phrase planning and spatial realization
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
scope, routine, and fixture/group target. The runtime aggregates this memory at overall,
fixture, song, artist, and song-section levels. Semantic routines store these
moments and are resolved into the current rig rather than replaying old DMX.

The current phrase vocabulary is `breathe`, `fan_sweep`, `figure_eight`,
`opposing_chase`, `beat_nod`, and `counter_rotate`. A routine is held for a
two-bar phrase; beat accents continue inside it. Positive feedback reinforces the active
routine, while movement/timing corrections select or reject routine families.

The operator controls apply centered biases to the audio-derived expression;
they do not replace energy or motion. Spotify polling and persistent trace
writes run outside the audio/DMX loop. Compact half-second performance samples
are retained locally for last-run diagnosis and future training data.

It should learn semantic decisions, not raw DMX bytes. Geometry, constraints,
output encoding, and hardware transport remain deterministic.
