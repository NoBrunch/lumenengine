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
- `training`: lossless PCM capture, frame-synchronized semantic labels,
  checksummed segments, and model-dataset export
- `structure` / `datasets`: normalized independent functional, energy, and
  content axes plus EDM-98, Harmonix, CCMusic, and SALAMI adapters
- `research`: dependency-free source/environment/readiness management
- `offline`: coherent recording preparation, durable teacher jobs, normalized
  predictions, and streaming-student example alignment
- `student`: small causal NumPy neural network for CPU structure inference
- `choreography`: complete sequence preference learning and boundary leases
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
7. Background training-audio and annotation persistence

Communication between loops should use immutable snapshots and bounded queues.
The training recorder follows this rule: the audio loop performs a bounded,
non-blocking enqueue, while a dedicated writer owns WAV I/O, SHA-256 hashing,
and batched SQLite annotation writes. A saturated queue is represented as
dropped-frame metadata and a timeline gap rather than stalling DMX output.

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

## Rehearsal boundary

The rehearsal runtime is an explicit programming path, separate from live
audio interpretation and feedback:

```text
Selected routine + scope + tempo + size + look
                         ↓
              exact generated beat clock
                         ↓
             normal resolver and fixture output
                         ↓
                 Virtual DMX or live rig
```

Rehearsal bypasses the song planner but does not bypass the resolver, saved
calibration envelope, fixture profiles, or DMX transport. This makes it useful
for testing the actual building blocks that the planner may select. Auditioning
does not create supervision; only the explicit preferred-action placement does.
Preview/live output selection is immutable while rehearsal is running because
the physical transport is chosen when the runtime opens.

Motion Studio edits the shared definition of a semantic routine. Cycle length,
path travel/center, fixture relationship, direction, and center-body/arm travel
therefore affect both rehearsal and later automatic performances that select
that routine. Tempo, audition scale, intensity, strobe, palette, and fixture
scope belong to the rehearsal instance and do not mutate the shared definition.

The operator surface separates parameter families—movement, intensity, color,
and strobe—so one controller owns each attribute at a time. This mirrors the
effect/preset/cue separation used by professional console workflows and avoids
multiple layers fighting over the same DMX channel.

The operator controls apply centered biases to the audio-derived expression;
they do not replace energy or motion. Spotify polling and persistent trace
writes run outside the audio/DMX loop. Compact half-second performance samples
are retained locally for last-run diagnosis and future training data.

Monitor and Live sessions additionally retain the original 48 kHz stereo PCM16
stream in one-minute WAV segments plus a ten-Hz semantic timeline. Each
semantic frame uses the audio-frame index as its authoritative clock and
includes the musical observation, heuristic decision, complete fixture DMX,
Spotify identity, and operator controls. Feedback rows link back to that same
session and audio frame.

The heuristic decision is explicitly marked as baseline context rather than
ground truth. Model exports create feedback-centered audio windows and require
song/session-level data splits to prevent temporal leakage. A learned model
should produce semantic decisions and choreography plans; geometry,
calibration, output encoding, and hardware transport remain deterministic.

## Research and model boundary

The research stack is isolated from the live environment:

```text
Public annotations ──────────────┐
                                 ├─ normalized multi-axis timelines
Captured coherent recordings ───┤
                                 └─ offline teacher predictions
                                              ↓
                                 causal 10 Hz student examples
                                              ↓
                              small CPU streaming structure model

DMX history + preferred actions + feedback
                                              ↓
                           complete choreography sequence ranker
                                              ↓
                       boundary-leased semantic performance plan
                                              ↓
                      deterministic rig/spatial/DMX realization
```

EDMFormer runs in an isolated Python environment as an offline CPU teacher.
SongFormer also runs as an isolated CPU teacher using the official MuQ,
MusicFM, strict EMA head, and post-processing. Its context is capped at the
measured 30–60 second range to fit the target computer instead of using the
upstream 420-second CUDA context; that adaptation is retained in timeline
provenance. A missing or failed teacher never prevents the live engine from
starting after the offline worker has stopped. Lumen does not permit an engine
mode and an offline teacher/student worker to run concurrently on the target
computer; a cancellation request must reach its checkpoint before the engine
can start. Capture-manifest preparation is similarly serialized against manual
exports and engine startup.

Teacher eligibility requires at least 10 seconds of coherent recording audio.
Short identity-boundary fragments remain preserved capture evidence, but queue
preparation reports `recording_too_short`; the worker enforces the same rule
before resolving model paths or loading a heavyweight dependency.

Offline teacher timelines are relative to the reconstructed capture. Student
targets therefore join on `recording_offset_ms`, derived from the exact audio
frame and span start. Provider playback position remains in `position_ms` for
song-level provenance but is never used as the teacher's zero point. This
keeps mid-song and partial-track captures trainable without losing their
Spotify context.

Student partitioning follows the provider/song identity rather than the
recording-version PCM hash, preventing repeated analog captures of one song
from crossing train/validation/test boundaries. Unidentified captures use the
capture-session identity as their split group. Rows retain both the human-
readable teacher provenance and the segment's versioned provenance details.

The live student is trained from causal ten-Hz frames but audio arrives in
larger PCM packets at approximately 23.44 packets per second. Prediction passes
the authoritative audio sample-clock timestamp to the student, which advances
its causal memories by elapsed seconds rather than by packet count. Without
this boundary conversion the intended 0.5/2/8/30/60-second memories would be
compressed by more than a factor of two.

Model fusion is confidence gated per structure axis. Energy labels require
0.52 confidence, functional labels 0.60, and content labels 0.55 before they
enter runtime choreography context. Below those thresholds the diagnostic
prediction is retained but the runtime receives `unknown`; rejected energy also
contributes zero structural motion and zero boundary expansion. Accepted energy
labels may replace the authored analyzer's energy section and flow into the
expression policy. Functional and content labels remain independent context for
phrase-level candidate ranking; a label such as `chorus` must not masquerade as
an energy state such as `build` or `release`. Actual routine changes remain
leased to choreography boundaries.

Dataset and teacher conclusions retain provenance, source/model revision,
preprocessing version, confidence, and stable recording identity. Dataset
partitions are grouped by source track to prevent leakage. Teacher examples
reuse the exported provider-track split group across repeated analog captures;
only unidentified audio falls back to a capture-session group.

The sequence model can update while a phrase is playing, but
`BoundarySequencePlanner` leases the selected plan until its boundary changes.
Repeated feedback raises evidence and urgency without replacing the running
motion. Preferred sequences become candidates at later boundaries, where their
step duration, fixture scope, intensity, palette, and strobe character are
realized by the runtime. The model state is local, event-backed versioned JSON;
deleting feedback removes its identified update and replays the retained
events. It ranks semantic sequences, not raw DMX frames.
