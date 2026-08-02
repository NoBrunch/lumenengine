# Lumen Index

Living operator and development reference for Lumen Engine.

Lumen is a private, local audio-to-DMX lighting engine for the garage audio
system. It listens to the PC's line input, interprets musical energy and
timing, chooses an expressive lighting gesture, resolves that gesture into the
configured room, and sends DMX to the patched fixtures.

This file is intentionally kept in the project directory. When a feature,
control, term, or workflow changes, update this index in the same change.

## Contents

1. [Quick start](#quick-start)
2. [System flow](#system-flow)
3. [Operator features](#operator-features)
4. [Rehearsal](#rehearsal)
5. [Feature impact](#feature-impact)
6. [Feedback and learning](#feedback-and-learning)
7. [Neural training dataset](#neural-training-dataset)
8. [Musical-structure research stack](#musical-structure-research-stack)
9. [Calibration](#calibration)
10. [Spotify](#spotify)
11. [3D room interaction](#3d-room-interaction)
12. [Troubleshooting](#troubleshooting)
13. [Development maintenance](#development-maintenance)
14. [Glossary](#glossary)

## Quick start

1. Launch Lumen from the desktop shortcut.
2. Open the desktop console on the Lumen PC.
3. Confirm the Audio laboratory shows fresh PCM from the selected line input.
4. Confirm the DMX adapter appears in System diagnostics.
5. Use Monitor to verify audio analysis without physical light output.
6. Use Live when the rig and DMX output are ready.
7. Use the phone/tablet `/remote` page for quick influence and feedback.
8. Stop the engine from the desktop console before changing rig calibration.

The desktop console is for technical work: audio proof, DMX diagnostics,
fixture patching, calibration, room geometry, Spotify setup, and memory. The
remote page is for listening: intensity, movement, presets, blackout, and
feedback.

## System flow

```text
Line input / Spotify metadata
        ↓
Audio capture, sample-clock timing, and spectral analysis
        ↓
Musical observation
        ↓
Expression engine
        ↓
Gesture and motion choreography
        ↓
Spatial targeting and fixture calibration
        ↓
DMX frame
        ↓
Open-DMX interface and fixtures
```

Spotify supplies identity and playback metadata. The line input remains the
authoritative timing source for the lighting response.

## Operator features

### Performance console

The main page shows:

- Current recording, artist, position, and duration when Spotify identifies it.
- Audio energy, beat confidence, section, and expression state.
- Live room composition, resolved beams, fixtures, and target.
- Operator influence controls.
- Feedback controls for the current musical moment.
- Interpretation and system event history.

The 3D room view is available from the room-view buttons. The rig view can be
used to inspect and edit fixture placement.

### Audio laboratory

The Audio laboratory provides proof that the PC is receiving audio rather than
merely running a visual demo. It displays the waveform, frequency bands, onset
strength, beat lock, packet counts, and input diagnostics.

### Engine modes

- **Monitor**: captures and analyzes audio without intentionally driving the
  physical DMX rig.
- **Live**: captures the line input and sends generated DMX to the physical
  output.
- **Demo**: uses generated observations for testing the UI and choreography.
- **Rehearsal**: drives one selected routine with a stable generated beat clock,
  either into Virtual DMX or the physical rig.
- **Stop**: returns the engine to standby and closes the output.

### Operator controls

- **Master**: final output level.
- **Intensity bias**: preferred brightness/energy.
- **Movement bias**: preferred amount of pan, tilt, and fixture motion.
- **Spatial focus**: narrows or widens the generated room target.
- **Color warmth**: shifts the palette toward warmer or cooler expression.
- **How much Lumen follows you**: blends the automatic audio response with the
  operator's explicit intensity, movement, focus, and warmth preferences.
- **Presets**: restrained, balanced, open room, and drive control several of
  these values together.
- **Blackout**: gates the DMX output while preserving the current engine state.

### Feedback controls

Feedback may target:

- Overall performance
- Both movers as a group
- The center multi-effect fixture

The six quick calls shown during listening are:

- Keep this
- More
- Less
- Timing good
- Timing off
- Wrong look

The expandable correction drawer contains:

- Increase movement
- Decrease movement
- No strobes
- Strobe
- Faster/slower strobe
- Faster/slower
- Brighter/dimmer
- Faster/slower side arms
- Cooler/warmer palette
- Free-form note

Repeated quick calls increase the recorded urgency/agreement. They do not
interrupt the currently leased choreography phrase. **Undo last** removes the
newest feedback record and rebuilds its preference contribution.

The phone interface uses touch-gesture filtering so a scroll or screen-wake
gesture cannot be interpreted as a feedback click.

Free-form notes are interpreted when they contain direct cues such as “no
strobe,” “stop flashing,” “strobe,” “slow,” “faster,” “blue,” “purple,” “warm,”
or “amber.” Other prose is retained as memory but does not automatically change
a runtime parameter. This parser is deliberately local and deterministic; it
does not silently send private listening data to ChatGPT.

## Rehearsal

Rehearsal separates learning the lighting vocabulary from judging an automatic
show. Open workspace **2 — Rehearsal** and use the Movement Lab as follows:

1. Select one of Breathe, Fan sweep, Figure eight, Opposing chase, Beat nod,
   or Counter-rotate.
2. Choose **Preview only** to inspect generated DMX without the physical rig,
   or **Live rig** to send the routine through the Open-DMX adapter.
3. Choose Both movers, Center effect, or Whole rig. The movers are intentionally
   treated as one choreographic group; individual-mover selection is not part of
   the rehearsal vocabulary.
4. Adjust tempo, movement size, intensity, palette, and strobe. The generated
   clock is exact, so motion differences are not obscured by live beat analysis.
5. Use Previous/Next for individual routines or Tour to advance every eight
   beats.
6. Stop rehearsal before switching between Preview and Live output.

### Motion Studio

Motion Studio edits the actual routine generator while Rehearsal is running.
Its trajectory display uses horizontal position for normalized pan and vertical
position for normalized tilt. Cyan is the first mover; amber is the second.
The moving markers indicate the current point in the selected cycle.

Editable values are:

- **Cycle length**: beats required to complete the entire path. Longer cycles
  allow continuous movement across several bars while color and intensity can
  continue expressing individual beats.
- **Pan/Tilt travel**: proportion of the saved calibration envelope used by the
  routine. At 100% Pan/Tilt travel and 100% Audition scale, the path reaches
  the saved endpoints without another hidden movement reduction.
- **Pan/Tilt center**: shifts the path inside the calibrated envelope when its
  useful visual center is not the numeric midpoint.
- **Fixture relationship**: synchronized, exactly opposed, mirrored, evenly
  chased, or counter-direction movement.
- **Direction**: forward or reverse traversal.
- **Center body/arm travel**: separate movement amounts for the rotating body
  and two outer arms of the multi-effect fixture.

The velocity indicator compares the requested trajectory at the current
rehearsal BPM with each mover's saved pan/tilt speed model. **Velocity OK** means
the authored path fits; **Too fast** means the physical output would be
rate-limited and the visible shape could deform. Increase Cycle length or
reduce the affected travel until the indicator clears.

Edits are applied to Virtual or Live rehearsal immediately and stored in the
private `state/motion-routines.json` file. These values define the shared
routine generator: they also affect future automatic performances whenever the
planner selects that routine. Rehearsal tempo, audition scale, intensity,
strobe, palette, and scope are temporary audition controls and do not rewrite
the routine. **Reset this routine** restores only the selected routine's
authored defaults. Scope isolation explicitly zeros fixtures outside the
selected mover/center group.

Auditioning, adjusting, or touring routines does **not** teach the model. When a
routine is appropriate at a real song moment, use **Save song context** and
**Use this routine here**. That explicit placement stores a musical annotation
and a preferred-action annotation without confusing exploration with approval.

This follows the common console split between programming and playback: motion
effects have parameters such as rate, size, direction/phase, scope, color, beam,
and entry/exit behavior; a cue or timeline records where the resulting look is
used. Lumen currently exposes the parameters supported by its six authored
routine families. Song Rehearsal timelines and show review are subsequent
extensions of the same workspace rather than additions to live feedback.

### Memory

Memory stores recognized recordings, song identity, timing positions, feedback,
decisions, analyses, and routines in the local SQLite database. Recent
feedback is visible in the Memory page. Each recent feedback item has an
**Undo** action.

## Feature impact

### Audio response

Musical energy controls brightness and the size of movement envelopes. Soft
sections reduce activity. Strong onsets and beat pulses provide accents.

### Silence behavior

When the line input falls below the silence threshold, movers hold their last
position while the center effect transitions toward its parked state. The
center effect clears strobe and motor activity as the silence fade completes.

### Mover choreography

Movers use phase-offset patterns so the fixtures do not repeat the same motion
at the same time. Available motion families include figure-eight paths, circles,
pan sweeps, tilt/nod patterns, beat alternation, convergence, expansion, and
release gestures.

### Strobe behavior

Moving-head strobe remains off unless rehearsal, choreography, or positive
strobe feedback explicitly requests it. The center fixture may add an automatic
high-energy beat burst. A strobe request holds the fixture's characterized
strobe-rate channel for the requested interval; Lumen does not simulate it by
rapidly changing color or by sending a single-frame dimmer flicker. **No
strobes** suppresses the automatic burst and learned strobe preference.

### Center multi-effect

The center fixture changes motor speed and pattern with musical energy. It uses
alternating emitters, circles, sweeps, nods, laser accents, strip programs,
and strobe accents. Its activity is reduced during soft passages and parked
during sustained silence.

## Feedback and learning

Feedback is both stored and used.

Each feedback event has a label, value, timestamp, song, playback position,
active gesture, inferred section, energy, motion, tension, confidence, BPM,
scope, and optional fixture/group target. The label maps to movement,
intensity, strobe, and palette deltas. For example, **Too busy** contributes a negative movement
bias, while **Timing was right** contributes a smaller positive movement and
intensity bias.

Lumen aggregates this context at overall, fixture, song, artist, and song
section levels. Positive feedback reinforces the gesture that was active;
controls such as **Calm down**, **Pick it up**, and **More movement** also
nominate gesture families. Timing feedback can cause a learned gesture to be
replaced in the same context rather than changing every song globally.

### Decay

Historical influence uses exponential recency decay with a 21-day time scale:

```text
recency weight = exp(-age_in_days / 21)
```

Older feedback gradually matters less without being abruptly discarded.

### Confidence

Agreement confidence increases when the same preference is recorded repeatedly:

```text
confidence = min(1, sqrt(number_of_agreements) / 2)
```

The decayed weighted effect is multiplied by this confidence and clamped to the
engine's usable range. A single accidental or isolated opinion therefore has
limited long-term influence.

### Scope combination

Overall feedback applies to every fixture. Fixture feedback is added on top of
the overall bias for that fixture. The movers group expands into both moving
fixture IDs.

### Undo

Use Memory → Recent teaching moments → Undo. Lumen deletes the event, rebuilds
the decayed/confidence-weighted profile, removes that event from the
event-backed choreography learner, replays the remaining learning events, and
replaces the running runtime's feedback profile. The currently leased phrase
is still allowed to finish.

### Semantic routines and palette families

Every feedback event updates that song's semantic routine. The routine stores
positions, labels, gestures, sections, and scopes—not raw DMX bytes—so it can
be resolved against the current calibrated rig. The desktop and phone controls
also expose Automatic, Party vivid, Midnight teal, Cool blue/violet, Warm red/amber,
Magenta/blue, Cyan/violet, and Red/amber palette families. Palette feedback
shifts the family for the relevant learned context.

### Beat and motion behavior

Moving-head strobes are off by default. The center multi-effect retains the
beat-flash role, while its
ball and arm emitters exchange saturated palette colors on the beat. High-level
gesture changes are held for a readable phrase instead of being allowed to
alternate on adjacent audio packets. A named routine (`breathe`, `fan_sweep`,
`figure_eight`, `opposing_chase`, `beat_nod`, or `counter_rotate`) is held for
two bars and can be learned from contextual feedback. The center fixture and
movers consume the same routine. Compound motion uses the detected bar
phase when available, keeping sweeps, arm chases, and body rotation on the same
tempo grid as the movers.

The Audio laboratory chart shows 24 seconds of physical input level and
audio-derived expression energy. Section labels use temporal hysteresis rather
than classifying each PCM packet independently. A release is a short transition
event; builds, grooves, and breakdowns must persist before the label changes.

Lumen writes a compact performance trace twice per second. Each sample includes
the audio observation, expression, gesture, routine, controls, song position,
and resolved mover angles. This makes the most recent run reviewable even when
the gesture itself did not change.

## Neural training dataset

Lumen collects the information needed to replace or supplement its heuristic
interpretation with locally trained neural models. Collection is enabled by
default for **Monitor** and **Live** modes. **Demo** is never recorded.

Open **Audio laboratory → Neural training dataset** to see recording state,
duration, WAV segments, semantic-frame count, feedback labels, dataset size,
available disk space, dropped frames, and storage paths.

### What is recorded

- The original 48 kHz stereo signed-PCM16 line-in stream, without lossy
  compression, divided into one-minute WAV segments.
- A ten-Hz semantic timeline tied to exact audio frame numbers.
- Loudness, spectral bands, spectral flux and brightness, rhythm density,
  harmonic change, arrangement change, onset, tempo, beat/bar phase, section
  estimate, and confidence.
- Expression state, gesture, phrase routine, palette, operator controls, mover
  solutions, and complete DMX values for every patched fixture. Zero-valued
  channels are included so “off” remains meaningful.
- Spotify song identity and approximate position when available.
- Feedback labels and notes linked to the precise PCM frame being heard.
- Explicit song-context labels and preferred next-action labels from the
  desktop or phone feedback surface. These labels are recorded without
  interrupting or replanning the routine currently in progress.

Audio files live under `state/training/audio`. The database stores file
indexes, checksums, timing, semantic frames, and label links; it does not store
large audio blobs.

### Why runtime output is not ground truth

The present heuristic routine is saved as context because it explains what the
operator saw. It is explicitly marked
`heuristic_runtime_baseline_not_ground_truth`. Training must treat operator
feedback and later preferred-action labels as supervision rather than teaching
a model to imitate every current Lumen mistake.

### Building a model-ready manifest

Stop Monitor or Live mode, then press **Build training manifest**. Lumen creates
a local export under `state/training/exports` containing:

- `dataset.json`: format, validation result, counts, and split guidance.
- `sessions.jsonl`: capture settings and session metadata.
- `segments.jsonl`: WAV paths, sample ranges, byte counts, and SHA-256 hashes.
- `frames.jsonl`: the synchronized ten-Hz semantic timeline.
- `feedback_examples.jsonl`: sixteen-second audio windows centered on each
  teaching moment, including windows that cross WAV boundaries.
- `annotation_examples.jsonl`: the same synchronized windows for explicit
  musical-context and preferred-action labels.

The export references the original WAV files instead of duplicating them.
Train/validation/test partitions must be separated by song and session, not by
random neighboring frames, to avoid a model being tested on audio adjacent to
its training examples.

### Storage controls

The default ceiling is 100 GB and can be changed from 1–800 GB in the Audio
laboratory while the engine is stopped. Lumen does not automatically erase old
recordings. It refuses to start another capture after reaching the configured
ceiling or when less than 5 GiB remains free.

One hour of 48 kHz stereo PCM16 is approximately 659 MiB. A 100-GB ceiling
therefore represents roughly 155 hours before filesystem overhead.

## Musical-structure research stack

Lumen uses two separate learning systems:

1. **Musical understanding** learns what is happening in a song.
2. **Choreography preference** learns what the lighting should do about it.

Public research annotations provide general musical vocabulary. Local line-in
captures, corrections, preferred actions, and actual DMX history provide the
garage-specific lighting vocabulary.

### Dataset roles

- **EDM-98 / EDMFormer**: EDM energy form such as intro, buildup, drop,
  breakdown, outro, and silence. EDMFormer is an offline teacher.
- **Harmonix Set**: broad functional sections plus beats and downbeats.
- **SALAMI**: hierarchical structure from multiple annotators.
- **CCMusic**: pop functional structure. The authorized label-only package has
  300 timelines and 2,918 segments; Lumen excludes audio, mel images, and media
  links and preserves a manifest audit of two source repairs and thirteen
  source timeline discontinuities.
- **SongFormer**: a large general-purpose offline teacher. Lumen's isolated CPU
  runner uses the official MuQ, MusicFM, strict EMA head, and post-processing
  with a measured 30–60 second context window. The upstream 420-second CUDA
  context is recorded as an explicit adaptation in prediction provenance.

All imported labels are normalized onto independent axes:

- **Functional form**: intro, verse, pre-chorus, chorus, bridge, outro, and
  related song roles.
- **Energy form**: low, groove, build, release, breakdown, sustained, silence.
- **Content role**: vocal, instrumental, solo, transition, silence.

Every label retains provenance, model/dataset version, source file, confidence,
and split identity. Tracks—not neighboring frames—are assigned to training,
validation, and test partitions. Repeated or partial captures of the same
Spotify track keep the same provider-based split group even though their PCM
recording fingerprints differ; unidentified captures fall back to their
listening session.

### Capture-to-model workflow

```text
Completed PCM capture
        ↓
Integrity-checked, coherent recording WAV
        ↓
Queued offline teacher analysis
        ↓
Normalized multi-axis timeline
        ↓
Teacher labels aligned to Lumen's causal 10 Hz feature frames
        ↓
Small CPU streaming student training and held-out evaluation
        ↓
Cached model loaded for live first-play interpretation
```

Teacher alignment uses the reconstructed recording's own zero-based clock.
This matters when a capture begins partway through a Spotify track: the
original Spotify `position_ms` is retained as provenance, while
`recording_offset_ms` selects the matching teacher section. A capture starting
at 2:39 in Spotify therefore aligns its first feature frame to 0:00 in the
teacher timeline instead of producing an empty training file.

Training partitions use the stable song/provider group, such as a Spotify
track ID, rather than the analog capture's PCM fingerprint. Repeated listens to
the same identified track therefore remain in one train, validation, or test
partition even when line-in noise produces different recording hashes.
Unidentified audio is grouped only within its capture session. Each generated
target also carries the timeline version, teacher-run identity, and the
segment's teacher/version provenance.

Stopping Monitor or Live finalizes the capture and starts lightweight
preparation in a background thread. Heavy teacher inference is represented by
durable SQLite jobs and executes outside the DMX loop. The Audio laboratory's
**Musical-structure research stack** is the normal operator workflow:

1. After Monitor or Live stops, wait while **Preparing the most recent
   capture** is shown. Lumen verifies continuity and identity before deciding
   which recordings are full-song structure candidates. Captured partial and
   unidentified fragments remain useful evidence, but are reported separately
   instead of being presented as completed songs.
2. Press **Analyze new recordings**. Lumen rebuilds the verified manifest,
   queues both EDMFormer and SongFormer, and processes the durable queue as a
   batch. Held-out songs are prioritized so honest validation becomes
   available early. **Pause analysis** requests cancellation at the current
   job's checkpoint and returns unfinished work to the queue; completed jobs
   are retained. Press Analyze again to resume. Engine modes remain unavailable
   until preparation or the offline worker has actually stopped, preventing a
   heavyweight teacher from competing with audio/DMX timing.
3. Read captured recordings, structure-eligible songs, completed/total teacher
   jobs, percentage, estimated time, usable examples, held-out examples, label
   balance, and exact failures. The capture count includes every recording;
   eligible, partial, and unidentified counts come from the capture inventory,
   including fragments that were never queued for a teacher. An eligible song
   is shown as processed only after both configured teachers complete it. A
   preliminary model may be trained once trusted completed runs contain at
   least two distinct training-song groups plus a separate held-out song;
   remaining teacher work can continue later.
4. Press **Train and validate**. Automatic training accepts only example files
whose path, checksum, teacher-run ID, recording ID, and row count belong to
completed teacher runs in the active Lumen database. Unowned smoke-test or
stale JSONL files are ignored.

When both teachers label the same captured frame, Lumen merges their
complementary axes instead of duplicating the training example: SongFormer is
preferred for functional/content form and EDMFormer for energy form. Older
verified teacher exports that predate the recording-completeness snapshot can
inherit that snapshot from a newer complementary teacher. Two explicit but
contradictory snapshots are still rejected and reported as a provenance error.

The streaming student retains causal summaries at 0.5, 2, 8, 30, and 60
seconds. Live prediction advances those memories from the audio sample-clock
timestamp rather than the approximately 23.44-Hz PCM packet arrival rate, so
the windows retain the same real-time meaning as the 10-Hz training examples.
Its loss is class-balanced and includes a separate section-boundary
head. It is evaluated on a song that was not used for training. A newly
trained file is saved first as `lumen-structure-student.candidate.npz`; it
replaces the active model only if held-out energy, functional-section, and
boundary checks pass. A failed candidate remains inspectable and cannot
control Live.

Live predictions pass through a stable section decoder. A different label
must persist, and current regions have a minimum hold time; high boundary
confidence permits a faster legitimate transition. Energy, functional, and
content axes have independent confidence gates. Weak guesses remain visible
for diagnosis but are replaced with `unknown` before choreography ranking;
weak energy guesses also contribute neither boundary expansion nor motion
size. An accepted energy section can replace the live analyzer's
section and influence expression. Accepted energy confidence scales calibrated
motion expansion/contraction, while choreography still changes only on a
phrase boundary.

Recordings shorter than 10 seconds remain part of the capture record but are
explicitly marked `recording_too_short` and are not queued for EDMFormer or
SongFormer. The worker repeats this check before any model is loaded, so an old
or externally created short job is completed as skipped rather than consuming
teacher resources.

EDMFormer is locally adapted into independent 30–60-second CPU windows instead
of passing an entire song through the upstream 420-second transformer context.
Every teacher subprocess is also supervised as a complete process group. Lumen
records its current and peak resident memory and stops it at the default 8 GiB
offline limit before it can exhaust the 16 GiB lighting computer. The Audio
Laboratory reports current usage while analysis is running and presents a
memory-limit stop as an exact, retryable teacher failure rather than a generic
crash. This limit is an emergency cutoff, not a RAM allocation or throttle;
teachers use as much memory as their workload requires below it. A future
teacher with a measured legitimate need for more can use the
`LUMEN_OFFLINE_MAX_RSS_GIB` setting rather than changing live audio/DMX timing.

Offline jobs carry a worker identity, process ID, and heartbeat. If Lumen or the
desktop session ends during analysis, the next Lumen start—and every subsequent
Analyze request before it counts available work—returns the abandoned job to
the durable queue. A related unfinished teacher attempt is retained as failed
provenance; completed jobs and teacher results are never rewritten. Audio
Laboratory explicitly reports how many interrupted jobs were recovered, and
Analyze resumes them normally.

Technical maintenance commands:

```text
lumen research-status
lumen research-import-annotations
lumen research-prepare-export <export-directory>
lumen research-worker --max-jobs 1
lumen research-train-student
```

Normal listening does not require these commands.

When `lumen-structure-student.npz` exists and passes its feature/label contract
check, Lumen loads it at application startup. The Audio laboratory distinguishes
an artifact merely existing on disk from the runtime successfully loading it,
and reports a rejected latest candidate without hiding an older validated model
that remains active. It also reports raw and stable predictions, accepted axes,
boundary probability, confidence, model path, training-example count, and
activation state. If the file is absent, invalid, below the live confidence
gate, or a candidate fails validation, the existing live analyzer remains the
fallback.

### Complete choreography learning

The sequence preference model ranks ordered, beat-addressed plans rather than
only choosing one gesture. A plan can express “fan sweep for four beats, beat
nod for two, then opposing chase for two.” It considers song, artist,
functional/energy/content context, tempo, preferred actions, repeated feedback
urgency, age decay, and normalized DMX history.

Feedback never replaces a plan in the middle of its current musical boundary.
Repeated feedback increases evidence and urgency, but the active sequence is
leased until the next phrase or section boundary. This prevents multiple
phones or rapid taps from visibly interrupting a motion.

Preferred actions and characteristic corrections create reusable candidates
for later boundaries. Beat timing, step duration, fixture group, intensity,
palette, and strobe character are resolved when the learned step runs; they
are not merely stored as descriptive metadata. The interface labels and model
vocabulary share the same identifiers, including **No strobes**, **Not busy
enough**, **Too dim**, and faster/slower side-arm requests.

## Calibration

Open Room & rig on the desktop PC and select a moving head.

### Live mover calibration

1. Select a mover in Fixture inventory.
2. Press **Start calibration**.
3. Lumen moves it to that fixture's saved Home position and lights the beam.
4. Use **Pan position** and capture **Left**, **Home**, and **Right**.
5. Use **Tilt position** and capture **High**, **Home**, and **Low**.
6. Click any saved-position card to preview that point again.
7. Save the selected fixture. Lumen briefly pauses the engine, validates and
   writes the rig, then resumes the prior engine mode.

Calibration uses direct 8-bit DMX-style pan and tilt jog values (0–255) while
the mover is in the calibration state. The selected mover is visibly lit at
low brightness during jogging. The jog controls are deliberately whole-number
controls. The operator-facing readout is percentage of the fixture's travel;
raw 16-bit endpoints and internal angles are derived automatically.

### Envelope controls

- **Left/Right**: the horizontal edges where the beam remains useful in the
  room.
- **High/Low**: the highest and lowest useful vertical beam positions. These
  are visual room positions; there is no universal vertical degree value.
- **Home**: the useful neutral position for that axis.
- **Normal/reversed direction**: detected from the order of the captured
  physical points. Reversed is valid software calibration and does not require
  physically rotating the fixture.

Calibration values are the software movement envelope used by the resolver and
generated motion. They are room-use limits, not an additional hardware safety
system. Lumen maps musical gestures inside this envelope so motion is spent in
the visible part of the room. Pan captures never alter tilt, and tilt captures
never alter pan.

## Input level and clipping

The audio page reports clipping only when PCM16 samples reach actual digital
full scale. If that count rises, reduce the Ubuntu ALSA Line input level a
small amount (for example 3–6 dB) and leave the Chromecast output where it is.
The goal is headroom below 0 dBFS, not a quieter listening system.

## Party Parrot-inspired center fixture behavior

The center multi-effect now uses the proven Party Parrot style of saturated
foreground/background/contrast colors. Its ring walks through the fixture’s
built-in effect bank by musical bar, while the body and two arms rotate through
chase, opposing sweep, figure-eight, beat alternation, broad fan, and
counter-rotating-circle gestures. Energy controls the travel amount and speed,
so quiet sections settle without removing the routine variety.

## Desktop workspace sizing

Desktop dashboard panels can be resized from their lower-right corner and
scroll internally when text or diagnostics exceed the panel. The layout is
saved by the browser where supported; resizing is a view preference and does
not change the rig or DMX behavior.

## Spotify

Spotify integration provides:

- Account identity
- Current artist and track
- Album
- Track duration
- Playback position
- Play/pause, previous, and next
- Playlist selection
- Track search
- Active-device and supported-device transfer
- Active-device volume when Spotify exposes it

Spotify's own device picker remains the authoritative way to choose Chromecast
Audio. Lumen follows the active Spotify route while reading the physical line
input for timing.

The built-in Spotify page includes Back and Forward controls for the Lumen
playlist/search browsing history, plus Refresh player.

## 3D room interaction

In the rig's 3D view:

- Click-drag pans the room in space.
- Ctrl + click-drag rotates the room like a Fusion 360 orbit control.
- Mouse wheel zooms.
- Fixture labels, floor grid, walls, ceiling edges, target, and resolved beams
  provide spatial context.

The 3D display is a lightweight canvas projection designed for this dedicated
PC; it is not a full CAD renderer.

## Troubleshooting

### Feedback says it cannot be saved

Restart the Lumen service from the desktop shortcut so the current backend and
database migration are loaded. Check the selected feedback scope and fixture.

### Feedback appears to do little

One event is intentionally restrained by confidence weighting. Repeated,
consistent feedback has stronger influence. Check the Memory page to confirm
the event exists and inspect its scope.

### Movers do not move as expected

Check the fixture's calibration envelope, especially DMX 43. Use live
calibration, save the useful pan/tilt limits, then stop and restart the engine.

### Center effect remains active during silence

Confirm the Audio laboratory shows the line input as quiet and that the engine
has been allowed to complete the silence fade.

### No physical output

Check System diagnostics for the FT232R/Open-DMX adapter, confirm Live mode,
and verify the fixture universe/address patch.

## Development maintenance

Whenever a feature changes, update the relevant sections above in the same
commit. At minimum update:

- Contents if a new top-level section is added.
- Operator features for new controls and workflows.
- Feature impact for behavior changes.
- Feedback and learning when labels, weights, decay, confidence, or storage
  change.
- Glossary when new analysis, DMX, spatial, or musical language is introduced.
- Troubleshooting when a new failure mode or recovery procedure is found.
- `CHANGELOG.md` for every pushed progress backup, including the version, date,
  operator-visible changes, limitations, and complete test result.
- `docs/backup-and-restore.md` whenever installation, state layout, research
  provisioning, authentication, or hardware recovery changes.

The private GitHub repository backs up deterministic code and history. Runtime
databases, learned preferences, recordings, credentials, provider tokens,
downloaded research assets, and environments remain ignored and require a
separate encrypted owner-controlled backup. Follow the backup and restore
runbook before treating any Git checkout as operational.

## Glossary

### Audio and music analysis

- **Audio input / line input**: The electrical audio signal entering the Lumen
  PC. It is the timing source for lighting.
- **Amplitude**: Signal strength. Lumen uses it as part of loudness and energy.
- **Audio frame**: One simultaneous sample from every captured channel. It is
  the exact timing unit used to align recordings, features, and feedback.
- **Beat**: A detected repeating pulse in the music.
- **Beat confidence**: How certain the tempo tracker is that a beat estimate is
  reliable.
- **Beat phase**: Position within the current beat, expressed from 0 to 1.
- **Beat pulse**: Short-lived accent value generated around a detected beat.
- **BPM**: Beats per minute, the estimated tempo.
- **Energy**: Perceived musical intensity derived from loudness and spectral
  features.
- **Frequency band**: A portion of the spectrum, such as low, mid, or high.
- **High energy**: Treble-weighted spectral activity.
- **Low energy**: Bass-weighted spectral activity.
- **Loudness**: Normalized perceived signal level.
- **Mid energy**: Mid-frequency spectral activity.
- **Novelty**: How different the current audio frame is from recent frames;
  useful for transitions and changes.
- **Onset**: The beginning of a musical event such as a hit, note, or transient.
- **Onset strength**: Estimated force of that event.
- **PCM16**: Uncompressed audio represented as signed 16-bit samples. Lumen
  records the original stereo line input in this form inside WAV files.
- **Section**: A coarse musical region such as intro, verse, build, drop, or
  release.
- **Spectral analysis**: Measuring how audio energy is distributed across
  frequencies.
- **Tempo tracker**: Component estimating BPM and beat/bar phase.
- **Tension**: Expressive value representing pressure, contrast, or urgency.
- **Waveform**: Signal amplitude plotted over time.

### Expression and behavior

- **Expression**: Lumen's normalized state: energy, tension, motion, intimacy,
  and confidence.
- **Gesture**: Named expressive behavior such as breathe, converge, expand, or
  release.
- **Intimacy**: Expressive tendency toward close, restrained focus.
- **Motion bias**: Feedback/control preference for more or less movement.
- **Intensity bias**: Feedback/control preference for brighter or dimmer output.
- **Influence**: Blend between automatic audio interpretation and operator
  preferences.
- **Performance decision**: The explainable lighting choice produced from an
  observation and controls.
- **Routine**: A named phrase-level performance motif adapted to the installed
  rig rather than replaying fixed DMX.

### Lighting and DMX

- **Address**: Starting DMX channel of a fixture.
- **Beam**: The visible projected light direction from a fixture.
- **Calibration envelope**: Useful pan/tilt limits inside the fixture's total
  possible range.
- **Channel**: One 8-bit DMX control slot from 0 to 255.
- **DMX**: Digital Multiplex lighting-control protocol.
- **Fixture**: A light or effect device patched into the rig.
- **Fixture group**: Multiple fixtures controlled as one feedback target; Lumen
  currently exposes the movers group.
- **Gobo**: Pattern inserted into a light beam; profile-dependent.
- **Home position**: Saved pan/tilt position used as a fixture reference or
  parked position.
- **Master dimmer**: Overall brightness channel for a fixture.
- **Mover / moving head**: Fixture with motorized pan and tilt.
- **Pan**: Horizontal rotation.
- **Pan sweep**: A deliberate back-and-forth horizontal movement.
- **Profile**: Lumen's description of a fixture's channels and capabilities.
- **Resolver**: Converts an expressive room target into calibrated fixture
  angles and DMX values.
- **Strobe**: Rapid modulation of light output.
- **Tilt**: Vertical rotation.
- **Universe**: A DMX collection of up to 512 channels.

### Motion vocabulary

- **Alternation**: Fixtures or emitters take turns on successive beats.
- **Circle**: Pan and tilt vary as coordinated sine/cosine motion.
- **Converge**: Fixtures aim toward a narrower shared region.
- **Expand**: Fixtures widen the spatial focus.
- **Figure eight**: Pan and tilt use different harmonics to trace a crossing
  path.
- **Strobe**: High-energy rapid modulation of light output; disabled for soft
  passages or by explicit feedback.
- **Nod**: Short tilt accent, usually synchronized to a beat.
- **Phase offset**: Timing difference between fixtures so they do not duplicate
  one another exactly.
- **Release**: An expressive opening or outward accent after tension.
- **Sweep**: Continuous travel across a range and back.
- **Target**: A point in the room toward which a fixture is resolved.

### Software and operation

- **Activation gate**: Held-out performance requirements a newly trained
  candidate must satisfy before it may replace the model used by Live.
- **Boundary head**: The student model output trained to recognize that a new
  musical section has begun, independently of naming that section.
- **Causal model**: A model that uses only the present and past, making it
  suitable for live use without looking ahead in the song.
- **Choreography sequence**: An ordered set of semantic fixture actions with
  beat start times and durations.
- **Content role**: Independent label for vocal, instrumental, solo, or
  transitional material.
- **Dataset manifest**: Machine-readable index connecting audio files, timing,
  semantic frames, and human labels.
- **Energy form**: Independent structural label such as build, release,
  breakdown, groove, or silence.
- **Functional form**: Independent structural label such as intro, verse,
  chorus, bridge, or outro.
- **Ground truth**: A target known or intended to be correct. Current heuristic
  lighting choices are explicitly not treated as ground truth.
- **Held-out song**: A complete song group excluded from training and used only
  to measure whether a model generalizes beyond the songs it learned from.
- **JSONL**: A text format containing one JSON record per line, suitable for
  large streaming datasets.
- **Confidence decay**: Reduction of old feedback influence over time.
- **Data leakage**: An invalid evaluation in which closely related or adjacent
  examples appear in both training and test sets.
- **Feedback scope**: Overall, fixture-specific, or group-specific target.
- **Memory**: Local SQLite store for song identity, decisions, feedback, and
  routines.
- **Offline teacher**: A large research model that labels completed recordings
  outside the live DMX timing loop.
- **Provenance**: The dataset, model, operator, version, and confidence
  responsible for a label.
- **Student model**: The small CPU-capable causal model trained from normalized
  ground truth and offline teacher predictions.
- **Semantic frame**: A timestamped model record containing analyzed music,
  expression, routine, fixture state, media context, and controls.
- **Supervision**: Human-provided information used to teach a model, such as
  feedback or a selected preferred action.
- **Training segment**: One one-minute WAV portion of a longer listening
  session, indexed by its starting audio frame.
- **SHA-256**: File checksum saved for every finalized WAV segment so corruption
  or accidental changes can be detected.
- **Remote page**: Phone/tablet-friendly operator interface at `/remote`.
- **Rehearsal**: Isolated audition of one lighting routine against a generated
  beat clock; it does not train until Use this routine here is selected.
- **Routine tour**: Rehearsal playback that advances through the authored
  movement vocabulary every eight beats.
- **Preview output**: Virtual DMX rehearsal that does not open the physical DMX
  adapter.
- **Motion Studio**: Rehearsal editor for cycle length, calibrated path size,
  center, fixture relationship, direction, and compound-fixture travel.
- **Fixture relationship**: Exact phase/direction arrangement among movers,
  such as synchronized, opposed, mirrored, chased, or counter-direction.
- **Runtime**: The live loop connecting observation, decision, targeting, and
  DMX output.
- **Virtual DMX**: Test output used by Demo mode instead of the physical adapter.

### Recent implementation notes

- Live calibration starts at each mover's actual saved home, keeps it visibly
  lit, and captures independent Left/Home/Right and High/Home/Low room points.
  It derives raw endpoints, angles, offsets, and reversed-axis direction, then
  pauses and resumes the engine around a validated rig save.
- Fixture strobe requests now sustain the characterized hardware rate channel;
  automatic center beat bursts occupy a visible part of the beat instead of a
  single analysis frame.
- Generated live choreography uses the saved Party Parrot-style calibration
  envelope as its software-defined pan/tilt range; there are no additional
  hidden runtime clamps.
- Generated paths are expressed as physical Left→Right and Low→High motion.
  Each mover's captured axis direction converts that path to its own DMX order,
  so a reversed channel-31 tilt does not perform the vertical path upside down.
- Spotify console responses tolerate temporary `/me` rate limits and can show
  the last cached player state.
- The 3D rig camera uses normal drag for pan, Ctrl-drag for orbit rotation, and
  the wheel for zoom.
- Monitor and Live modes collect local PCM training sessions, ten-Hz semantic
  frames, exact-frame feedback links, complete fixture DMX, and checksummed
  model-export manifests. Demo is excluded.
- Feedback targeting no longer lists each mover separately; both movers are
  taught as one group while the center multi-effect remains individually
  selectable.
