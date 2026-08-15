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
9. [Threadripper compute node](#threadripper-compute-node)
10. [Calibration](#calibration)
11. [Spotify](#spotify)
12. [3D room interaction](#3d-room-interaction)
13. [Troubleshooting](#troubleshooting)
14. [Development maintenance](#development-maintenance)
15. [Glossary](#glossary)

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

Audio Laboratory's **Interpretation path** is the primary explainability view.
It shows whether structure came from cached offline teachers, the validated
streaming student, or the live-analyzer fallback. It also shows the separately
leased Movers and Center plans, their active step, selection confidence and
source, whether feedback is waiting for the next phrase, listener/event
evidence, the choreography model revision, and the number of learned sequence
candidates. Cached labels include confidence and lookup time; this status does
not imply that metadata is driving the beat.

### Audio laboratory

The Audio laboratory provides proof that the PC is receiving audio rather than
merely running a visual demo. It displays the waveform, frequency bands, onset
strength, beat lock, packet counts, and input diagnostics.

The **24-second interpretation history fix** is the named repair that separates
fresh physical input measurements from the 30 Hz interpolated lighting-control
clock. The trace advances at its intended 10 Hz and carries the most recent
physical dBFS measurement between analyzer frames, recording whether each point
is fresh and its measurement age. Interpolated control frames are no longer
painted as `-120 dBFS`, so the graph cannot manufacture rhythmic input dropouts
between healthy ALSA packets. Demo and Rehearsal may still display expression
history, but mark physical input as unavailable rather than silent.

The **Rhythm lock** panel separates the published BPM from the clock source,
spectral candidate, spectral lock, and octave decision. Tempo search covers
72–200 BPM. When a slow autocorrelation peak also has strong intervening
pulses, Lumen can promote the supported double-tempo interpretation (for
example, 73.5 to 147 or 87 to 174) while leaving a genuinely sparse slow meter
unchanged. A quiet musical passage hides the public BPM but retains the private
song hypothesis for up to thirty seconds; Spotify track changes and seeks still
reset it immediately.

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

**Remember for** controls how far the teaching is allowed to generalize:

- **This song section** (default) affects only the stable cue in this recording.
- **This whole song** may affect other sections of the same identified song.
- **This artist** may affect other identified songs by the same artist.
- **General taste** may affect any song and should be chosen deliberately.

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

Each browser receives a private local participant ID. A listener may add a
display name of up to 32 characters; the name and ID are stored only in Lumen's
local database. Every submission also receives a client event ID. Retrying the
same request returns the original event instead of storing or learning it
again. Different phones can submit simultaneously: distinct listeners raise
agreement more strongly, while repeated calls from one listener add urgency.
The system serializes database writes but never interrupts an active lighting
phrase to process them.

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
4. Adjust tempo, movement size, intensity, palette, and strobe. **Pure Blue**
   is the default Rehearsal color so movement is shown by one stable,
   saturated beam. The generated
   clock is exact, so motion differences are not obscured by live beat analysis.
5. Use Previous/Next for individual routines or Tour to advance every eight
   beats.
6. Stop rehearsal before switching between Preview and Live output.

### Motion Studio

Motion Studio edits the actual routine generator while Rehearsal is running.
Its trajectory display uses horizontal position for normalized pan and vertical
position for normalized tilt. Cyan is the first mover; amber is the second.
The moving markers indicate the current point in the selected cycle.
The Movers trajectory and Center mechanical schematic remain visible together;
the Editing selector changes the controls being edited, not whether the other
fixture group's movement can be observed.

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
- **Center cycle and side-pod relationship**: independent center-fixture timing with
  synchronized, opposed, mirrored, quarter-cycle chase, or counter-direction
  side-pod relationships.
- **Center-carriage controls**: direction, musical-rate multiplier, travel, and
  starting phase for the 300-degree rotating center component.
- **Side-pod controls**: separate direction, musical-rate multiplier, travel,
  and starting phase for the two 180-degree rotating pods.
- **Center optical/effect controls**: center/pod emitter pattern, color
  behavior, pod laser mode and level, segmented-ring program and rate,
  characterized hardware strobe, fixture intensity, and beat-blackout accent.

The velocity indicator compares the requested trajectory at the current
rehearsal BPM with each mover's saved pan/tilt speed model. **Velocity OK** means
the authored path fits; **Too fast** means the physical output would be
rate-limited and the visible shape could deform. Increase Cycle length or
reduce the affected travel until the indicator clears.

Rehearsal and Live use the same continuous mover path clock. Tempo estimates,
motion speed, and activity density change the path's future velocity; they do
not recalculate its accumulated position. This prevents a BPM correction or
feedback change from jumping a mover to another point in every routine. A
routine inside the saved calibration envelope is resolved directly and cannot
lose a DMX frame because an unrelated room target is unreachable.

Edits are applied to Virtual or Live rehearsal immediately and stored in the
private `state/motion-routines.json` file. Movers and Center have separate
tuning records for each semantic routine name. An edit therefore affects future
automatic performances only when the planner selects that routine for the
edited group. Rehearsal tempo, audition scale, intensity, strobe, palette, and
scope are temporary audition controls and do not rewrite the routine. **Reset
this group's routine** restores only the selected group/routine defaults. Scope
isolation explicitly zeros fixtures outside the selected mover/center group.

### Gesture Movement Editor

The Rehearsal-only Gesture Movement Editor associates expression gestures such
as Breathe, Sweep, Pulse, and Release with one or more authored movement
routines. **Add selected movement** associates the routine currently selected
in Movement Lab. **Edit selected movement** opens that routine in Motion Studio,
where Movers and Center retain separate characteristics. **Save associations**
persists the map locally and updates a running engine without restarting it.

These associations constrain Lumen's generated choreography candidates when it
calls that gesture. An exact song-timeline sequence remains authoritative, so a
general gesture association cannot erase a cue deliberately authored for one
recording and position.

Auditioning, adjusting, or touring routines does **not** teach the model. A
routine name is also not permanently associated with a section: the same
figure eight may suit a breakdown at low speed and brightness or a drop with
greater travel, activity, and intensity. **Live teaching reference** therefore
shows the identified title/artist, a large playback marker, Lumen's structural
source, energy/rhythm/change evidence, and the actual effective Movers and
Center outputs. **Load this cue into editor** copies the complete observed cue
as an editable draft. It does not teach anything until the owner reviews and
saves that draft.

This follows the common console split between programming and playback: motion
effects have parameters such as rate, size, direction/phase, scope, color, beam,
and entry/exit behavior; a cue or timeline records where the resulting look is
used. Lumen exposes separate mover and center implementations of its six
authored semantic routine families.

### Song Timeline and Sequence Editor

The desktop editor builds an ordered phrase for Movers, Center, or Whole rig.
The visible step controls set order/start beat, duration, routine, intensity,
brightness, motion speed, travel, activity density, beat sync, palette, and
strobe. Each saved step also carries entry/exit semantics; the
current editor supplies `phrase_boundary` and `resolve` defaults. **Save
complete cue sequence at this position** stores the semantic sequence and a placement at the current
identified-song position. An optional silence, intro, groove, breakdown,
build, drop, or outro state records the intended context alongside that time
placement. The cached teacher summary shows the normalized functional,
energy, and content axes available at that position.

Open **Analyzed song database & timeline review** to browse retained recordings
without playing them. Its visible table shows title, artist, duration, teacher,
timeline-review state, active-model training status, split, capture status, and
analysis age. Search and status filters narrow the table; clicking a row opens
the song's large color-coded section bar and detailed timeline below. The
editor spans the rehearsal workspace, gives the time column enough width for
both endpoints, and scrolls horizontally when a complete teacher record is
wider than the panel. Each
timeline entry names its teacher/version provenance,
normalization version, model confidence (or **unscored**), normalized axes,
segment boundaries, and the original raw teacher label. **Approve** makes an
otherwise unscored timeline authoritative for this recording through a
separate operator-trust field; it does not rewrite the model probability.
**Reject** excludes it from recall and automatic student targets. After either
decision Lumen removes that item from the active review view and opens the next
unreviewed timeline for that song, then the next song. **Reopen review** returns
a completed decision to the queue. **Correct labels** saves a new linked
operator timeline, including any adjusted boundaries, while retaining the
complete teacher original for audit and later training.

Saved song sequences can be loaded for editing, removed from the song timeline,
or deleted. Every edit and soft deletion creates a local revision snapshot;
**Undo last edit/delete** restores the preceding sequence or placement state.
The stored material is semantic fixture intent, not raw DMX, so Live resolves it
through current fixture characterization and calibration.

The phone/tablet **Teach a specific song action** drawer is the compact form of
the same workflow. A listener chooses Movers, Center, or Whole rig; a routine;
four, eight, sixteen, or thirty-two beats; and intensity, then presses **Use
this here**. Desktop and mobile submissions carry their participant identity
and are recalled only at a future phrase boundary.

### Memory

Memory stores recognized recordings, song identity, timing positions, feedback,
decisions, analyses, and routines in the local SQLite database. Recent
feedback is visible in the Memory page. Each recent feedback item has an
**Undo** action.

## Feature impact

### Audio response

Musical energy controls brightness and the size of movement envelopes. The
current expression value is an attack/release-smoothed weighted blend of 62%
log-mapped line level, 14% onset strength, 8% bass share, and 16% beat pulse,
followed by the operator intensity bias. It is not currently normalized from
the quietest to loudest point of each complete song. Rhythm density, spectral
brightness, harmony, and arrangement trajectories currently affect structural
state and boundary detection but do not all directly raise the displayed
energy percentage. Consequently a genuinely intense sustained passage may
read in the 80s; 100% requires all weighted evidence to peak together. Soft
sections reduce activity. Strong onsets and beat pulses provide accents.

### Silence behavior

When physical silence is confirmed for 550 milliseconds, movers hold their last
position and the center effect parks immediately. The center clears strobe,
publishes zero effective activity, centers its body/arms, and sends the slowest
characterized body-motor command. It no longer adds a second motor-tail delay
after the authoritative silence decision.

### Mover choreography

Movers use phase-offset patterns so the fixtures do not repeat the same motion
at the same time. Available motion families include figure-eight paths, circles,
pan sweeps, tilt/nod patterns, beat alternation, convergence, expansion, and
release gestures.

### Strobe behavior

Moving-head strobe remains off unless rehearsal or a duration-bounded authored
choreography cue explicitly permits it. Positive strobe feedback may tune the
enable/rate characteristics of a later eligible cue, but cannot authorize
hardware strobe by itself. The center fixture may add a short automatic burst
only inside its high-energy, section, beat-confidence, and beat-phase gates. A
legal strobe request holds the fixture's characterized strobe-rate channel for
the cue interval; Lumen does not simulate it by rapidly changing color or by
sending a single-frame dimmer flicker. **No strobes** suppresses automatic
bursts and learned strobe preference.

### Center multi-effect

The center fixture is mounted upside-down from the ceiling. It has a fixed
control base and a hanging rectangular center tower characterized for 300
degrees of rotation. The tower carries a floor-facing RGBW emitter behind a
hemispherical scatter lens, surrounded by a segmented RGBW ring. Two box-shaped
side pods travel with the tower; each is characterized for 180 degrees of
independent rotation and contains a downward-facing RGBW beam plus red/green
lasers. Lumen changes their
motor speed and optical patterns with musical energy, reduces activity during
soft passages, and parks them during sustained silence.

### Color Studio and color latching

Color Studio exists only in Rehearsal. Its Paint-style hue/saturation wheel and
brightness bar produce a named hexadecimal solid color. Saved solids can be
selected directly during fixture calibration or Rehearsal. Custom palette
families are optional collections of those colors for Live development; they
are not required for a fixture test. Automatic is the default for Live
Performance, not Rehearsal.

During Live, Lumen latches one resolved RGB color per fixture lane for the
active choreography lease. The lease normally lasts 16 beats or longer. Energy
and intensity may still modulate brightness, but they do not silently change
the selected hue. A confirmed musical boundary, a new explicit cue, or a new
song can replace the latch. This makes a solid beam useful for judging pan,
tilt, arm travel, and fixture direction.

### Center motion preview

Motion Studio's Center multi-effect view models the ceiling-mounted fixed base,
300-degree hanging rectangular tower, floor-facing hemispherical scatter lens,
segmented ring, and two tower-mounted 180-degree RGBW/laser pods. The tower and pods follow the
same routine-coordinate calculation used by fixture output. Mechanical ranges
come from the characterized fixture profile rather than drawing constants.
This graphic exists only in Rehearsal; Room & Rig does not display it.

## Feedback and learning

Feedback is both stored and used.

Each feedback event has a label, value, timestamp, song, playback position,
active gesture, inferred section, energy, motion, tension, confidence, BPM,
scope, optional fixture/group target, listening-session identity, participant
identity/name, and an idempotent client event ID. The label maps onto literal,
independent characteristics: motion speed, travel size, activity density,
brightness, palette, strobe permission, strobe rate, beat sync, and cue timing.
For example, **Too busy** reduces activity density without slowing or shrinking
the motion, while **Timing was right** reinforces cue timing without secretly
changing brightness or routine choice.

### Multiple listeners and agreement

Lumen accepts concurrent feedback from multiple desktop, phone, and tablet
browsers. Recent matching calls are separated into distinct participants and
repeat calls. The first call begins with moderate urgency; additional distinct
listeners contribute more agreement than repeated taps from the same listener,
while those repeated taps still communicate urgency. HTTP retries with the same
participant/session/client event key return the existing database record and do
not train the choreography model twice.

Feedback is recorded immediately, but matching presses in one five-second
window, musical section, active routine, active sequence, and boundary become
one replaceable preference-model event per fixture lane. Overall feedback
therefore teaches Movers and Center against their own actual performances,
not a shared compatibility routine. Repeated presses increase
that event's urgency logarithmically; they do not create a growing stack of
nearly identical training examples. The currently running Movers and Center
leases remain intact. This makes a crowded listening session additive rather
than interrupt-driven.

Musical-context buttons use a separate consensus path. They always describe
the whole song, even if a legacy phone sends the currently selected lighting
group. Calls within three seconds are clustered by musical axis. Each browser
participant receives one vote in that cluster; rapid repeat presses from the
same participant are collapsed rather than impersonating additional people.
Matching listeners raise confidence. A tied conflict remains visible evidence
but is not allowed to become structural authority.

After the listening run, Lumen rebuilds a sparse song correction timeline.
State calls such as **Build**, **Drop**, or **Breakdown** override only the
remainder of the current EDMFormer section. Event calls such as **Drop onset**
or **Build start** also supervise a 1.5-second boundary window. Untouched axes
remain null in the correction view, so EDMFormer continues to supply them.
The raw participant records are retained unchanged for audit.

Lumen aggregates this context at overall, fixture, song, artist, and song
section levels. The scalar fixture bias and the choreography ranker enforce the
same selected lifetime. Cue-local learning is stored only under its stable
recording/section token; it does not also write hidden global routine or metric
weights. Positive feedback reinforces the gesture and routine that were
active. Negative feedback rejects the active choice. Directional controls such
as **Calm down**, **Pick it up**, and **More movement** teach movement or
intensity characteristics only; they do not secretly nominate a named routine.
A named alternative is learned only when the listener submits the explicit
**Preferred action** control.

Brightness, palette, and strobe calls follow the same rule. **Strobe**, **No
strobes**, **Brighter**, **Dimmer**, and palette-shift feedback score or modify
the future look; they never become complete choreography sequences and never
pin the routine that happened to be running when the call was made. Preference
files written before this separation are migrated by replaying their reversible
events without the obsolete characteristic-as-sequence candidates.

### Decay

Historical influence uses exponential recency decay with a 21-day time scale:

```text
recency weight = exp(-age_in_days / 21)
```

Older feedback gradually matters less without being abruptly discarded.

### Confidence

Agreement confidence increases with distinct listeners and repeated urgency:

```text
confidence = clamp(0.45 + 0.14 × additional_listeners
                   + 0.06 × repeated_taps, 0, 1)
```

The decayed weighted effect is multiplied by this confidence and clamped to the
engine's usable range. A single accidental or isolated opinion therefore has
limited long-term influence.

### Scope combination

Overall feedback applies to both permanent performance groups. **Movers**
expands into both moving heads; **Center** targets only the multi-effect
fixture. Individual movers are not teaching targets. The same group identities
are used by Rehearsal, Motion Studio, feedback, song placements, choreography
lanes, and explainability status.

### Undo

Use Memory → Recent teaching moments → Undo, or use **Undo last** on the remote
page for that browser's most recent contribution. Lumen deletes the event,
rebuilds the decayed/confidence-weighted profile, removes that event from the
event-backed choreography learner, replays the remaining learning events, and
replaces the running runtime's feedback profile. Literal characteristics such
as travel, speed, density, and brightness update without replacing the
currently leased routine; the fixture rate limiter keeps the adjustment
continuous. Learned routine ranking changes are considered at a natural
sequence boundary. Sequence-editor revisions and placement deletions have
their own history-backed undo and do not depend on feedback undo.

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
its complete authored duration and can be learned from an explicit preferred
action. Automatic planning will not begin more than two consecutive phrases
with the same routine when an alternative exists. Movers and Center run
independent sequence lanes: one may hold a multi-beat mover sweep while the
other runs a center arm chase or counter-rotation. Both leases advance from the
same audio-derived musical boundary clock, but their routine definitions,
steps, colors, motion, strobe, confidence, and replacement decisions remain
separate. Compound motion uses the detected bar phase when available, keeping
sweeps, arm chases, and body rotation on the same sample-clock tempo grid.

The Audio laboratory chart shows 24 seconds of physical input level and
audio-derived expression energy. Section labels use temporal hysteresis rather
than classifying each PCM packet independently. `drop_onset` is an instantaneous
transition event; builds, drops, grooves, and breakdowns are sustained states
that must persist before the label changes.

Lumen writes a compact performance trace twice per second. Trace schema v2
includes the raw audio observation, physical input measurements, resolved
observation, expression, gesture, routine, controls, song position, resolved
mover angles, and the effective source/confidence/provenance of every structure
axis. Cached teacher, streaming student, live fallback, and physical-silence
decisions are therefore distinguishable in every frame.

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
- **SongFormer**: the active functional/content teacher. Its functional song
  roles, content roles, and boundary evidence complement EDMFormer's techno
  energy labels. SongFormer energy labels do not override EDMFormer.

All imported labels are normalized onto independent axes:

- **Functional form**: intro, verse, pre-chorus, chorus, bridge, outro, and
  related song roles.
- **Energy form**: silence, intro, groove, breakdown, build, drop, and outro.
  Transitional events such as `drop_onset` are stored separately from these
  sustained states.
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
Versioned current-analyzer features regenerated from the coherent WAV
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
preparation in a background thread. This automatic step builds a compact index
for only the newly completed session: verified segments, recording boundaries,
and teacher eligibility. It does not regenerate historical frame,
student-sequence, or choreography-sequence exports. The full historical export
is built only when explicitly requested. Heavy teacher inference is represented
by durable SQLite jobs and executes outside the DMX loop. The Audio laboratory's
**Musical-structure research stack** is the normal operator workflow:

1. After Monitor or Live stops, wait while **Preparing the most recent
   capture** is shown. Lumen verifies continuity and identity before deciding
   which recordings are full-song structure candidates. Captured partial and
   unidentified fragments remain useful evidence, but are reported separately
   instead of being presented as completed songs. Source-audio gaps are scoped
   to the song whose sample range they overlap; a gap in one recording does not
   invalidate later complete songs from the same listening session. A rejected
   recording is still entered into the capture inventory with its exact reason,
   so **Analyze new recordings** reports retained partial or unidentified audio
   rather than incorrectly claiming that nothing was captured.
2. Press **Analyze new recordings**. Lumen rebuilds the verified manifest,
   queues both current EDMFormer and SongFormer work, and processes the durable
   queue as a resumable batch. Held-out songs are prioritized so honest validation becomes
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
   is shown as fully processed after its required current teacher jobs complete.
   Per-teacher progress and failures remain visible. A
   preliminary model may be trained once trusted completed runs contain at
   least two distinct training-song groups plus a separate held-out song;
   remaining teacher work can continue later.
4. Press **Train and validate**. Automatic training accepts only example files
   whose path, checksum, teacher-run ID, recording ID, and row count belong to
   completed teacher runs in the active Lumen database. Unowned smoke-test or
   stale JSONL files are ignored.

Automatic examples come from current-version EDMFormer and SongFormer
timelines. Before training, Lumen merges aligned ten-Hz frames by axis:
EDMFormer is authoritative for energy, SongFormer is authoritative for
functional/content labels, and boundary evidence is the union of both.
Retired, rejected, noncanonical, corrupt, or unowned artifacts remain durable
history but cannot enter student training or active recall. Every merged row
records its source timeline and teacher for each axis.

Before automatic training is queued, Lumen refreshes the participant-consensus
view and overlays its accepted sparse corrections on the corresponding
EDMFormer frames. The training manifest records the consensus revision and the
number of corrected rows by axis. A new correction revision makes an older
candidate and active student explicitly stale until retrained; this prevents a
model from appearing current after the owner has corrected its targets.

Recognized Spotify identity recalls the corrected timeline across later PCM
captures of the same song when their durations agree within three percent or
three seconds. This means a second play can use the first play's reviewed
structure without letting Spotify metadata drive beat timing. The physical
audio sample clock remains authoritative.

The selected energy state now directly shapes fixture output as well as routine
choice. Breakdown and outro lower movement speed, travel, activity density,
brightness, and palette-change rate; build develops those axes; drop permits
the broadest and most active output. The strength of this shaping follows
provenance and confidence: participant consensus and cached EDMFormer evidence
have more authority than the live fallback. Structure never invents a strobe;
an authored duration-bounded strobe step must also occur in an accepted build
or drop context (Motion Studio rehearsal is exempt so it can be auditioned).

Before training, Lumen re-analyzes each coherent WAV with the current audio
analyzer and caches versioned causal 10-Hz features under
`state/training/research/features`. This lets older recordings gain real
harmonic change, rhythm density, spectral trajectory, and arrangement-change
features introduced after they were captured. The cache is reused until the
feature contract changes, so retraining does not repeatedly scan the WAVs.

The streaming student retains causal summaries at 0.5, 2, 8, 30, and 60
seconds plus sample-clock elapsed-song context at 30, 60, 120, 240, and 480
seconds. Live prediction advances those memories from the audio sample-clock
timestamp rather than the approximately 23.44-Hz PCM packet arrival rate, so
the windows retain the same real-time meaning as the 10-Hz training examples.
A new Monitor or Live session and a new recognized song both reset the causal
memories, elapsed clock, and stable decoder before accepting new frames.

Training computes each song's causal contexts in chronological order once,
standardizes them using training-only statistics, then shuffles those fixed
contexts into mini-batches. Adam, gradient clipping, class-balanced independent
heads, validation early stopping, and best-checkpoint restoration prevent the
old behavior in which long final sections dominated online updates. Reported
loss is recomputed from the frozen restored model instead of measuring a model
while it is still changing inside a labeled passage.

The student has separate functional, energy, content, and boundary heads and is
evaluated on a song excluded from training. A newly trained file is saved first
as `lumen-structure-student.candidate.npz`. Each axis has its own held-out gate:
only approved axes are persisted as live-capable. Classification heads must
beat their held-out majority baseline by a nonzero margin. Boundary approval
requires useful precision as well as F1, preventing a transition detector that
fires constantly from entering Live. Automatic activation additionally
requires at least five independent test-song groups; a smaller test population
may be used for diagnostics but is too volatile to authorize Live. Energy must
also beat a balanced-accuracy gate calculated from the recall of every energy
class present in the test songs, so a model cannot pass merely by predicting a
dominant `drop` label. Energy is the only student
axis allowed to replace Live's section decision, so a proven energy head can be
activated while a failed functional head remains explicitly quarantined.
Unapproved functional/content predictions cannot enter choreography ranking,
and an unapproved boundary head cannot accelerate the stable decoder. The
candidate remains inspectable with exact per-axis reasons.

The combined teacher contract supplies all four student targets. EDMFormer
supplies energy and boundary supervision; SongFormer supplies functional,
content, and boundary supervision. Each head still qualifies independently,
so a failed functional head cannot contaminate a passed energy head and no
teacher is credited with an axis outside its assigned authority.

Boundary training uses a versioned 1.5-second causal target window immediately
after each teacher transition. Qualification does not demand exact 10-Hz frame
overlap: predicted transition clusters are collapsed into events and matched
one-to-one to teacher events within ±1.5 seconds, with a two-second refractory
period. The report retains frame metrics for diagnosis while activation uses
event precision and event F1.

The console describes this as an **unseen-song qualification test**. It shows
the held-out energy/content accuracy against each majority-baseline threshold,
the boundary precision and F1, and whether functional-section examples exist.
A rejected current candidate is distinct from a previous active artifact that
predates the full-song EDMFormer pipeline: the former is a completed model that
failed qualification, while the latter is an informational obsolete-model
notice rather than a load failure. The console does not prescribe repeating
Analyze and Train with unchanged inputs. Review or correct a held-out timeline
only when its labels are actually wrong, then retrain after trusted data or the
student implementation changes; a failed gate can also expose insufficient
song diversity or model generalization rather than operator annotation error.
An expandable unseen-song result list identifies every test song and reports
its energy accuracy, balanced energy accuracy, boundary-event F1, review state,
and example count. The analyzed-song database's **Split** column remains the
route to open that song's full timeline. Stable provider/song identity keeps
every repeated capture in one partition, and newly captured songs assigned to
`test` remain excluded from weight fitting even after their labels are reviewed
for correctness.

Candidate and active evaluation reports are separate. A rejected candidate
writes `lumen-structure-student.candidate.evaluation.json`. A current-gate
active model is preserved; an artifact approved by an obsolete gate is backed
up with a `pre-<gate-version>` name and replaced by the rejected candidate's
empty approved-axis artifact, so stale approval cannot control Live. An
activated candidate atomically replaces both the active model and its active
evaluation report, after retaining the previous pair for diagnosis.

Teacher segment transitions supervise the boundary head even when the teacher
does not publish a calibrated probability. The segment label remains unscored,
but the existence and timestamp of the normalized timeline transition are a
valid categorical training target. Derived student-example files carry their
own schema version and are rebuilt from the durable teacher timelines when
training requests a newer schema; the heavy teachers do not rerun.

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

EDMFormer analyzes each recording as one song-length structural sequence. For
the current library (all recordings are under the published 420-second limit),
MuQ and MusicFM extract consecutive 30-second local representations plus one
global representation covering the complete recording. EDMFormer fuses those
representations and decodes boundaries only once for the complete track. The
30-second pieces are never classified as independent timelines. Recordings
longer than 420 seconds are retained but explicitly rejected until Lumen has a
validated overlapping-long-context implementation that merges frame
probabilities before boundary decoding.

On the target CPU, MuQ and MusicFM use PyTorch's bounded
scaled-dot-product-attention kernel for their rotary conformer layers. This is
the same Q/K/V attention operation and preserves the complete global context;
it avoids retaining the eager implementation's full square attention
probability matrix when no caller requested that matrix. The inference
provenance records this adapter as `cpu_sdpa_v1`.

The adapter was validated on this PC against the same 221.55-second PCM: its
eight section labels and every millisecond boundary were identical to eager
attention, peak worker RSS fell from 8.71 to 6.91 GiB, and runtime changed by
1.3%. A 371.65-second capture that previously reached Lumen's 11 GiB cutoff
completed at 9.74 GiB and produced one contiguous full-song timeline.

Results created by the retired 30–60-second context adapter remain visible as
diagnostic history, but cannot drive Live, enter student training, satisfy the
Analyze queue, or be approved as current evidence. **Analyze new recordings**
reuses the already reconstructed local WAV files and queues full-song
replacements; no captured audio is deleted or needs to be recorded again.

Every teacher subprocess is supervised as a complete process group. Lumen
records its current and peak resident memory and stops it at the configured
offline limit before it can exhaust the lighting computer. The Audio
Laboratory reports current usage while analysis is running and presents a
memory-limit stop as an exact, retryable teacher failure rather than a generic
crash. This limit is an emergency cutoff, not a RAM allocation or throttle;
teachers use as much memory as their workload requires below it. A measured
legitimate need for more can use the `LUMEN_OFFLINE_MAX_RSS_GIB` setting rather
than changing live audio/DMX timing.

On this dedicated Lumen PC, `scripts/configure-lumen-appliance` provides a
reviewed, reversible workstation profile. Running the script's `apply` command
with `sudo` disables unrelated printing, modem, remote-desktop, AnyDesk, OLA,
search,
Evolution, update-notification, backup-notification, and virtual-machine guest
helpers; retains the browser, GNOME, networking, Bluetooth, audio, USB/DMX,
Avahi/Chromecast discovery, and development tools; adds an 8 GiB emergency
swap file; raises the user-session memory-pressure threshold from the desktop
default to 90% for 45 seconds; and sets Lumen's supervised offline-teacher
cutoff to 11 GiB. This does not make the hard disk fast and does not relax the
Live engine's timing separation. It gives full-song offline inference more
headroom while making the disposable teacher process the preferred kernel OOM
victim. Use `status` to inspect the profile and `sudo
./scripts/configure-lumen-appliance rollback` to restore the saved settings.
Apply and rollback require a reboot to complete the user-session policy change.
OLA is disabled because Lumen writes its FT232R/Open-DMX adapter directly with
native libftdi. OLA's USB detector is not part of the Lumen output path and can
otherwise repeatedly probe the same serial adapter and consume a full CPU core.

Offline jobs carry a worker identity, process ID, and heartbeat. If Lumen or the
desktop session ends during analysis, the next Lumen start—and every subsequent
Analyze request before it counts available work—returns the abandoned job to
the durable queue. A related unfinished teacher attempt is retained as failed
provenance; completed jobs and teacher results are never rewritten. Audio
Laboratory explicitly reports how many interrupted jobs were recovered, and
Analyze resumes them normally.

If a command-line research worker is already active, an open console reports
that external worker and its memory use, disables Live/Analyze/Train, and does
not launch a competing model process. Status polling recovers a worker that
dies after the console opens. When an external training worker successfully
activates a validated student artifact, the idle console loads it without
requiring an application restart.

Technical maintenance commands:

```text
lumen research-status
lumen research-import-annotations
lumen research-prepare-export <export-directory>
lumen research-worker --max-jobs 1
lumen research-train-student
```

Normal listening does not require these commands.

### Cached teacher recall for recognized songs

The Rehearsal page includes an **Analyzed song database & timeline review**
workspace above the timeline editor. It lists every recording for which Lumen
has retained a generated timeline, places **Needs review** recordings first,
and shows the title, artist, duration, teacher, timeline count/review state,
capture eligibility, active-model inclusion, split, and analysis age. Use
**Find a song**, the visible table, or the **Song to review** selector to inspect
a recording without playing it. The table defaults to all retained songs with
actionable work sorted first; **Show** can narrow it to needs-review, reviewed,
or diagnostic-only timelines.
**Select playing song** jumps back to the currently identified Spotify item.
The selected recording has direct Spotify Play/Pause, seek, and ten-second
transport controls. Its review surface correlates current SongFormer
function/content with EDMFormer energy in one table. Every displayed time uses
minute/second timecode. Bright timeline partitions can be dragged directly;
the operator can also add a partition at the Spotify playhead, merge either
neighbor, and undo or reset unsaved edits. Selecting a table time moves the
song playhead to that partition.

**Save & complete review** writes one immutable operator-composite timeline and
marks its eligible raw EDMFormer/SongFormer sources as reviewed through that
correction. It does not rewrite or falsely approve the model originals. Those
remain under **Original model evidence** with their raw labels, confidence, and
individual rejection history. Missing context cells are visible in the merged
table, and saving with missing function/energy context requires explicit
confirmation. One successful completion advances to the next song awaiting
review instead of requiring one correction pass per teacher.

When Spotify identifies a recording, Lumen resolves the provider/track identity
and duration to the matching stable recording version. Current normalized
EDMFormer timelines supply energy; current normalized SongFormer timelines
supply functional and content context. The combined draft clusters near-equal
model boundaries and retains the union of meaningful partitions. A newer raw
teacher can contribute candidate boundaries without overwriting an earlier
operator correction.
The selected context retains the exact recording and
timeline IDs, teacher version, raw/model confidence, separate operator trust,
boundary information, and segment provenance. Incomplete, rejected,
superseded, obsolete, or noncanonical teacher runs are not recalled directly.
Superseded raw teachers remain available as training-row provenance; the
operator composite overlays their corrected axes and boundaries when a new
student snapshot is prepared.

Teacher outputs without an upstream confidence score are stored honestly as
unscored rather than receiving an invented default. Unreviewed unscored output
cannot cross Live's normal model-confidence gate. The owner may approve the
unchanged timeline after reviewing its raw and normalized segments; Live then
records `model_confidence: 0` and `operator_trust: 1` as distinct facts. The
normalization version is recorded on jobs, timelines, preprocessing, and
training rows; older results remain preserved for audit but are excluded from
Live and new automatic training until reprocessed.

SQLite and JSON lookup runs in a background memory-context thread, normally no
more than twice per second. The audio/DMX thread atomically adopts the prepared
result; it never blocks on timeline storage. Cached energy can replace the live
section only when it clears its calibrated confidence gate. Legacy teacher
defaults that have not earned that confidence remain diagnostic context and do
not override a stronger approved student. A cached boundary is a short pulse at
the start of a changed segment, not a state held throughout that segment.
Cached functional/content
context can help rank an authored sequence, and matching semantic song
placements become high-priority candidates for the appropriate group lane.

Spotify position is deliberately coarse structural context. It never supplies
beat, downbeat, bar phase, or synchronous DMX timing. Those remain derived from
the line-input audio sample clock. Audio Laboratory reports the cache source,
axes, confidence, lookup duration, and the statement that line-in remains beat
authority. A detected seek within the same Spotify track invalidates the
student's causal memories, stable decoder, cached timeline selection, and
active choreography lease before the new position is interpreted.

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

The sequence preference model independently ranks ordered, beat-addressed plans
for Movers and Center rather than only choosing one shared gesture. A mover plan
can express “fan sweep for four beats, beat nod for two, then opposing chase for
two” while the center lane runs its own emitter/arm sequence. It considers song, artist,
functional/energy/content context, tempo, preferred actions, repeated feedback
urgency, age decay, and normalized DMX history.

Feedback never replaces either lane in the middle of its current musical
boundary. Repeated feedback increases evidence and urgency, but each active
sequence is independently leased until the next phrase or section boundary.
One lane can therefore replan without forcing the other to restart. This
prevents multiple phones or rapid taps from visibly interrupting a motion.

Preferred actions and characteristic corrections create reusable candidates
for later boundaries. Beat timing, step duration, fixture group, intensity,
palette, and strobe character are resolved when the learned step runs; they
are not merely stored as descriptive metadata. The interface labels and model
vocabulary share the same identifiers, including **No strobes**, **Not busy
enough**, **Too dim**, and faster/slower side-arm requests.

## Threadripper compute node

**Lumen Link** keeps this Ubuntu PC responsible for line-in timing, the
operator interface, feedback, canonical song memory, choreography, spatial
resolution, model activation, and DMX. A direct Gigabit Ethernet connection
sends eligible full-song EDMFormer and SongFormer jobs plus student training
and held-out evaluation to the 3970X/128 GiB Threadripper WSL node.

The transfer protocol uses a shared mode-600 secret for timestamped,
nonce-bound HMAC authentication. Audio objects are immutable and addressed by
SHA-256; uploads resume by byte offset. The worker accepts only fixed job types
and verifies the Lumen/model/source revisions, checksums, normalization,
feature, preprocessing, and student-format contracts appropriate to each job.
Results are signed in transit, identity-checked, and imported into canonical
local state only while Lumen is in standby. Teacher jobs return normalized
timelines; `student.train` returns a candidate model and held-out evaluation.
The compute node cannot activate that model. A disconnect leaves Live,
feedback, Spotify, audio timing, and DMX independent.

The standalone **Lumen Link** page shows the local/remote topology, connection
latency, queue, active job stage, transfer volume, worker resources, event
history, and supported versus gated capabilities. The phone/tablet interface
has a compact status card. **Test connection**, **Enable link**, **Pause
dispatch**, **Resume dispatch**, and **Disable link** change only the offline
coordinator. It keeps up to six teacher jobs supplied to the Threadripper;
student training runs alone because it consumes the complete trusted dataset.
Disabling returns
unstarted jobs to local eligibility while an active remote job drains to its
verified result.

The Threadripper worker also serves a read-only Lumen-style dashboard at
`http://127.0.0.1:8765/dashboard` when opened on the Threadripper itself. It shows heartbeat, worker uptime,
parallel-slot use, queue totals, memory headroom, stages, elapsed time, and
peak memory without exposing recording or song identities. The Windows setup
installs a desktop shortcut and a 30-second watchdog that starts the WSL user
service and repairs NAT forwarding after Windows or WSL restarts. The worker
service uses an always-restart policy; ordinary Lumen restarts reconnect
without operator commands.

Student preparation writes a compact, content-addressed numerical snapshot.
Rich per-frame teacher provenance stays in immutable teacher exports rather
than being repeated inside every training row. A new Train request cancels
older queued snapshots instead of overwriting their input path. This prevents
the multi-copy memory pressure and mechanical-disk swap storm observed on the
16 GiB Lumen PC.

An authenticated worker with a different code or asset contract is reported
as **INCOMPATIBLE**, not Offline. The dashboard shows both short revisions and
the exact WSL update/restart commands. **Enable link** remains interactive in
that state so it can return the specific mismatch instead of behaving like a
dead button; it will not enable dispatch until at least one current job
contract verifies.

The worker advertises `teacher.edmformer`, `teacher.songformer`, and
`student.train`; held-out evaluation is part of the training job rather than a
separate job type. The deployment procedure is
`docs/lumen-link-wsl-deployment.md`; the Threadripper-side Codex handoff is
`docs/lumen-link-codex-handoff.md`; the full authority boundary is
`docs/threadripper-compute-node.md`. Physical two-PC cable, restart, resume,
and local/remote comparison checks remain to be run after deployment and must
not be represented as already proven.

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

Below the ALSA hardware report, Audio Analysis repeats the live **Expressive
state** shown on the Performance Console: current gesture and explanation,
decision confidence, energy, tension, motion, and intimacy. Both displays are
driven by the same runtime decision; this second readout is a convenient way to
compare the physical input proof with Lumen's interpretation without changing
pages.

## Center fixture behavior

The center multi-effect uses saturated
foreground/background/contrast colors. Its ring walks through the fixture’s
built-in effect bank by musical bar, while the body and two arms rotate through
chase, opposing sweep, figure-eight, beat alternation, broad fan, and
counter-rotating-circle gestures. Energy controls the travel amount and speed,
so quiet sections settle without removing the routine variety.

## Desktop workspace sizing

Every desktop dashboard panel has a **↗** control. It floats the panel above the
dashboard so it can be dragged by its title bar and resized from its left,
right, top, bottom, or any corner. **↙** docks it back into the normal layout.
Floating position and size are saved locally by the browser, panels scroll
internally when their contents exceed the chosen size, and these view settings
do not change rig or DMX behavior.

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

During Live, current-track identity is polled independently from the audio path
at a two-second cadence. The active Spotify/remote page refreshes playback at a
five-second cadence and advances the displayed position locally between API
answers. Play/pause and seek paint optimistically, then reconcile with Spotify.
The independent player, device, and profile requests run concurrently; profile,
device, playlist, and selected-playlist item results are cached while playback
alone is refreshed. Concurrent browsers share the same serialized result. This
keeps phone sessions responsive without allowing Spotify network latency or
rate limits into the line-in/DMX timing path.

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
has declared the effective energy section **Silence**. Once declared, the
center must publish `Parked`, zero activity, strobe off, body/arms 128, and body
speed 255 without another fade delay.

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

The public Git repository backs up deterministic code and history. Runtime
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
- **Loudness**: Perceptually log-mapped physical RMS level. The mapping reserves
  1.0 for full-scale RMS instead of clipping ordinary mastered passages at
  1.0, preserving contrast between loud and exceptionally intense audio.
- **Mid energy**: Mid-frequency spectral activity.
- **Novelty**: How different the current audio frame is from recent frames;
  useful for transitions and changes.
- **Onset**: The beginning of a musical event such as a hit, note, or transient.
- **Onset strength**: Estimated force of that event.
- **PCM16**: Uncompressed audio represented as signed 16-bit samples. Lumen
  records the original stereo line input in this form inside WAV files.
- **Section**: A sustained techno region: silence, intro, groove, breakdown,
  build, drop, or outro. Instantaneous changes use separate transition events.
- **Spectral analysis**: Measuring how audio energy is distributed across
  frequencies.
- **Tempo tracker**: Component estimating BPM and beat/bar phase. Lumen combines
  spectral-onset autocorrelation with a transient-interval clock. A broad
  log-tempo prior resolves supported half-time and 3:2 ambiguities; it chooses
  a metrical family while the strongest physical-audio correlation chooses the
  exact BPM. Confirmed clocks hand off without repeatedly switching sources.
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
- **Fixture group**: Permanent semantic performance target. Lumen exposes
  Movers (both moving heads), Center (the multi-effect fixture), and Whole rig.
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
- **Compute node**: The Threadripper service that executes versioned offline
  analysis or training jobs and returns checksummed artifacts. It is not the
  live lighting authority and does not drive DMX.
- **Lumen Link**: The authenticated, resumable private-LAN coordinator between
  the Lumen PC and the Threadripper WSL compute node. It offloads EDMFormer,
  SongFormer, student training, and held-out evaluation/artifact generation
  while leaving all live authority, model activation, and canonical memory
  local.
- **Choreography sequence**: An ordered set of semantic fixture actions with
  group scope, beat start times, durations, intensity, palette, strobe, and
  entry/exit behavior.
- **Choreography placement**: A versioned link between a semantic sequence and
  a recognized song time or section.
- **Choreography lane**: One independently selected and leased sequence stream;
  Live has separate Movers and Center lanes on a shared musical clock.
- **Cached teacher recall**: Background lookup of eligible local teacher or
  operator-corrected timelines for an exact recognized recording and coarse
  playback position. It supplies structural context, never the beat clock.
- **Client event ID**: Browser-generated identifier that makes a feedback or
  teaching HTTP retry idempotent.
- **Content role**: Independent label for vocal, instrumental, solo, or
  transitional material.
- **Dataset manifest**: Machine-readable index connecting audio files, timing,
  semantic frames, and human labels.
- **Energy form**: Independent structural label such as build, drop,
  breakdown, groove, or silence. “Release” is ordinary operator vocabulary for
  a drop, while `drop` is the current stored techno label.
- **Functional form**: Independent structural label such as intro, verse,
  chorus, bridge, or outro.
- **Ground truth**: A target known or intended to be correct. Current heuristic
  lighting choices are explicitly not treated as ground truth.
- **Operator trust**: Explicit approval that lets a reviewed exact-recording
  timeline control recall without pretending the teacher supplied a calibrated
  probability.
- **Raw teacher label**: The teacher's original section text before Lumen maps
  it to normalized functional, energy, or content axes. Corrections never erase
  it.
- **Held-out song**: A complete song group excluded from training and used only
  to measure whether a model generalizes beyond the songs it learned from.
- **JSONL**: A text format containing one JSON record per line, suitable for
  large streaming datasets.
- **Confidence decay**: Reduction of old feedback influence over time.
- **Data leakage**: An invalid evaluation in which closely related or adjacent
  examples appear in both training and test sets.
- **Feedback scope**: Overall or permanent-group target in the current operator
  interface. Older fixture-specific records remain readable as legacy memory.
- **Feedback lifetime**: The selected generalization boundary for a teaching
  event: current song section, whole song, artist, or explicitly global taste.
- **Listener agreement**: Evidence from distinct participant IDs submitting a
  matching correction; repeated calls from one participant add urgency but do
  not impersonate additional listeners.
- **Structure consensus**: Derived song-wide instruction formed from nearby
  participant musical-context calls. It is versioned and replaceable; the raw
  calls remain the audit record.
- **Consensus anchor**: One accepted structural state or event at a song
  position after participant deduplication and agreement scoring.
- **Sparse correction timeline**: Operator-consensus timeline containing only
  corrected axes and cue windows. Uncorrected time/axes continue to come from
  the authoritative EDMFormer timeline.
- **Memory**: Local SQLite store for song identity, decisions, feedback, and
  routines.
- **RAM spool**: A bounded temporary queue in system memory that holds completed
  recording segments while a background worker transfers them to durable
  storage. It absorbs HDD latency without changing the sample timeline.
- **Pipeline deadline**: The wall-clock time available to analyze one captured
  audio packet and publish its resulting DMX state before the next packet is
  due. At 48 kHz with 2,048-frame packets, Lumen's deadline is about 42.7 ms.
- **Control update**: A newly resolved DMX universe delivered to the Open-DMX
  transmitter. It differs from a transmitted USB frame because the transmitter
  repeats the newest universe continuously between control updates.
- **Offline teacher**: A large research model that labels completed recordings
  outside the live DMX timing loop.
- **Provenance**: The dataset, model, operator, version, and confidence
  responsible for a label.
- **Participant ID**: Private per-browser identity stored locally so concurrent
  feedback can distinguish listener agreement from repeated taps. An optional
  participant name is limited to 32 characters.
- **Phrase lease**: Promise to keep one lane's active sequence stable until the
  next eligible phrase or section boundary; feedback can queue a replacement
  but cannot interrupt the lease.
- **Revision history**: Local snapshots used to undo semantic sequence and
  timeline-placement edits or soft deletions.
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
- **Motion Studio**: Rehearsal editor with separate Movers and Center tuning.
  Movers expose calibrated path geometry and relationships; Center exposes
  body, both arms, emitters, colors, laser, strip, hardware strobe, intensity,
  and blackout accent.
- **Gesture Movement Editor**: Rehearsal control that associates an expression
  gesture with allowed generated movement routines while preserving exact-song
  choreography authority.
- **24-second interpretation history fix**: Repair that prevents interpolated
  lighting-control frames from appearing as physical audio silence.
- **Fixture relationship**: Exact phase/direction arrangement among movers,
  such as synchronized, opposed, mirrored, chased, or counter-direction.
- **Runtime**: The live loop connecting observation, decision, targeting, and
  DMX output.
- **Virtual DMX**: Test output used by Demo mode instead of the physical adapter.

### Recent implementation notes

- ALSA is drained continuously by a dedicated bounded queue. Interface, SQLite,
  research-status, and system-probe work runs outside the live publication lock.
  The live queue holds approximately sixty seconds of PCM (roughly 23 MiB for
  48-kHz stereo PCM16), so filesystem and database pauses do not become missing
  audio. If an exceptional stall fills the entire minute, the oldest backlog is
  collapsed toward the newest quarter-buffer and every exact source-frame gap
  remains recorded.
- Live database connections disable SQLite's automatic WAL checkpoint. Trace
  rows are committed in worker batches, waveform thumbnails are omitted from
  durable semantic rows because the lossless WAV is authoritative, and WAL
  pages are checkpointed deliberately from standby/preparation paths.
- Post-capture teacher preparation uses a session-only identity index. The
  48-minute regression capture produces an approximately 81-KiB index in about
  2.2 seconds instead of an interrupted 380-MiB historical export. Incomplete
  captures stay indexed but are not duplicated into the teacher-audio cache.
  A legacy preparation marker without either a capture-inventory span or a
  teacher run is treated as unfinished and automatically revisited.
- Tempo publication now suppresses weak startup guesses, searches through 200
  BPM at quarter-BPM resolution using fractional audio-frame delays, separates
  a locked clock from challenger confidence, and exposes its spectral candidate
  and half/double-time ambiguity in Audio Laboratory. A supported double-tempo
  pulse from either the spectral or transient tracker can repair the other
  tracker's half-time lock; the transient tracker must sustain high confidence
  before it can make that correction. Unsupported octave, triplet, and
  3:4/4:3 rivals do not interrupt the clock. Quiet passages retain private tempo memory, and a
  non-close fallback or internal tracker retune must persist before replacing
  the established sample-clock grid. Published beat pulses come from one
  continuous phase clock, so dropout/reacquisition cannot masquerade as a bar.
- Musical structure is resolved once per frame and independently per axis.
  Physical silence is authoritative; otherwise confident cached teacher axes,
  approved student axes, and the live analyzer form the fallback chain.
- `opposing_chase` now exchanges calibrated mover position, saturated color,
  and illumination on the musical beat instead of sending both movers an
  effectively identical look.
- Live calibration starts at each mover's actual saved home, keeps it visibly
  lit, and captures independent Left/Home/Right and High/Home/Low room points.
  It derives raw endpoints, angles, offsets, and reversed-axis direction, then
  pauses and resumes the engine around a validated rig save.
- Fixture strobe requests now sustain the characterized hardware rate channel;
  automatic center beat bursts occupy a visible part of the beat instead of a
  single analysis frame.
- Generated live choreography uses the saved live calibration
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
- Motion routines now keep independent Movers and Center tuning. The center
  editor directly programs its characterized body, two arms, emitters, color
  relationship, laser, strip, strobe, intensity, and blackout accent.
- Live choreography now uses parallel Movers and Center phrase leases. Both
  share the audio musical clock but may select, hold, and replace different
  ordered routines.
- Recognized songs recall completed offline teacher structure and versioned
  semantic sequence placements in a background memory thread. The line-input
  sample clock remains authoritative for all synchronous output.
- Desktop and mobile song teaching store group-scoped ordered sequences with
  participant provenance. Concurrent feedback is idempotent per client event,
  distinguishes listener agreement from repeated urgency, and never interrupts
  the active routine for ordinary characteristic feedback. An explicit
  Preferred action may revise the next eligible phrase.
- Sequence and placement edits use revision snapshots, soft deletion, and
  history-backed undo. Audio Laboratory exposes cached structure provenance,
  per-lane active plans, confidence, pending feedback, and applied evidence.
- Student training now merges duplicate teacher frames, ignores unknown axes,
  regenerates versioned current-analyzer features from coherent WAVs,
  standardizes precomputed causal contexts, uses mini-batch Adam with
  best-validation restoration, and resets temporal state at each listening
  boundary. Each semantic head is activated independently: a failed energy
  head cannot discard a proven functional head, and Live checks the saved
  approved-axis list before consuming any prediction. A held-out axis must have
  enough examples, more than one target class, and a measurable improvement
  over the majority baseline; a single-class 100% score remains quarantined.
  Boundary examples derive from normalized teacher transitions and must pass
  both precision and F1 gates. Obsolete approvals are backed up and removed
  from Live authority during the next training pass.
- UI and command-line research workers honor the same durable lease. The UI
  reports an external worker, avoids competing Live/training processes,
  recovers dead leases while polling, and loads newly activated models while
  idle.
- Audio analysis and live choreography run as separate pipelines. A fixed
  30-Hz show clock coalesces any burst of completed analyzer frames to the
  newest musical state instead of rapidly replaying stale FIFO states. It
  advances beat and bar phase from authoritative physical sample timing but
  does not invent spectral or structural evidence. Source-capture age,
  processed-analysis age, show-clock interval/jitter, queue depth, and
  coalescing counts remain separately diagnosable.
- The browser's 10-Hz live-status loop does not wait for system scans, research
  summaries, teaching history, Spotify requests, or other database-heavy
  panels. Those refresh independently at slower rates, and unchanged status,
  event, and DMX cells are not rebuilt. The clock header reports live display
  latency and explicitly marks stale data instead of disguising a delayed UI
  as an audio interruption.
- The exact neural-readiness audit verifies and parses the trusted teacher
  corpus only from an offline operation or one background refresh. Its last
  verified result is cached under `state/training/research/cache`, so browser
  startup and Audio Laboratory polling do not repeatedly scan the growing
  example library. The interface shows when that background refresh is active;
  mature-library refreshes run in a low-CPU/low-I/O-priority local subprocess,
  preventing Python interpreter-lock contention and temporary audit memory from
  entering the long-lived console. Analyze and Train retain their exact
  server-side provenance checks.
- Only Audio Laboratory requests the rolling 240-point analysis history. Other
  desktop and mobile pages retain the same 10-Hz live cadence without decoding
  and transferring that unused scope payload on every poll.
- Multiple browsers share one serialized technical-status generation at the
  dashboard's 10-Hz cadence. A room full of phones therefore does not rebuild the 3D solution,
  analysis history, DMX heatmap, and training state once per network request.
  This display cache is isolated from the audio and control locks.
- The local HTTP console accepts a 128-connection pending burst, preventing the
  small standard-library default backlog from resetting simultaneous phone
  feedback when several listeners wake or submit at once.
- Live recording now stages complete WAV segments in a bounded RAM spool under
  `/dev/shm`. A separate persistence worker copies and checksums each segment
  to the training directory sequentially. The audio consumer has a roughly
  three-minute packet reserve, so a mechanical-disk latency spike cannot stop
  sample analysis or freeze a DMX look. Audio Laboratory reports queued RAM
  segments, pending bytes, persistence duration, and queue capacity.
- SQLite uses WAL once at database initialization, a 64 MiB worker cache,
  memory-backed temporary tables, and memory-mapped reads. Live no longer asks
  SQLite to renegotiate journal mode on every short-lived query. Filesystem
  free-space checks are throttled instead of running for every audio packet,
  memory recall polls are paced, and performance traces are committed in
  background batches.
- Audio diagnostics expose the measured per-packet pipeline budget and timing
  for analysis, structural resolution, runtime choreography, recorder submit,
  and publication. Physical Open-DMX diagnostics separately count USB frames,
  control updates, content changes, and the age of the newest control update;
  this distinguishes a healthy transmitter repeating a stale look from an
  actively changing show.
- Phrase-boundary feedback activation no longer recursively acquires its own
  runtime lock. Directional feedback can therefore become active at the next
  boundary without stopping the lighting thread. Preference rebuilds also
  obtain feedback and song artists in one joined database read rather than one
  song query per historical event, keeping simultaneous listener input off the
  critical audio-to-DMX path.
- The 10-Hz operator display cadence remains unchanged. Its feedback evidence
  payload is summarized to the exact listener/event totals actually displayed,
  reducing each local status response without removing visible information or
  slowing the live analysis feed.
- Repeated feedback and deletion revise only the identified reversible model
  event. Lumen subtracts that event's original weighted contribution instead
  of replaying the entire preference history, while producing the same learned
  weights, evidence, urgency, and phrase-boundary behavior.
- Live multi-listener bursts coalesce the full feedback-bias rebuild and the
  choreography-model snapshot on background workers. Raw feedback and its
  musical/performed context are still committed before acknowledgement; the
  latest combined characteristic influence is staged for the next phrase
  boundary.
  Explicitly preferred sequences remain in song choreography memory, while
  redundant copies of an already reversible performed sequence are not
  rewritten for every phone tap.
- The feedback-history panel counts decisions and feedback with indexed
  per-song subqueries instead of joining the two histories into a large
  intermediate table. This preserves the mobile/desktop response while
  avoiding repeated mechanical-disk work when several phones refresh together.
- Ordinary continuous mover routines smooth their dimmer envelope so the beat
  component of semantic brightness cannot resemble an accidental strobe.
  Deliberate `opposing_chase`, blackout, rehearsal, and authored hardware
  strobe behavior remain immediate. Mover strobe channels remain zero unless
  an eligible duration-bounded rehearsal or authored choreography cue permits
  them; positive feedback alone cannot manufacture such a cue.
- Mover paths use a continuous audio-derived phase clock. BPM, speed, and
  density changes alter velocity without rewriting accumulated phase;
  continuous routines no longer stop at the end of an activity-density window.
  Calibrated performance paths bypass unrelated spatial-target reachability,
  guaranteeing one resolved output for each mover on every active frame.
- Structure establishes the automatic speed, travel, density, and brightness
  baseline before literal operator feedback is applied. **More movement** can
  therefore expand the calibrated travel even in a trusted quiet section
  without requesting a new routine. `opposing_chase` keeps fluid opposing
  motion while its beam and color alternate on the authoritative beat clock.
- **24-second interpretation history fix:** the Audio Laboratory stores 240
  measurements on an absolute 10-Hz grid. Interpolated show ticks retain the
  latest measured dBFS and are marked with their physical-input age; they no
  longer draw false dropouts. The source/processed age history is sampled by
  the internal clock rather than browser requests.
- Live Performance defaults to Automatic color selection; Rehearsal defaults
  to a solid Pure Blue test beam. Color Studio provides a
  hue/saturation wheel, brightness, reusable solid colors, and named palette
  families. Color libraries replace atomically and invalidate active color
  latches, so a save cannot expose a partial palette or leave an old solid
  color running.
- Rehearsal Motion Studio displays only the fixture group being edited. Movers
  use their pan/tilt path plot; the center fixture uses three cycle traces for
  center rotation, Pod A tilt, and Pod B tilt with live position markers and
  physical degree ranges. This is an operator-facing motion representation,
  not a decorative fixture drawing or neural-model input. Gesture Movement
  Editor associates an
  allowed movement pool with each expression gesture; Lumen ranks complete,
  measure-aligned choices from that pool rather than treating unordered checks
  as a forced sequence. The center graphic is intentionally absent from Room &
  Rig.
- Stable cue colors are latched per fixture role, not shared as one hue across
  the entire rig. This preserves a solid beam while allowing deliberate mover
  exchanges and center ball/arm contrast.
- Feedback learning now receives normalized semantic samples decoded from the
  post-gate fixture frame. Movement, intensity, strobe, color change, and
  blackout evidence therefore describe what the rig actually emitted after
  choreography, feedback, smoothing, color latching, fixture encoding, and
  operator blackout.
- Spotify uses one refresh-locked token path and one authoritative background
  playback poll. Desktop and phone consoles consume the shared state; static
  profile/device/library data refreshes independently, and an operator search
  or navigation request queued behind an automatic refresh is replayed instead
  of being discarded.
