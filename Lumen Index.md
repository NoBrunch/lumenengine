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
4. [Feature impact](#feature-impact)
5. [Feedback and learning](#feedback-and-learning)
6. [Calibration](#calibration)
7. [Spotify](#spotify)
8. [3D room interaction](#3d-room-interaction)
9. [Troubleshooting](#troubleshooting)
10. [Development maintenance](#development-maintenance)
11. [Glossary](#glossary)

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
Audio capture and spectral analysis
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
- One fixture
- Both movers as a group

Current feedback vocabulary includes:

- Increase movement
- Decrease movement
- Too busy
- Not busy enough
- Calm down
- Pick it up
- Too bright
- Too dim
- Timing was right
- Perfect motion
- More like this
- Great transition
- I liked that
- Hold this idea
- Bad timing
- Free-form note

The phone interface uses touch-gesture filtering so a scroll or screen-wake
gesture cannot be interpreted as a feedback click.

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

### Move-then-flash

Moving heads can hide their beam while travelling and reveal it after settling
on a musical accent. This creates a movement-then-flash/strobe-like gesture:
the light appears when the head is not moving.

### Center multi-effect

The center fixture changes motor speed and pattern with musical energy. It uses
alternating emitters, circles, sweeps, nods, laser accents, strip programs,
and strobe accents. Its activity is reduced during soft passages and parked
during sustained silence.

## Feedback and learning

Feedback is both stored and used.

Each feedback event has a label, value, timestamp, song, playback position,
scope, and optional fixture/group target. The label maps to movement and
intensity deltas. For example, **Too busy** contributes a negative movement
bias, while **Timing was right** contributes a smaller positive movement and
intensity bias.

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
the decayed/confidence-weighted profile, and replaces the running runtime's
feedback profile.

### Current learning boundary

The preference model currently learns motion and intensity tendencies. It does
not yet automatically synthesize a complete semantic song routine, artist
profile, or genre profile from feedback. Those can be built on top of the
stored, weighted feedback history without changing the operator workflow.

## Calibration

Open Room & rig on the desktop PC and select a moving head.

### Live mover calibration

1. Select a mover in Fixture inventory.
2. Press **Start calibration**.
3. Use Pan jog, Tilt jog, and Jog speed to position the fixture.
4. Capture left boundary, home, and right boundary.
5. Adjust tilt boundaries as needed.
6. Press **Stop calibration**.
7. Save the selected fixture.

Calibration uses direct DMX-style pan and tilt values while the mover is in the
calibration state. The selected mover is held at low brightness during jogging.

### Envelope controls

- **Pan minimum/maximum**: useful horizontal travel limits.
- **Tilt minimum/maximum**: useful vertical travel limits.
- **Home pan/tilt DMX**: parked/home position values.
- **Wide**: broad starting envelope.
- **Center**: conservative centered envelope.
- **Loaded**: restore the saved values currently in the rig.

Calibration values are used by the spatial resolver and by generated motion.
They should describe the fixture's useful physical range, not blindly assume
the manufacturer's full theoretical range.

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

## Glossary

### Audio and music analysis

- **Audio input / line input**: The electrical audio signal entering the Lumen
  PC. It is the timing source for lighting.
- **Amplitude**: Signal strength. Lumen uses it as part of loudness and energy.
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
- **Routine**: A future-facing semantic performance description that can be
  adapted to the installed rig rather than replaying fixed DMX.

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
- **Move-then-flash**: Beam is hidden during travel and revealed after settling.
- **Nod**: Short tilt accent, usually synchronized to a beat.
- **Phase offset**: Timing difference between fixtures so they do not duplicate
  one another exactly.
- **Release**: An expressive opening or outward accent after tension.
- **Sweep**: Continuous travel across a range and back.
- **Target**: A point in the room toward which a fixture is resolved.

### Software and operation

- **Confidence decay**: Reduction of old feedback influence over time.
- **Feedback scope**: Overall, fixture-specific, or group-specific target.
- **Memory**: Local SQLite store for song identity, decisions, feedback, and
  routines.
- **Remote page**: Phone/tablet-friendly operator interface at `/remote`.
- **Runtime**: The live loop connecting observation, decision, targeting, and
  DMX output.
- **Virtual DMX**: Test output used by Demo mode instead of the physical adapter.

### Recent implementation notes

- Live calibration now holds a selected mover at visible low brightness while
  pan, tilt, and speed are jogged. Captured boundaries and home are saved back
  to the rig.
- Generated live choreography uses the full software-defined pan/tilt range;
  the spatial preview no longer acts as an invisible runtime clamp.
- Spotify console responses tolerate temporary `/me` rate limits and can show
  the last cached player state.
- The 3D rig camera uses normal drag for pan, Ctrl-drag for orbit rotation, and
  the wheel for zoom.
