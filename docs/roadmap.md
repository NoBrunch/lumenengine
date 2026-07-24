# Roadmap

## Milestone 0 — foundation (implemented)

- Standalone local repository
- Core domain and coordinate system
- Spatial target solver
- Expression baseline
- Private song memory
- Virtual DMX
- Audio and Spotify adapters
- FT232R/Open-DMX transport from Party Parrot
- Party Parrot active-show importer and fixture registry
- Stable Party Parrot tempo tracker
- Tests and diagnostics

## Milestone 1 — measure the real system

- Identify the ALSA line-in device and stable device name
- Confirm the imported active-show fixture inventory against the room
- Fill exact channel semantics for archived-show fixture profiles as needed
- Confirm imported mounting positions and housing rotations
- Calibrate pan/tilt zero, direction, range, and latency
- Add a calibration UI that never requires hand-editing JSON

## Milestone 2 — dependable live perception

- Improve beat/downbeat tracking
- Add bar and phrase state
- Track spectral flux and rhythmic density
- Detect silence, restart, pause, and source loss
- Record compact feature timelines, not raw audio by default
- Reconcile Spotify progress with the sample clock

## Milestone 3 — expression laboratory

- Local web dashboard
- 3D room and predicted beam visualization
- Plain-language feedback controls
- Moment markers and section correction
- Semantic routine editor
- Explanation and confidence history

## Milestone 4 — dependable direct operation

- Reconnect behavior after unplug/replug
- Dedicated fixed-rate output process if thread isolation proves insufficient
- Operator blackout and channel inspection
- Fixture-specific motion timing
- Long-duration soak testing

## Milestone 5 — personal learning

- Preference aggregation from corrections
- Per-song and per-artist semantic memory
- Related-recording handling
- Learned gesture ranking
- Small CPU-optimized temporal model if it outperforms the authored baseline
