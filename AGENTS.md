# Lumen Engine conventions

- Lumen is a private, local-first project. Never add a public remote or upload
  recordings, credentials, tokens, learned preferences, or runtime databases.
- Store all room and fixture distances in meters.
- The canonical room axes are right-handed and Z-up:
  - `x`: room left to room right
  - `y`: front of room to back of room
  - `z`: floor to ceiling
- Fixture housing rotations are intrinsic XYZ Euler rotations in degrees unless
  a field explicitly states another unit. Integration adapters must convert at
  their boundaries instead of changing the canonical room model.
- Physical output commands write directly. Automated tests and the plain `demo`
  command use fakes or virtual output and must never open the USB adapter.
- Fixture profiles and calibration values must not be assumed from a preview.
  Measure pan/tilt direction, zero, range, channel mapping, and physical latency
  before enabling a real fixture.
- Audio sample timing is authoritative for live musical events. Metadata
  providers may identify a recording and estimate playback position, but must
  not drive beat-synchronous output directly.
- Learning operates on semantic gestures and the owner's feedback. Geometry,
  fixture constraints, and output encoding remain deterministic.
- Keep the dependency-free core runnable on the target i5-8400/16 GiB Ubuntu
  computer. Add optional heavy dependencies only when their measured benefit
  justifies their runtime cost.
- Run `PYTHONPATH=src python3 -m unittest discover -s tests -v` after changes.
