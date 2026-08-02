# Lumen Engine progress log

This file records restorable Lumen releases. Keep the newest entry honest: note
operator-visible changes, migrations, known limitations, and the exact test
result. Git commit history remains the detailed engineering log.

## Unreleased

- Add new work here while it is in progress.
- Before pushing a backup, move completed items into a dated version entry,
  run the complete test suite, and record the result.

## 0.6.0 - 2026-08-01

Last known-good code snapshot for the initial private GitHub backup.

### Added

- Lossless, sample-aligned local training capture with checksummed WAV segments,
  semantic and DMX context, feedback links, quotas, and JSONL export.
- Normalized musical-structure datasets and independent functional, energy,
  content, and boundary timelines.
- Isolated, CPU-bounded EDMFormer and SongFormer offline teacher workflows with
  pinned sources, models, dependency locks, provenance, and resumable jobs.
- A small causal student-model training and held-out activation workflow.
- Beat-addressed choreography sequence learning, phrase-boundary feedback
  application, and exact reusable mover paths in Motion Studio.
- Operator views for research readiness, batch progress, model status, training
  health, learned choreography, and motion editing.
- Private backup and recovery instructions.

### Changed

- Live interpretation, silence handling, beat/phrase timing, fixture motion,
  preference learning, and offline analysis are integrated through the local
  memory database.
- Research assets and large model environments live under ignored `state/` and
  are never imported by Lumen's dependency-light live process.
- Ignore rules explicitly exclude runtime state, databases, environments,
  provider tokens, credentials, and common private-key files from Git.

### Verification

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`
- 166 tests passed on 2026-08-01.
- Tests and Demo use virtual output; this verification did not open the USB DMX
  adapter or validate a physical fixture.

### Restore point

- Release tag: `v0.6.0`
- Branch: `main`
- Runtime databases, learned preferences, recordings, model files, provider
  tokens, and research environments are deliberately not contained in this
  GitHub snapshot. Restore them only from the separate encrypted local-state
  backup described in `docs/backup-and-restore.md`.

## 0.5.0 - 2026-07-30

- Stabilized live musical interpretation and phrase timing.
- Added beat-locked learned phrase routines and context-aware preference reset.
- Made physical-axis fixture paths respect each mover's captured direction.
- Sustained characterized hardware strobe commands and improved silence
  behavior, calibration, palette handling, and feedback learning.
