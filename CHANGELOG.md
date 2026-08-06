# Lumen Engine progress log

This file records restorable Lumen releases. Keep the newest entry honest: note
operator-visible changes, migrations, known limitations, and the exact test
result. Git commit history remains the detailed engineering log.

## Unreleased

### Fixed

- Corrected exact-song teacher fusion to resolve authority from each timeline's
  own teacher instead of a stale database-loop value, and added strict
  SongFormer artifact ownership/checksum validation before a completed job can
  suppress reprocessing.
- Timeline review now advances after approve/reject, hides completed cards from
  active review, supports reopening a decision, and uses direct professional
  confirmation text.
- Separated an obsolete previous student artifact from a rejected current
  candidate in the research console. Expected model-version retirement is now
  an informational notice instead of a load error.
- Replaced the misleading instruction to repeat Analyze and Train with exact
  unseen-song metrics and guidance to review/correct held-out timelines, then
  retrain only after trusted inputs or the student implementation changes.
- Corrected active-artifact reporting so a disabled previous model remains
  visible without being presented as active.
- Replaced frame-exact boundary qualification with one-to-one musical event
  matching at a documented ±1.5-second tolerance while preserving frame
  metrics for diagnosis.

### Added

- Axis-specific EDMFormer/SongFormer fusion: EDMFormer owns techno energy,
  SongFormer owns functional/content form, both contribute boundaries, and
  every merged target retains teacher/timeline provenance. Analyze, readiness,
  exact-song recall, student training, evaluation, and staleness checks now use
  the combined contract.
- A full-width song timeline/sequence workspace with readable start/end times,
  plus float/dock controls that let every desktop panel move and resize from
  every edge or corner with browser-local layout persistence.
- Mirrored the Performance Console's live expressive state beneath the ALSA
  report in Audio Analysis, including gesture, decision reason, confidence,
  energy, tension, motion, and intimacy from the same runtime decision.
- Per-song validation/test results with recording identity, review state,
  energy accuracy, balanced energy accuracy, boundary-event F1, and example
  count in the research console.
- A five-independent-test-song activation minimum and explicit diagnostic-only
  state while the final test population is smaller.
- Per-class classification precision, recall, F1, support, macro F1, and
  balanced accuracy in student evaluation reports.
- A combined-teacher applicability contract with independently evaluated
  functional, energy, content, and boundary axes.
- Versioned 1.5-second causal transition targets and a new v3 activation gate,
  preventing models approved under older evaluation rules from controlling
  Live.

### Verification

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`
- 345 tests passed on 2026-08-05.
- `node --check web/app.js` and `git diff --check` passed.

## 0.7.2 - 2026-08-04

### Fixed

- Pinned the title bar, menu, toolbar, optional task strip, workspace, and
  status bar to named desktop-grid areas. Hiding the task strip can no longer
  auto-place the status bar into the flexible workspace row and enlarge the
  bottom status display.
- Added a regression contract covering both the hidden idle task strip and its
  visible active row.

### Verification

- `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src python3 -m unittest discover -s tests -v`
- 334 tests passed on 2026-08-04.
- `node --check web/app.js` and `git diff --check` passed.

## 0.7.1 - 2026-08-04

### Added

- A console-wide task strip with an activity spinner, current stage,
  human-readable detail, and elapsed time for capture preparation, EDMFormer
  analysis, student feature preparation, training/validation, engine startup
  and shutdown, hardware scans, and manifest generation.
- Visible confirmation animation for every accepted desktop and mobile button
  press. Wake/scroll-protected mobile feedback confirms only after the touch is
  accepted.
- A durable Threadripper compute-node architecture note covering the private
  Gigabit link, authority boundary, checksummed job contract, implementation
  stages, and acceptance tests.
- Static operator-interface contract tests for task status, press confirmation,
  typography, and the compute-node decision record.

### Changed

- Every fixed-pixel interface font is two pixels larger; the smallest fixed
  text increased from 7 px to 9 px and normal body text from 15 px to 17 px.
- Analyze explicitly names EDMFormer as the active resumable teacher. Analyze
  and Train buttons remain visibly busy for their server-side work and expose
  `aria-busy` state.

### Verification

- `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src python3 -m unittest discover -s tests -v`
- 333 tests passed on 2026-08-04.
- `node --check web/app.js` passed.
- The local console and research API were smoke-tested on `127.0.0.1:4099` and
  stopped afterward. No engine mode or physical DMX output was opened.

## 0.7.0 - 2026-08-04

### Added

- Full-song, resumable EDMFormer processing with 30-second local features,
  420-second global context, current-version provenance, and strict recording
  completeness checks.
- Participant-aware musical-structure consensus. Nearby calls are deduplicated
  per listener, conflicts remain non-authoritative, and accepted corrections
  become sparse song timelines over EDMFormer.
- Human-corrected student targets with correction-revision tracking, held-out
  per-axis activation gates, and automatic staleness when trusted teachers or
  operator consensus change.
- Cross-capture recall for the same provider track with duration protection.
- Independent Movers/Center choreography lanes, editable multi-step routines,
  Motion Studio group tuning, permanent group feedback, and multi-listener
  idempotency/urgency handling.
- RAM-spooled recording persistence, continuously drained PCM capture, a
  separate 30 Hz control clock, and detailed source/analysis/DMX latency
  diagnostics.

### Changed

- Corrected structure now drives routine selection plus movement speed, travel,
  activity density, brightness, palette family, palette-change rate, and
  duration-bounded strobe eligibility.
- Beat arbitration protects tempo continuity and metrical octave selection;
  section analysis uses causal arrangement, rhythm, harmony, spectral, and
  energy-change features instead of loudness alone.
- Database, system scans, feedback-model rebuilds, status serialization,
  recording writes, and offline jobs remain outside the live timing locks.
- The public repository contains deterministic source and documentation only;
  runtime state, recordings, learned preferences, credentials, tokens, models,
  research assets, and isolated environments remain ignored and local.

### Migration

- Existing musical-context records are interpreted song-wide even when older
  clients stored a lighting group. Raw reported scope remains available for
  audit. The current local database rebuild produced 114 accepted consensus
  cues across 30 songs without changing the raw annotation rows.
- An ignored, checksummed pre-migration database snapshot is stored under
  `state/backups/` on the Lumen PC.

### Verification

- `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src python3 -m unittest discover -s tests -v`
- 329 tests passed on 2026-08-04.
- The suite used virtual/fake DMX and did not open the physical USB adapter.
  Physical fixture behavior remains the owner's next Live test.

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
