# Lumen Engine progress log

This file records restorable Lumen releases. Keep the newest entry honest: note
operator-visible changes, migrations, known limitations, and the exact test
result. Git commit history remains the detailed engineering log.

## Unreleased

### Fixed

- Stopped queueing EDMFormer for songs beyond its validated 420-second
  whole-song context while retaining SongFormer coverage, classified earlier
  long-song rejections as known limitations, and separated unresolved Link
  failures from the durable historical failure total.
- Refresh the Audio Laboratory research truth immediately on entry, focus, and
  tab resume so a completed current candidate cannot leave an old stale-snapshot
  instruction displayed in a long-lived browser tab.
- Made Train and Validate publish snapshot materialization progress by song
  group, follow exclusive Threadripper feature/training/validation stages from
  the Audio Laboratory, and name local artifact verification, held-out
  revalidation, and atomic activation instead of appearing idle or stuck.
- Made student snapshot preparation single-flight across browsers so two
  simultaneous Train and Validate requests cannot build competing snapshots.
- Moved the returned student's memory-heavy local qualification into a bounded
  disposable process, preserving independent activation checks without leaving
  the 16 GiB live computer holding the expanded training corpus afterward.
- Made Lumen Link workload status describe durable local state instead of a
  capped 20-item receipt window, separated local/imported totals from the
  compute node's retained spool counts, and quarantined legacy partial teacher
  jobs so one invalid result cannot block later eligible work indefinitely.
- Published explicit captured-audio preparation stages, progress, timestamps,
  completion, and failures, and kept that status polling while preparation is
  active on any console page. This prevents a completed preparation from
  remaining displayed as an hours-long running task.
- Removed Spotify playback controls from the phone `/remote` interface and
  removed Rehearsal's palette selector and Color Studio pending a future color
  workflow redesign.
- Kept authenticated Link health independent from individual job-preparation
  failures, quarantined stale legacy student snapshots, and re-keyed remote
  transport jobs when a source upgrade changes the immutable manifest for an
  existing canonical job ID.
- Prevented remotely completed but not-yet-imported jobs from being repeatedly
  counted as fresh prefill submissions, so serial local result verification no
  longer collapses a six-slot teacher queue into single-job execution.
- Kept a six-job routed standby buffer replenished while another result is
  running or being imported, allowing newly free Threadripper slots to refill
  without waiting for the serial canonical-import path.
- Reworked research readiness to merge one recording at a time, coalesce all
  Link imports into one post-queue audit, and cap the disposable audit address
  space. On the installed library this reduced the measured peak from multiple
  4-6 GiB processes to 171 MiB and prevented readiness from blocking Link
  telemetry or imports.
- Parallelized student audio feature preparation across up to 24 clean worker
  processes, added recording-level progress, reused content-addressed feature
  caches across training runs, and enforced the compute-node memory ceiling
  across the complete process group.
- Replaced the Threadripper CPU meter's one-minute load approximation with
  sampled CPU utilization, shortened Link queue discovery and handoff delays,
  and raised the 48-thread WSL worker default from four to six concurrent
  eight-thread teacher jobs. Student training remains exclusive.
- Made the Windows dashboard shortcut use `127.0.0.1`; NAT setup now maintains
  a loopback port proxy alongside the dedicated two-PC Link address.
- Restored musical headroom to the live loudness axis. The former logarithmic
  curve reached 1.0 at ordinary mastered-program levels; the revised curve
  reserves 1.0 for full-scale RMS while physical clipping remains an
  independent PCM measurement.
- Made beat evidence independent of program loudness and strengthened
  half-time, double-time, and 3:2 tempo-family arbitration with spectral
  confirmation and stable source ownership.
- Removed the full teacher-corpus readiness audit from browser bootstrap and
  research polling. The verified result is now durable and cached; a mature
  cache refresh runs as a low-CPU/low-I/O-priority subprocess rather than
  competing with Live's interpreter or remaining in the console heap.
- Serialized and shared Spotify console work across browsers, reused stable
  profile/device/library data between playback refreshes, and increased the
  active player cadence without moving network work into Live timing.
- Scoped rolling analysis history to Audio Laboratory instead of transferring
  it to every page and every connected phone.
- Added the unused OLA daemon to the reversible appliance disable list after
  confirming its USB detector could spin on the FT232R adapter at one full CPU
  core even though Lumen uses native libftdi directly.
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

- Lumen Link: an HMAC-authenticated private-LAN coordinator and WSL compute
  service for remote EDMFormer, SongFormer, student training, and held-out
  evaluation/artifact return. Transfers are content-addressed, checksummed,
  resumable, and idempotent across retries; canonical timeline/model import
  remains local and waits until Live is stopped.
- A KDE-inspired Lumen Link dashboard with topology animation, connection and
  authentication state, queue/job progress, transfer volume, Threadripper
  CPU/RAM/disk telemetry, activity history, capability gates, and compact
  phone/tablet status.
- Read-only-by-default deployment tooling for the Lumen Ethernet interface,
  Windows mirrored/NAT WSL networking, mode-600 secret pairing, a persistent
  WSL user service, pinned Python environments, authenticated capability
  verification, and a step-by-step Codex handoff. The worker advertises all
  three implemented job types and returns candidates/evaluation without remote
  activation authority.
- Remote EDMFormer execution clamps the general 24-thread Threadripper request
  to the runner's validated eight-thread maximum; SongFormer and student
  training retain the wider worker ceiling.
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
- 388 tests passed on 2026-08-07; 42 focused Link, deployment, real student
  child-runner, and interface checks passed again after the final cross-review.
- `node --check web/app.js` and `git diff --check` passed.
- The installed 5-GiB database/corpus was smoke-tested locally: cold bootstrap
  fell from roughly 35 seconds to 0.19 seconds, status to 0.01 seconds, and
  cached research status to 0.17 seconds. The exact readiness refresh completed
  successfully in its low-priority subprocess. No engine mode or physical DMX
  output was opened, and the test console was stopped afterward.
- Lumen Link protocol, deployment, and interface tests use loopback/fakes. The
  physical Windows/WSL-to-Lumen cable, restart, resume, and local/remote output
  comparison remain operator acceptance work after both PCs are deployed.

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
