# Threadripper compute-node link

Status: approved architecture direction, 2026-08-04. This document records the
proposal discussed in the shared “Lumen Threadripper Architecture” conversation
and the refinements agreed after comparing it with Lumen 0.7.0.

Discussion reference:
<https://chatgpt.com/share/6a72ab1f-5cb4-83ea-a096-bf037105812f>

Deployment package:

- [Beginner WSL/Windows/Lumen setup](lumen-link-wsl-deployment.md)
- [Codex handoff on the Threadripper](lumen-link-codex-handoff.md)

Lumen Link v1 implements the authenticated transport and remote EDMFormer
executor. SongFormer, student training and held-out evaluation remain explicit
gated capabilities until their immutable result importers are implemented and
validated.

## Purpose

Connect the dedicated Ubuntu Lumen PC to the Windows Threadripper workstation
over their unused Gigabit Ethernet ports. The Threadripper will perform heavy
offline musical analysis, feature generation, model training, evaluation, and
simulation without placing that work on the computer responsible for live
audio timing and DMX.

This is an offline-compute extension, not a migration of the live engine.

```text
Lumen PC — live authority
├── physical line-in and authoritative sample clock
├── beat/onset tracking and live student inference
├── operator interface and feedback
├── canonical song memory and research queue
├── choreography, spatial resolution, and DMX
└── approved local timelines and model artifacts

             private point-to-point Gigabit Ethernet

Threadripper — offline compute node
├── coherent full-song EDMFormer inference
├── feature and embedding generation
├── student-model training and held-out evaluation
├── offline performance simulation
└── reconstructable object and result cache
```

## Physical network

- Threadripper second Ethernet port: `192.168.50.1/24`
- Lumen `enp1s0`: `192.168.50.2/24`
- No gateway or DNS on either direct-link interface
- Lumen continues using `wlp0s20f3` for internet access
- The Threadripper continues using its existing normal network connection
- A normal Cat5e or Cat6 cable connects the two ports

Services on the Threadripper must bind only to the direct link or otherwise
restrict access to the Lumen address.

## Authority and data ownership

The Lumen PC remains the sole authority for the live show and the canonical
runtime database. The machines must never share a writable SQLite database or
network-mounted `state/` directory. The compute node receives immutable,
checksummed job bundles and returns immutable, checksummed result bundles.
Lumen validates and imports a result locally before it becomes available for
timeline review, student training, or Live.

Raw line-in is not streamed through the Threadripper. Lumen records and
reconstructs a coherent song using the authoritative local sample clock, then
transfers that completed WAV for offline work. An unavailable compute node
must leave jobs queued and must never interrupt audio, feedback, choreography,
or DMX.

Recordings, feedback, learned preferences, databases, provider tokens, model
artifacts, and job results remain outside Git. The public repository may carry
only deterministic compute-node source, schemas, tests, example configuration,
and documentation.

## Job contract

The existing durable `analysis_jobs` queue remains authoritative. A remote job
bundle will identify at least:

- schema, job type, and job ID
- recording and capture-session identity
- audio SHA-256 and duration
- model and source revision
- preprocessing and ontology version
- expected result schema
- training split identity when applicable

The returned bundle will include the input identity, normalized timeline or
candidate model, evaluation metrics, code revision, resource measurements, and
artifact checksums. Import is idempotent: retrying a transfer or result cannot
create duplicate teacher authority.

## Implementation stages

1. Configure and verify the point-to-point link without changing either
   machine's normal internet route.
2. Install a headless Ubuntu environment on the Threadripper, initially using
   WSL2 unless its networking or lifetime proves unsuitable. Keep Windows as
   the workstation environment.
3. Split the current offline worker into a pure job executor and a local result
   importer. Remote code must not open Lumen's canonical database.
4. Add a versioned, job-oriented compute service with health, submit, object
   transfer, progress, result, and cancellation operations.
5. Add `local`, `threadripper`, and `automatic` execution targets to the
   existing research queue and show node/job status in Audio Laboratory.
6. Move full-song EDMFormer processing first, then student training and held-out
   evaluation. Keep the validated causal student and exact-song timelines on
   Lumen for network-independent Live operation.
7. Add offline show simulation only after teacher and training parity is proven.

The RX 5700 XT is not required for the first implementation. The initial node
uses the Threadripper 3970X and 128 GB RAM for CPU analysis. GPU acceleration
can later replace an executor without changing Lumen's job contract.

## Acceptance checks

- The same recording produces equivalent normalized local and remote results.
- Interrupted upload, inference, result download, or machine restart resumes
  without duplicating completed work.
- A checksum, schema, revision, duration, or ontology mismatch is rejected.
- Lumen can run physical Live output and accept simultaneous listener feedback
  while the Threadripper analyzes recordings, without changing audio/DMX timing.
- Disconnecting or shutting down the Threadripper does not affect Live.
- A remotely trained candidate cannot become active unless the existing
  held-out validation gates pass.
- Approved timelines and activated models continue working after both Lumen
  and the compute node restart.
