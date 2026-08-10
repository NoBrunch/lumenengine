"""Command-line entry point for development and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from lumen_engine.audio import AudioCaptureConfig, AlsaLineIn, RealtimeAudioAnalyzer
from lumen_engine.config import RigConfig, load_rig
from lumen_engine.control import serve
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.expression import ExpressionEngine, ExpressionPolicy
from lumen_engine.media import (
    SpotifyNowPlayingProvider,
    SpotifyOAuthPKCE,
    SpotifyTokenCache,
)
from lumen_engine.memory import SongMemoryStore
from lumen_engine.link import serve_link_node
from lumen_engine.models import Feedback, MediaIdentity, MusicalObservation, Vec3
from lumen_engine.offline import (
    EDMFORMER_JOB,
    SONGFORMER_JOB,
    STUDENT_TRAIN_JOB,
    OfflineResearchWorker,
    ResearchJobCoordinator,
    enqueue_student_training,
)
from lumen_engine.party_parrot import import_party_parrot_show
from lumen_engine.research import ResearchManager
from lumen_engine.runtime import PerformanceRuntime
from lumen_engine.spatial import SpatialTargetingEngine, UnreachableTargetError
from lumen_engine.usb_dmx import (
    OpenDmxUsbOutput,
    describe_open_dmx_environment,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RIG = PROJECT_DIR / "config" / "example-rig.json"
DEFAULT_PARTY_PARROT_DATABASE = (
    PROJECT_DIR.parent / "the partied out parrot" / "parrot_cloud.db"
)
DEFAULT_IMPORTED_RIG = PROJECT_DIR / "config" / "party-parrot-active.json"
DEFAULT_MEMORY = PROJECT_DIR / "state" / "lumen.sqlite3"
DEFAULT_RESEARCH_ROOT = PROJECT_DIR / "state" / "training" / "research"
DEFAULT_SPOTIFY_TOKEN = (
    Path.home() / ".local" / "state" / "lumenengine" / "spotify-token.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumen",
        description="Lumen Engine development and diagnostic commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run the simulated expression-to-virtual-DMX pipeline"
    )
    demo.add_argument("--rig", type=Path, default=DEFAULT_RIG)
    demo.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    demo.add_argument("--steps", type=int, default=18)
    demo.add_argument("--realtime", action="store_true")
    demo.set_defaults(handler=_demo)

    live_demo = subparsers.add_parser(
        "live-demo", help="run the expression demo through the FT232R/Open-DMX cable"
    )
    live_demo.add_argument("--rig", type=Path, default=DEFAULT_IMPORTED_RIG)
    live_demo.add_argument("--duration", type=float, default=15.0)
    live_demo.add_argument("--driver", choices=("native", "tty"), default="native")
    live_demo.add_argument("--port")
    live_demo.set_defaults(handler=_live_demo)

    run = subparsers.add_parser(
        "run", help="drive the imported rig continuously from ALSA line-in"
    )
    run.add_argument("--rig", type=Path, default=DEFAULT_IMPORTED_RIG)
    run.add_argument("--device", default="default")
    run.add_argument("--sample-rate", type=int, default=48_000)
    run.add_argument("--channels", type=int, default=2)
    run.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; zero continues until Ctrl+C",
    )
    run.add_argument("--driver", choices=("native", "tty"), default="native")
    run.add_argument("--port")
    run.set_defaults(handler=_run_live_audio)

    target = subparsers.add_parser(
        "target", help="solve one 3D target for every configured moving head"
    )
    target.add_argument("x", type=float)
    target.add_argument("y", type=float)
    target.add_argument("z", type=float)
    target.add_argument("--rig", type=Path, default=DEFAULT_RIG)
    target.set_defaults(handler=_target)

    listen = subparsers.add_parser(
        "listen", help="analyze the ALSA line input without generating DMX"
    )
    listen.add_argument("--device", default="default")
    listen.add_argument("--duration", type=float, default=10.0)
    listen.add_argument("--sample-rate", type=int, default=48_000)
    listen.add_argument("--channels", type=int, default=2)
    listen.set_defaults(handler=_listen)

    devices = subparsers.add_parser(
        "audio-devices", help="show ALSA capture hardware and device names"
    )
    devices.set_defaults(handler=_audio_devices)

    dmx_devices = subparsers.add_parser(
        "dmx-devices", help="show the FT232R/Open-DMX environment"
    )
    dmx_devices.set_defaults(handler=_dmx_devices)

    blackout = subparsers.add_parser(
        "dmx-blackout", help="write a zeroed universe through the FT232R cable"
    )
    blackout.add_argument("--driver", choices=("native", "tty"), default="native")
    blackout.add_argument("--port")
    blackout.set_defaults(handler=_dmx_blackout)

    import_party = subparsers.add_parser(
        "import-party-parrot",
        help="import a Party Parrot show database into a Lumen rig",
    )
    import_party.add_argument(
        "--database", type=Path, default=DEFAULT_PARTY_PARROT_DATABASE
    )
    import_party.add_argument("--show", help="show slug; defaults to active show")
    import_party.add_argument("--output", type=Path, default=DEFAULT_IMPORTED_RIG)
    import_party.set_defaults(handler=_import_party_parrot)

    doctor = subparsers.add_parser(
        "doctor", help="check the local runtime, configuration, and audio visibility"
    )
    doctor.add_argument("--rig", type=Path, default=DEFAULT_RIG)
    doctor.set_defaults(handler=_doctor)

    spotify_login = subparsers.add_parser(
        "spotify-login", help="connect the owner's Spotify account using local PKCE"
    )
    spotify_login.add_argument(
        "--client-id", default=os.environ.get("LUMEN_SPOTIFY_CLIENT_ID")
    )
    spotify_login.add_argument("--token-cache", type=Path, default=DEFAULT_SPOTIFY_TOKEN)
    spotify_login.add_argument("--no-browser", action="store_true")
    spotify_login.set_defaults(handler=_spotify_login)

    spotify_now = subparsers.add_parser(
        "spotify-now", help="display the current Spotify Connect playback identity"
    )
    spotify_now.add_argument(
        "--client-id", default=os.environ.get("LUMEN_SPOTIFY_CLIENT_ID")
    )
    spotify_now.add_argument("--token-cache", type=Path, default=DEFAULT_SPOTIFY_TOKEN)
    spotify_now.add_argument("--remember", action="store_true")
    spotify_now.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    spotify_now.set_defaults(handler=_spotify_now)

    memory = subparsers.add_parser(
        "memory", help="show one locally remembered song and its feedback"
    )
    memory.add_argument("song_id", type=int)
    memory.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    memory.set_defaults(handler=_memory)

    feedback = subparsers.add_parser(
        "feedback", help="record plain-language feedback in private song memory"
    )
    feedback.add_argument("song_id", type=int)
    feedback.add_argument(
        "label",
        help="short category such as liked_this, too_busy, too_bright, or bad_timing",
    )
    feedback.add_argument("--position-ms", type=int)
    feedback.add_argument(
        "--value",
        type=float,
        default=-1.0,
        help="preference strength; negative rejects, positive approves",
    )
    feedback.add_argument("--note", help="your own words; no lighting jargon required")
    feedback.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    feedback.set_defaults(handler=_feedback)

    research_status = subparsers.add_parser(
        "research-status",
        help="inspect datasets, teacher environments, models, and local jobs",
    )
    research_status.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    research_status.add_argument(
        "--memory", type=Path, default=DEFAULT_MEMORY
    )
    research_status.set_defaults(handler=_research_status)

    research_sources = subparsers.add_parser(
        "research-provision-sources",
        help="download the public annotation and teacher source repositories",
    )
    research_sources.add_argument(
        "components",
        nargs="*",
        help="optional component IDs; omit for every configured source",
    )
    research_sources.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    research_sources.add_argument(
        "--memory", type=Path, default=DEFAULT_MEMORY
    )
    research_sources.set_defaults(handler=_research_provision_sources)

    research_import = subparsers.add_parser(
        "research-import-annotations",
        help="normalize available research annotations into Lumen memory",
    )
    research_import.add_argument(
        "components",
        nargs="*",
        choices=("edm98", "harmonix", "ccmusic", "salami"),
        help="optional dataset IDs; omit to import every annotation source",
    )
    research_import.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    research_import.add_argument(
        "--memory", type=Path, default=DEFAULT_MEMORY
    )
    research_import.set_defaults(handler=_research_import_annotations)

    research_prepare = subparsers.add_parser(
        "research-prepare-export",
        help="reconstruct captured songs and queue offline teacher analysis",
    )
    research_prepare.add_argument("export", type=Path)
    research_prepare.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    research_prepare.add_argument(
        "--memory", type=Path, default=DEFAULT_MEMORY
    )
    research_prepare.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT.parent,
    )
    research_prepare.add_argument(
        "--songformer",
        action="store_true",
        help="also queue SongFormer (only runs after its CPU gate passes)",
    )
    research_prepare.set_defaults(handler=_research_prepare_export)

    research_worker = subparsers.add_parser(
        "research-worker",
        help="process queued teachers/student jobs outside the live DMX loop",
    )
    research_worker.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    research_worker.add_argument(
        "--memory", type=Path, default=DEFAULT_MEMORY
    )
    research_worker.add_argument(
        "--job-type",
        action="append",
        choices=(EDMFORMER_JOB, SONGFORMER_JOB, STUDENT_TRAIN_JOB),
        default=[],
    )
    research_worker.add_argument("--max-jobs", type=int, default=1)
    research_worker.set_defaults(handler=_research_worker)

    research_train = subparsers.add_parser(
        "research-train-student",
        help="train/evaluate the small CPU streaming model from teacher labels",
    )
    research_train.add_argument(
        "examples",
        nargs="*",
        type=Path,
        help="teacher example JSONL files; defaults to all prepared examples",
    )
    research_train.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    research_train.add_argument(
        "--memory", type=Path, default=DEFAULT_MEMORY
    )
    research_train.add_argument("--epochs", type=int, default=30)
    research_train.set_defaults(handler=_research_train_student)

    link_node = subparsers.add_parser(
        "link-node",
        help="run the authenticated Lumen Link compute service in WSL",
    )
    link_node.add_argument("--host", default="0.0.0.0")
    link_node.add_argument("--port", type=int, default=8765)
    link_node.add_argument(
        "--secret-file",
        type=Path,
        default=Path.home() / ".config" / "lumen-link" / "secret",
    )
    link_node.add_argument(
        "--spool",
        type=Path,
        default=Path.home() / ".local" / "state" / "lumen-link",
    )
    link_node.add_argument(
        "--root", type=Path, default=DEFAULT_RESEARCH_ROOT
    )
    link_node.add_argument("--max-threads", type=int, default=24)
    link_node.add_argument("--max-memory-gib", type=float, default=96.0)
    link_node.add_argument("--maximum-parallel-jobs", type=int, default=4)
    link_node.set_defaults(handler=_link_node)

    ui = subparsers.add_parser(
        "ui", help="start the desktop console and phone/tablet remote"
    )
    ui.add_argument("--rig", type=Path, default=DEFAULT_IMPORTED_RIG)
    ui.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    ui.add_argument("--device", default="default")
    ui.add_argument("--host", default="0.0.0.0")
    ui.add_argument("--port", type=int, default=4042)
    ui.add_argument(
        "--open",
        action="store_true",
        help="open the desktop console in the default browser",
    )
    ui.set_defaults(handler=_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _demo(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    rig = load_rig(args.rig)
    output = VirtualDMXOutput()
    runtime = _runtime_for_rig(rig, output)
    memory = SongMemoryStore(args.memory)
    song_id = memory.remember_media(
        MediaIdentity(
            provider="simulation",
            provider_item_id="lumen-first-light",
            title="Lumen First Light",
            artists=("Lumen Engine",),
            duration_ms=args.steps * 750,
            is_playing=True,
        ),
        count_play=True,
    )

    print(f"Lumen Engine · {rig.name}")
    print(f"Output: virtual only · fixtures: {len(rig.fixtures)} · song memory: {song_id}")
    print()
    last_gesture = None
    for index in range(args.steps):
        observation = _synthetic_observation(index, args.steps)
        result = runtime.step(observation)
        decision = result.decision
        position_ms = round(observation.timestamp_s * 1000)
        if decision.gesture != last_gesture:
            memory.log_decision(decision, song_id=song_id, position_ms=position_ms)
            last_gesture = decision.gesture
        targets = ", ".join(
            f"{solution.fixture_id}=({solution.pan_deg:6.1f}°, "
            f"{solution.tilt_deg:6.1f}°)"
            for solution in result.solutions
        )
        print(
            f"{observation.timestamp_s:5.2f}s "
            f"{decision.gesture.value:9} "
            f"energy={decision.expression.energy:.2f} "
            f"tension={decision.expression.tension:.2f} "
            f"brightness={decision.brightness:.2f} · {targets}"
        )
        if result.warnings:
            for warning in result.warnings:
                print(f"  warning: {warning}")
        if args.realtime:
            time.sleep(0.75)
    runtime.close()
    print()
    print(f"Generated {output.frame_count} virtual DMX frames; no hardware was touched.")
    print(f"Reason for final decision: {result.decision.reason}")
    return 0


def _live_demo(args: argparse.Namespace) -> int:
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    rig = load_rig(args.rig)
    output = OpenDmxUsbOutput.open(driver=args.driver, port=args.port)
    runtime = _runtime_for_rig(rig, output)
    steps = max(1, round(args.duration / 0.75))
    print(
        f"Open-DMX: {output.status.backend}\n"
        f"Rig: {rig.name}; {len(rig.fixtures)} moving heads; "
        f"{len(rig.auxiliary_fixtures)} auxiliary fixtures"
    )
    started = time.monotonic()
    try:
        for index in range(steps):
            observation = _synthetic_observation(index, steps)
            # Use wall-relative time so motion runs at its intended speed.
            observation = MusicalObservation(
                timestamp_s=time.monotonic() - started,
                loudness=observation.loudness,
                onset_strength=observation.onset_strength,
                low_energy=observation.low_energy,
                mid_energy=observation.mid_energy,
                high_energy=observation.high_energy,
                beat_phase=observation.beat_phase,
                bar_phase=observation.bar_phase,
                beat_pulse=observation.beat_pulse,
                beat_confidence=observation.beat_confidence,
                bpm=observation.bpm,
                section=observation.section,
                section_confidence=observation.section_confidence,
                novelty=observation.novelty,
            )
            result = runtime.step(observation)
            print(
                f"\r{result.decision.gesture.value:9} "
                f"energy={result.decision.expression.energy:.2f} "
                f"frames={output.status.frames_sent}",
                end="",
                flush=True,
            )
            time.sleep(0.75)
    finally:
        runtime.close()
    print("\nOpen-DMX output closed.")
    return 0


def _run_live_audio(args: argparse.Namespace) -> int:
    if args.duration < 0:
        raise ValueError("--duration must be non-negative")
    rig = load_rig(args.rig)
    capture_config = AudioCaptureConfig(
        device=args.device,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    analyzer = RealtimeAudioAnalyzer(
        capture_config.sample_rate, capture_config.channels
    )
    output = OpenDmxUsbOutput.open(driver=args.driver, port=args.port)
    runtime = _runtime_for_rig(rig, output)
    started = time.monotonic()
    print(
        f"Listening on {capture_config.device!r}; output through "
        f"{output.status.backend}\n"
        f"Rig: {rig.name}. Press Ctrl+C to stop."
    )
    try:
        with AlsaLineIn(capture_config) as capture:
            for pcm in capture.chunks():
                observation = analyzer.analyze_pcm16(pcm)
                result = runtime.step(observation)
                bpm = "—" if observation.bpm is None else f"{observation.bpm:5.1f}"
                print(
                    f"\r{result.decision.gesture.value:9} "
                    f"loud={observation.loudness:.2f} "
                    f"onset={observation.onset_strength:.2f} "
                    f"bpm={bpm} "
                    f"confidence={observation.beat_confidence:.2f} "
                    f"dmx_frames={output.status.frames_sent}",
                    end="",
                    flush=True,
                )
                if args.duration and time.monotonic() - started >= args.duration:
                    break
    finally:
        runtime.close()
    print("\nLine-in performance stopped.")
    return 0


def _synthetic_observation(index: int, total: int) -> MusicalObservation:
    progress = index / max(total - 1, 1)
    timestamp = index * 0.75
    if progress < 0.24:
        section, section_confidence = "intro", 0.84
        loudness = 0.12 + 0.20 * progress / 0.24
        onset = 0.12 if index % 2 else 0.30
    elif progress < 0.67:
        section, section_confidence = "build", 0.91
        build = (progress - 0.24) / 0.43
        loudness = 0.33 + 0.46 * build
        onset = 0.40 + 0.26 * (index % 2 == 0)
    elif index == round(0.70 * (total - 1)):
        section, section_confidence = "drop", 0.96
        loudness, onset = 0.96, 1.0
    else:
        section, section_confidence = "chorus", 0.88
        loudness = 0.78 + 0.10 * math.sin(index)
        onset = 0.72 if index % 2 == 0 else 0.36
    return MusicalObservation(
        timestamp_s=timestamp,
        loudness=max(0.0, min(1.0, loudness)),
        onset_strength=float(onset),
        low_energy=min(1.0, 0.35 + 0.55 * progress),
        mid_energy=min(1.0, 0.48 + 0.25 * progress),
        high_energy=min(1.0, 0.20 + 0.62 * progress),
        beat_phase=0.0,
        bar_phase=(index % 4) / 4.0,
        beat_pulse=1.0 if index % 2 == 0 else 0.12,
        beat_confidence=min(0.95, 0.45 + 0.50 * progress),
        bpm=120.0,
        section=section,
        section_confidence=section_confidence,
        novelty=0.82 if section == "drop" else 0.25 + 0.30 * progress,
    )


def _runtime_for_rig(
    rig: RigConfig, output: VirtualDMXOutput | OpenDmxUsbOutput
) -> PerformanceRuntime:
    width_target = max(0.25, rig.room.width_m * 0.35)
    center_height = min(1.2, rig.room.height_m * 0.45)
    policy = ExpressionPolicy(
        room_center=Vec3(0.0, 0.0, center_height),
        room_high=Vec3(0.0, 0.0, min(rig.room.height_m * 0.82, 2.4)),
        room_wide=Vec3(
            width_target,
            0.0,
            min(rig.room.height_m * 0.52, 1.4),
        ),
    )
    return PerformanceRuntime(
        rig.fixtures,
        output,
        auxiliary_fixtures=rig.auxiliary_fixtures,
        expression=ExpressionEngine(policy),
        motion_extents=Vec3(
            max(0.8, rig.room.width_m * 0.44),
            max(1.8, rig.room.depth_m * 0.33),
            min(2.6, max(2.2, rig.room.height_m * 0.45)),
        ),
    )


def _target(args: argparse.Namespace) -> int:
    rig = load_rig(args.rig)
    target = Vec3(args.x, args.y, args.z)
    engine = SpatialTargetingEngine()
    failed = False
    for fixture in rig.fixtures:
        try:
            solution = engine.solve(fixture, target)
            print(
                f"{fixture.fixture_id}: pan={solution.pan_deg:.3f}°, "
                f"tilt={solution.tilt_deg:.3f}°, distance={solution.distance_m:.3f}m, "
                f"error={solution.aim_error_deg:.8f}°, branch={solution.branch}"
            )
        except UnreachableTargetError as error:
            failed = True
            print(f"{fixture.fixture_id}: unreachable: {error}")
    return 1 if failed else 0


def _listen(args: argparse.Namespace) -> int:
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    config = AudioCaptureConfig(
        device=args.device,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    analyzer = RealtimeAudioAnalyzer(config.sample_rate, config.channels)
    started = time.monotonic()
    print(
        f"Listening to ALSA device {config.device!r} for {args.duration:.1f}s "
        "(analysis only; no DMX)..."
    )
    with AlsaLineIn(config) as capture:
        for pcm in capture.chunks():
            observation = analyzer.analyze_pcm16(pcm)
            bpm = "—" if observation.bpm is None else f"{observation.bpm:5.1f}"
            print(
                f"\rloud={observation.loudness:.2f} "
                f"onset={observation.onset_strength:.2f} "
                f"L/M/H={observation.low_energy:.2f}/"
                f"{observation.mid_energy:.2f}/{observation.high_energy:.2f} "
                f"bpm={bpm} confidence={observation.beat_confidence:.2f}",
                end="",
                flush=True,
            )
            if time.monotonic() - started >= args.duration:
                break
    print()
    return 0


def _audio_devices(_: argparse.Namespace) -> int:
    if shutil.which("arecord") is None:
        raise RuntimeError("arecord is not installed")
    for command in (["arecord", "-l"], ["arecord", "-L"]):
        print(f"$ {' '.join(command)}")
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        text = (result.stdout + result.stderr).strip()
        print(text or "(no devices reported)")
        print()
    return 0


def _dmx_devices(_: argparse.Namespace) -> int:
    report = describe_open_dmx_environment()
    print(json.dumps(report, indent=2))
    if report["native_driver_ready"]:
        print("Native libftdi transport is installed.")
    else:
        print("libftdi1 was not found.")
    return 0


def _dmx_blackout(args: argparse.Namespace) -> int:
    output = OpenDmxUsbOutput.open(driver=args.driver, port=args.port)
    try:
        output.blackout()
        time.sleep(0.15)
        status = output.status
        print(
            f"Zeroed universe {status.universe} through {status.backend}; "
            f"{status.frames_sent} frames sent."
        )
    finally:
        output.close()
    return 0


def _import_party_parrot(args: argparse.Namespace) -> int:
    imported = import_party_parrot_show(args.database, args.show)
    imported.write_lumen_rig(args.output)
    print(
        f"Imported {imported.name!r} revision {imported.revision}: "
        f"{len(imported.fixtures)} fixtures, "
        f"{len(imported.moving_heads)} spatial moving heads."
    )
    print(f"Wrote {args.output}")
    for warning in imported.warnings:
        print(f"warning: {warning}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "Python",
            sys.version_info >= (3, 12),
            sys.version.split()[0],
        )
    )
    checks.append(("arecord", shutil.which("arecord") is not None, "ALSA capture tool"))
    try:
        rig: RigConfig = load_rig(args.rig)
        checks.append(
            (
                "Rig config",
                True,
                f"{rig.name}; {len(rig.fixtures)} fixtures",
            )
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        checks.append(("Rig config", False, str(error)))

    if shutil.which("arecord"):
        result = subprocess.run(
            ["arecord", "-l"], text=True, capture_output=True, check=False
        )
        capture_report = (result.stdout + result.stderr).strip()
        capture_visible = (
            result.returncode == 0
            and "no soundcards found" not in capture_report.lower()
            and "no soundcards" not in capture_report.lower()
        )
        checks.append(
            (
                "Capture hardware",
                capture_visible,
                capture_report.splitlines()[-1]
                if capture_report
                else "none reported",
            )
        )
    dmx_environment = describe_open_dmx_environment()
    ftdi_devices = dmx_environment["ft232r_devices"]
    checks.append(
        (
            "Open-DMX library",
            bool(dmx_environment["native_driver_ready"]),
            str(dmx_environment["libftdi1"] or "libftdi1 not found"),
        )
    )
    checks.append(
        (
            "FT232R adapter",
            bool(ftdi_devices),
            (
                str(ftdi_devices[0].get("usb_path", "detected"))
                if isinstance(ftdi_devices, list) and ftdi_devices
                else "not visible"
            ),
        )
    )
    failed = False
    for name, ok, detail in checks:
        failed |= not ok
        print(f"{'OK' if ok else 'NOT READY':9} {name:18} {detail}")
    print()
    print("Virtual `demo` and direct `run`/`live-demo` output paths are available.")
    return 1 if failed else 0


def _spotify_components(
    args: argparse.Namespace,
) -> tuple[SpotifyOAuthPKCE, SpotifyNowPlayingProvider]:
    if not args.client_id:
        raise ValueError(
            "provide --client-id or set LUMEN_SPOTIFY_CLIENT_ID; "
            "do not commit credentials"
        )
    oauth = SpotifyOAuthPKCE(
        client_id=args.client_id,
        cache=SpotifyTokenCache(args.token_cache),
    )
    return oauth, SpotifyNowPlayingProvider(oauth.valid_token)


def _spotify_login(args: argparse.Namespace) -> int:
    oauth, _ = _spotify_components(args)
    print(
        "Spotify will ask for read-only playback-state access. "
        "The token is stored locally with owner-only permissions."
    )
    oauth.login(open_browser=not args.no_browser)
    print(f"Connected. Token cache: {args.token_cache}")
    return 0


def _spotify_now(args: argparse.Namespace) -> int:
    _, provider = _spotify_components(args)
    media = provider.now_playing()
    if media is None:
        print("Spotify reports no active playback.")
        return 0
    print(media.display_name)
    print(f"Album: {media.album or '—'}")
    print(f"Track identity: {media.provider_item_id or '—'}")
    print(
        f"Position: {media.observed_position_ms or 0} / {media.duration_ms or 0} ms"
    )
    print(f"Device: {media.device_name or '—'}")
    print(f"Playing: {'yes' if media.is_playing else 'no'}")
    if args.remember:
        song_id = SongMemoryStore(args.memory).remember_media(media, count_play=True)
        print(f"Remembered privately as Lumen song {song_id}.")
    return 0


def _memory(args: argparse.Namespace) -> int:
    store = SongMemoryStore(args.memory)
    song = store.get_song(args.song_id)
    if song is None:
        raise ValueError(f"song {args.song_id} is not in {args.memory}")
    feedback = store.list_feedback(args.song_id)
    print(json.dumps({"song": song, "feedback": feedback}, indent=2))
    return 0


def _feedback(args: argparse.Namespace) -> int:
    store = SongMemoryStore(args.memory)
    if store.get_song(args.song_id) is None:
        raise ValueError(f"song {args.song_id} is not in {args.memory}")
    feedback_id = store.add_feedback(
        Feedback(
            song_id=args.song_id,
            position_ms=args.position_ms,
            label=args.label,
            value=args.value,
            note=args.note,
        )
    )
    print(
        f"Saved feedback {feedback_id} for song {args.song_id}: "
        f"{args.label} ({args.value:+g})"
    )
    if args.note:
        print(f"Your note: {args.note}")
    return 0


def _research_status(args: argparse.Namespace) -> int:
    manager = ResearchManager(
        args.root,
        store=SongMemoryStore(args.memory),
    )
    print(json.dumps(manager.status(), indent=2, sort_keys=True))
    return 0


def _research_provision_sources(args: argparse.Namespace) -> int:
    manager = ResearchManager(
        args.root,
        store=SongMemoryStore(args.memory),
    )
    result = manager.provision_sources(args.components)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        1
        if any(item["state"] == "error" for item in result["results"])
        else 0
    )


def _research_import_annotations(args: argparse.Namespace) -> int:
    manager = ResearchManager(
        args.root,
        store=SongMemoryStore(args.memory),
    )
    result = manager.import_annotations(args.components)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if all(item["state"] == "imported" for item in result["results"])
        else 1
    )


def _research_prepare_export(args: argparse.Namespace) -> int:
    coordinator = ResearchJobCoordinator(
        SongMemoryStore(args.memory),
        training_root=args.training_root,
        research_root=args.root,
    )
    result = coordinator.prepare_export(
        args.export,
        queue_songformer=bool(args.songformer),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _research_worker(args: argparse.Namespace) -> int:
    if args.max_jobs < 1:
        raise ValueError("--max-jobs must be at least 1")
    worker = OfflineResearchWorker(
        SongMemoryStore(args.memory),
        research_root=args.root,
    )
    results = worker.run_until_empty(
        args.job_type, maximum_jobs=args.max_jobs
    )
    print(json.dumps({"jobs": results}, indent=2, sort_keys=True))
    return 1 if any(item["status"] == "failed" for item in results) else 0


def _research_train_student(args: argparse.Namespace) -> int:
    store = SongMemoryStore(args.memory)
    queued = enqueue_student_training(
        store,
        research_root=args.root,
        example_paths=args.examples,
        epochs=args.epochs,
    )
    result = OfflineResearchWorker(
        store, research_root=args.root
    ).run_once((STUDENT_TRAIN_JOB,))
    print(
        json.dumps(
            {"queued": queued, "result": result},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result and result["status"] == "complete" else 1


def _link_node(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    try:
        secret = args.secret_file.read_text(encoding="utf-8").strip().encode()
    except OSError as error:
        raise RuntimeError(
            f"Lumen Link secret is unavailable at {args.secret_file}"
        ) from error
    if len(secret) < 32:
        raise ValueError("Lumen Link secret must contain at least 32 bytes")
    if args.max_threads < 1:
        raise ValueError("--max-threads must be at least 1")
    if args.max_memory_gib < 1:
        raise ValueError("--max-memory-gib must be at least 1")
    if args.maximum_parallel_jobs < 1:
        raise ValueError("--maximum-parallel-jobs must be at least 1")
    print(
        f"Lumen Link compute node: http://{args.host}:{args.port} "
        f"(spool {args.spool})"
    )
    serve_link_node(
        host=args.host,
        port=args.port,
        secret=secret,
        spool_root=args.spool,
        research_root=args.root,
        project_root=PROJECT_DIR,
        max_threads=args.max_threads,
        max_memory_gib=args.max_memory_gib,
        maximum_parallel_jobs=args.maximum_parallel_jobs,
    )
    return 0


def _ui(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    serve(
        host=args.host,
        port=args.port,
        rig_path=args.rig,
        memory_path=args.memory,
        audio_device=args.device,
        open_browser=args.open,
    )
    return 0
