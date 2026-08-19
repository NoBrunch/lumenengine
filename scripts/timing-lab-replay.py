#!/usr/bin/env python3
"""Replay a local PCM16 WAV through Timing Lab without opening DMX hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import wave

from lumen_engine.timing_lab import TimingLabAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay local captured audio through the isolated Timing Lab "
            "detector. No fixture output, memory, or network service is opened."
        )
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument("--expected-bpm", type=float)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--maximum-tempo-switches", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.tolerance <= 0.0:
        parser.error("--tolerance must be positive")
    if arguments.maximum_tempo_switches < 0:
        parser.error("--maximum-tempo-switches cannot be negative")

    with wave.open(str(arguments.wav), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("Timing Lab replay requires 16-bit PCM WAV")
        analyzer = TimingLabAnalyzer(
            source.getframerate(), source.getnchannels()
        )
        chunk_frames = 1_024
        source_frame = 0
        bpms: list[float] = []
        locked_bpms: list[float] = []
        rejected: list[float] = []
        clock_states: dict[str, int] = {}
        beat_events = 0
        predicted_events = 0
        bass_transients = 0
        broad_transients = 0
        switch_events: list[dict[str, float | None]] = []
        previous_switch_count = 0
        while True:
            pcm = source.readframes(chunk_frames)
            if not pcm:
                break
            frame_count = len(pcm) // (source.getnchannels() * 2)
            timestamp_s = (
                source_frame + frame_count / 2.0
            ) / source.getframerate()
            source_frame += frame_count
            result = analyzer.analyze_pcm16(
                pcm, timestamp_s=timestamp_s
            )
            if result.bpm is not None:
                bpms.append(result.bpm)
                if result.clock_state == "locked":
                    locked_bpms.append(result.bpm)
            if result.rejected_candidate_bpm is not None:
                rejected.append(result.rejected_candidate_bpm)
            clock_states[result.clock_state] = (
                clock_states.get(result.clock_state, 0) + 1
            )
            beat_events += int(result.beat_event)
            predicted_events += int(result.beat_event and result.predicted_beat)
            bass_transients += int(result.bass_transient)
            broad_transients += int(result.broadband_transient)
            if result.tempo_switch_count > previous_switch_count:
                switch_events.append({
                    "timestamp_s": result.timestamp_s,
                    "retained_bpm": result.bpm,
                    "family_anchor_bpm": result.family_anchor_bpm,
                    "raw_candidate_bpm": result.raw_candidate_bpm,
                    "normalized_candidate_bpm": result.candidate_bpm,
                    "bass_interval_bpm": result.bass_interval_bpm,
                })
                previous_switch_count = result.tempo_switch_count

    retained_bpm = statistics.median(bpms) if bpms else None
    report = {
        "schema": "lumen_timing_lab_replay_v1",
        "path": str(arguments.wav),
        "duration_s": source_frame / source.getframerate(),
        "sample_rate": source.getframerate(),
        "channels": source.getnchannels(),
        "retained_bpm": retained_bpm,
        "locked_bpm_median": (
            statistics.median(locked_bpms) if locked_bpms else None
        ),
        "beat_events": beat_events,
        "predicted_beat_events": predicted_events,
        "bass_transients": bass_transients,
        "broadband_transients": broad_transients,
        "clock_state_frames": clock_states,
        "rejected_candidate_median": (
            statistics.median(rejected) if rejected else None
        ),
        "family_anchor_bpm": result.family_anchor_bpm,
        "tempo_switch_count": result.tempo_switch_count,
        "tempo_switch_events": switch_events,
        "last_pulse_interval_s": result.last_pulse_interval_s,
        "raw_candidate_bpm": result.raw_candidate_bpm,
        "candidate_harmonic_factor": result.candidate_harmonic_factor,
        "bass_interval_bpm": result.bass_interval_bpm,
        "internal_strobe_dmx": 0,
        "physical_output_opened": False,
    }
    passed = result.tempo_switch_count <= arguments.maximum_tempo_switches
    if arguments.expected_bpm is not None:
        error_ratio = (
            None
            if retained_bpm is None
            else abs(retained_bpm - arguments.expected_bpm)
            / arguments.expected_bpm
        )
        passed = passed and bool(
            error_ratio is not None
            and error_ratio <= arguments.tolerance
        )
        report["acceptance"] = {
            "expected_bpm": arguments.expected_bpm,
            "tolerance": arguments.tolerance,
            "relative_error": error_ratio,
            "maximum_tempo_switches": arguments.maximum_tempo_switches,
            "tempo_switch_count": result.tempo_switch_count,
            "passed": passed,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stdout.close()
        raise SystemExit(0)
