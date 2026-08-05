from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from lumen_engine.audio import AudioInputMetrics
from lumen_engine.control import GatedOutput, LumenApplication
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.models import MediaIdentity, MusicalObservation


class FiveSongKnownTimelineValidation(unittest.TestCase):
    """Deterministic local validation from identity through virtual fixtures."""

    def test_five_exact_recordings_recall_their_approved_timeline_and_cue(self) -> None:
        cases = (
            ("song-a", "intro", "breathe"),
            ("song-b", "breakdown", "fan_sweep"),
            ("song-c", "build", "figure_eight"),
            ("song-d", "drop", "opposing_chase"),
            ("song-e", "groove", "beat_nod"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rig_path = root / "rig.json"
            rig_path.write_text(
                Path("config/party-parrot-active.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            app = LumenApplication(
                rig_path=rig_path,
                memory_path=root / "memory.sqlite3",
                settings_path=root / "settings.json",
            )
            try:
                identities: dict[str, tuple[MediaIdentity, str, str, str]] = {}
                for index, (track_id, energy, routine) in enumerate(cases):
                    media = MediaIdentity(
                        provider="spotify",
                        provider_item_id=f"spotify:track:{track_id}",
                        title=track_id,
                        duration_ms=180_000 + index * 1_000,
                        observed_position_ms=45_000,
                        observed_at_unix_ms=round(time.time() * 1000),
                        is_playing=False,
                    )
                    app._remember_media_identity(media)
                    recording_id = app.memory.remember_recording_version(
                        provider=media.provider,
                        provider_item_id=media.provider_item_id,
                        song_id=app.song_id,
                        duration_ms=media.duration_ms,
                    )
                    run_id = app.memory.begin_teacher_run(
                        teacher_name="EDMFormer",
                        teacher_version="five-song-fake",
                        device="cpu",
                        preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
                        recording_id=recording_id,
                    )
                    timeline_id = app.memory.save_structure_timeline(
                        provenance="edmformer_teacher",
                        timeline_version=TEACHER_NORMALIZATION_VERSION,
                        confidence=0.0,
                        recording_id=recording_id,
                        song_id=app.song_id,
                        teacher_run_id=run_id,
                        segments=[{
                            "start_ms": 0,
                            "end_ms": media.duration_ms,
                            "functional_label": "chorus",
                            "energy_label": energy,
                            "content_label": "instrumental",
                            "raw_label": energy.title(),
                            "label_confidence": 0.0,
                            "boundary_confidence": 0.0,
                        }],
                    )
                    app.memory.finish_teacher_run(run_id, status="complete")
                    app.memory.review_structure_timeline(
                        timeline_id=timeline_id,
                        status="approved",
                        participant_id="validation-owner",
                    )
                    self.assertIsNotNone(
                        app.memory.cached_structure_at(
                            recording_id=recording_id,
                            playback_position_ms=45_000,
                        ),
                        f"initial lookup: {track_id}",
                    )
                    sequence = app.save_choreography_proposal({
                        "name": f"{track_id} exact cue",
                        "scope": "movers",
                        "place": True,
                        "steps": [{
                            "routine": routine,
                            "duration_beats": 16,
                            "motion_speed": 0.35 + index * 0.05,
                            "travel_size": 0.80,
                            "activity_density": 0.70,
                            "brightness": 0.75,
                            "palette": "midnight_teal",
                            "strobe_enabled": False,
                            "strobe_rate": 0.0,
                            "beat_sync": 0.85,
                            "cue_timing": 0.90,
                        }],
                    })
                    identities[track_id] = (
                        media, recording_id, timeline_id,
                        str(sequence["sequence_id"]),
                    )

                # Replay in reverse order to ensure the most recently queried
                # track cannot leak its timeline or taught cue into another.
                for index, (track_id, energy, routine) in reversed(
                    list(enumerate(cases))
                ):
                    media, recording_id, timeline_id, sequence_id = identities[
                        track_id
                    ]
                    app._remember_media_identity(media)
                    app._cached_structure_prediction = None
                    direct = app.memory.cached_structure_at(
                        recording_id=recording_id,
                        playback_position_ms=45_000,
                    )
                    self.assertIsNotNone(direct, f"direct lookup: {track_id}")
                    app._poll_memory_context_once()
                    cached = app._cached_structure_prediction
                    self.assertIsNotNone(cached, track_id)
                    assert cached is not None
                    self.assertEqual(cached["recording"]["id"], recording_id)
                    self.assertEqual(cached["axes"]["energy"]["label"], energy)
                    self.assertEqual(
                        cached["axes"]["energy"]["timeline_id"], timeline_id
                    )
                    self.assertEqual(
                        cached["axes"]["energy"]["model_confidence"], 0.0
                    )
                    self.assertEqual(
                        cached["axes"]["energy"]["operator_trust"], 1.0
                    )
                    self.assertEqual(
                        len(app._prepared_recalled_choreography), 1
                    )
                    recalled = app._prepared_recalled_choreography[0]
                    self.assertTrue(recalled.sequence_id.startswith(sequence_id))
                    self.assertEqual(recalled.steps[0].routine, routine)

                    raw = MusicalObservation(
                        timestamp_s=100.0 + index,
                        loudness=0.72,
                        onset_strength=0.58,
                        low_energy=0.68,
                        mid_energy=0.52,
                        high_energy=0.40,
                        beat_phase=0.0,
                        bar_phase=0.0,
                        beat_pulse=0.8,
                        beat_confidence=0.9,
                        bpm=124.0,
                        section="groove",
                        section_confidence=0.7,
                    )
                    metrics = AudioInputMetrics(
                        timestamp_s=raw.timestamp_s,
                        frame_count=2_048,
                        rms=0.11,
                        dbfs=-19.0,
                        peak=0.36,
                        channel_rms=(0.11, 0.10),
                        channel_peak=(0.36, 0.34),
                        clipped_samples=0,
                        waveform=(0.0,),
                    )
                    resolved = app._resolve_structure(raw, metrics)
                    self.assertEqual(resolved.section, energy)
                    self.assertEqual(
                        app._effective_structure["axes"]["energy"]["source"],
                        "operator_approved_timeline",
                    )

                    virtual = VirtualDMXOutput()
                    runtime = app._runtime_for_rig(
                        GatedOutput(virtual, app.controls)
                    )
                    try:
                        runtime.set_media_context(app.song_id, energy)
                        app._refresh_recalled_choreography(runtime, resolved)
                        frame = runtime.step(resolved)
                        trace = app._training_frame_payload(
                            resolved,
                            frame,
                            metrics,
                            raw_observation=raw,
                        )
                        self.assertTrue(trace["fixture_dmx"])
                        self.assertTrue(trace["effective_outputs"])
                        self.assertEqual(
                            trace["structure_resolution"]["axes"]["energy"][
                                "timeline_id"
                            ],
                            timeline_id,
                        )
                        self.assertTrue(all(
                            item["channels"] for item in trace["fixture_dmx"]
                        ))
                    finally:
                        runtime.close()
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
