from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from lumen_engine.control import LumenApplication, OperatorControls, OperatorExpressionEngine
from lumen_engine.expression import ExpressionPolicy
from lumen_engine.models import MusicalObservation


class ControlApplicationTests(unittest.TestCase):
    def test_default_operator_bias_preserves_audio_dynamics(self) -> None:
        engine = OperatorExpressionEngine(OperatorControls(), ExpressionPolicy())
        quiet = None
        for index in range(24):
            quiet = engine.decide(
                MusicalObservation(
                    timestamp_s=index * 0.1,
                    loudness=0.05,
                    onset_strength=0.02,
                    low_energy=0.3,
                    mid_energy=0.3,
                    high_energy=0.2,
                    section="breakdown",
                    section_confidence=0.8,
                )
            )
        assert quiet is not None
        loud = None
        for index in range(24, 64):
            loud = engine.decide(
                MusicalObservation(
                    timestamp_s=index * 0.1,
                    loudness=0.9,
                    onset_strength=0.7,
                    low_energy=0.6,
                    mid_energy=0.5,
                    high_energy=0.4,
                    beat_pulse=0.8,
                    beat_confidence=0.8,
                    section="groove",
                    section_confidence=0.8,
                )
            )
        assert loud is not None
        self.assertGreater(loud.expression.energy - quiet.expression.energy, 0.35)
        self.assertGreater(loud.expression.motion - quiet.expression.motion, 0.20)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.rig_path = root / "rig.json"
        self.rig_path.write_text(
            Path("config/example-rig.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=root / "memory.sqlite3",
            settings_path=root / "settings.json",
        )

    def tearDown(self) -> None:
        self.application.close()
        self.temporary.cleanup()

    def test_demo_drives_operator_status_without_hardware(self) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        status = self.application.snapshot()
        while status["decision"] is None and time.monotonic() < deadline:
            time.sleep(0.03)
            status = self.application.snapshot()
        self.assertEqual(status["engine"]["mode"], "demo")
        self.assertEqual(status["output"]["backend"], "Virtual DMX")
        self.assertEqual(status["audio"]["state"], "simulated")
        self.assertEqual(status["audio"]["packets_received"], 0)
        self.assertIsNotNone(status["decision"])
        self.assertGreater(len(status["dmx"]["active_channels"]), 0)
        self.assertEqual(self.application.memory.summary()["totals"]["decisions"], 0)
        self.application.stop()

    def test_controls_feedback_target_and_fixture_edit_are_operable(self) -> None:
        status = self.application.apply_preset("restrained")
        self.assertAlmostEqual(status["controls"]["motion"], 0.18)
        status = self.application.patch_controls({"blackout": True, "master": 0.4})
        self.assertTrue(status["controls"]["blackout"])
        self.assertAlmostEqual(status["controls"]["master"], 0.4)
        settings = self.application.patch_settings(
            {"audio_device": "hw:0,2", "spotify_client_id": "test-client-id-1234"}
        )
        self.assertEqual(settings["settings"]["audio_device"], "hw:0,2")
        self.assertEqual(
            settings["settings"]["spotify_client_id_masked"],
            "test-c…1234",
        )

        feedback = self.application.add_feedback(
            {"label": "liked_this", "value": 1, "note": "Good spatial restraint."}
        )
        self.assertGreater(feedback["feedback_id"], 0)
        self.assertEqual(
            self.application.memory.summary()["recent_feedback"][0]["note"],
            "Good spatial restraint.",
        )

        solutions = self.application.solve_target(
            self.application.selected_target
        )
        self.assertEqual(len(solutions), 2)

        fixture_id = self.application.rig.fixtures[0].fixture_id
        bootstrap = self.application.patch_fixture(
            {
                "fixture_id": fixture_id,
                "name": "Edited in console",
                "position_m": [-1.1, -2.2, 2.8],
            }
        )
        saved = json.loads(self.rig_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["fixtures"][0]["name"], "Edited in console")
        self.assertEqual(
            bootstrap["rig"]["fixtures"][0]["position_m"],
            [-1.1, -2.2, 2.8],
        )

    def test_operator_note_is_rebuilt_and_fixture_feedback_stays_scoped(self) -> None:
        self.application.add_feedback(
            {
                "label": "operator_note",
                "value": 1,
                "note": "more movement, no strobe, cooler",
            }
        )
        overall = self.application._feedback_biases["overall"]
        self.assertGreater(overall["motion"], 0.0)
        self.assertLess(overall["strobe"], 0.0)
        self.assertLess(overall["palette"], 0.0)

        fixture_id = self.application.rig.fixtures[0].fixture_id
        other_id = self.application.rig.fixtures[1].fixture_id
        self.application.add_feedback(
            {
                "label": "increase_movement",
                "value": 1,
                "scope": "fixture",
                "fixture_id": fixture_id,
            }
        )
        self.assertIn(fixture_id, self.application._feedback_biases)
        self.assertNotIn(other_id, self.application._feedback_biases)

        memory_path = self.application.memory_path
        self.application.close()
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=memory_path,
            settings_path=Path(self.temporary.name) / "settings.json",
        )
        rebuilt = self.application._feedback_biases["overall"]
        self.assertGreater(rebuilt["motion"], 0.0)
        self.assertLess(rebuilt["strobe"], 0.0)

    @patch("lumen_engine.control.subprocess.run")
    @patch("lumen_engine.control.shutil.which", return_value="/usr/bin/amixer")
    def test_default_input_prepares_dedicated_line_mixer(
        self,
        _which: object,
        run: object,
    ) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""
        self.application._prepare_dedicated_line_input()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "amixer",
                    "-q",
                    "-c",
                    "0",
                    "sset",
                    "Input Source",
                    "Line",
                ],
                ["amixer", "-q", "-c", "0", "sset", "Capture", "0dB"],
            ],
        )
        self.assertIn(
            "0 dB",
            self.application.snapshot()["events"][0]["message"],
        )


if __name__ == "__main__":
    unittest.main()
