from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from lumen_engine.control import LumenApplication


class ControlApplicationTests(unittest.TestCase):
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
        self.assertIsNotNone(status["decision"])
        self.assertGreater(len(status["dmx"]["active_channels"]), 0)
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


if __name__ == "__main__":
    unittest.main()
