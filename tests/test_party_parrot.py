from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from lumen_engine.config import load_rig
from lumen_engine.models import Vec3
from lumen_engine.party_parrot import import_party_parrot_show
from lumen_engine.spatial import SpatialTargetingEngine


class PartyParrotImportTests(unittest.TestCase):
    def test_imports_active_show_profiles_coordinates_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "parrot.db"
            self._create_database(database)
            imported = import_party_parrot_show(database)
            self.assertEqual(imported.slug, "garage")
            self.assertEqual(len(imported.fixtures), 2)
            self.assertEqual(len(imported.moving_heads), 1)
            mover = imported.moving_heads[0]
            self.assertEqual(mover.position_m, Vec3(-1.0, -2.0, 2.5))
            self.assertAlmostEqual(mover.housing_rotation.x_deg, 180.0)
            self.assertAlmostEqual(mover.housing_rotation.z_deg, 90.0)
            self.assertEqual(mover.address, 31)
            self.assertEqual(mover.dimmer_channel, 6)
            self.assertLess(
                mover.calibration.pan_dmx_min_u16,
                mover.calibration.pan_dmx_max_u16,
            )
            solution = SpatialTargetingEngine().solve(mover, Vec3(0, 0, 1.2))
            self.assertLess(solution.aim_error_deg, 1e-6)

            output = Path(directory) / "imported.json"
            imported.write_lumen_rig(output)
            rig = load_rig(output)
            self.assertEqual(len(rig.fixtures), 1)
            self.assertEqual(len(rig.auxiliary_fixtures), 1)
            self.assertEqual(
                rig.auxiliary_fixtures[0].profile_key,
                "generic_multi_effect_19ch",
            )

    @staticmethod
    def _create_database(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE shows (
                    id TEXT PRIMARY KEY, slug TEXT, name TEXT, archived INTEGER,
                    active INTEGER, revision INTEGER, floor_width REAL,
                    floor_depth REAL, floor_height REAL, updated_at TEXT
                );
                CREATE TABLE fixtures (
                    id TEXT PRIMARY KEY, show_id TEXT, order_index INTEGER,
                    fixture_type TEXT, name TEXT, group_name TEXT,
                    is_manual INTEGER, address INTEGER, universe TEXT,
                    x REAL, y REAL, z REAL, rotation_x REAL, rotation_y REAL,
                    rotation_z REAL, options TEXT
                );
                """
            )
            connection.execute(
                """
                INSERT INTO shows VALUES(
                    'show', 'garage', 'Garage', 0, 1, 7, 4.0, 6.0, 2.7,
                    '2026-01-01'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO fixtures VALUES(
                    'multi', 'show', 0, 'generic_multi_effect_19ch', NULL,
                    NULL, 0, 1, 'default', 0, 0, 2.4, 0, 0, 0, ?
                )
                """,
                (json.dumps({}),),
            )
            connection.execute(
                """
                INSERT INTO fixtures VALUES(
                    'mover', 'show', 1, 'generic_rgbw_moving_head_11ch', NULL,
                    'Movers', 0, 31, 'default', -1, 2, 2.5,
                    -3.141592653589793, 0, -1.5707963267948966, ?
                )
                """,
                (
                    json.dumps(
                        {
                            "room_pan_left_dmx": 80,
                            "room_pan_right_dmx": 130,
                            "room_tilt_low_dmx": 60,
                            "room_tilt_high_dmx": 210,
                            "home_pan_dmx": 105,
                            "home_tilt_dmx": 135,
                        }
                    ),
                ),
            )
            connection.commit()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()

