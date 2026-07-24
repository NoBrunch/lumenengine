"""Read-only migration boundary for Party Parrot show databases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from lumen_engine.models import (
    EulerXYZ,
    FixtureCalibration,
    FixturePatch,
    Vec3,
)
from lumen_engine.geometry import (
    apply_transpose,
    rotation_matrix_xyz,
)
from lumen_engine.profiles import (
    FixtureProfile,
    MotionKind,
    party_parrot_profile,
    profile_summary,
)


@dataclass(frozen=True, slots=True)
class ImportedPartyFixture:
    fixture_id: str
    fixture_type: str
    name: str
    group_name: str | None
    universe_name: str
    universe: int
    address: int
    position_m: Vec3
    housing_rotation: EulerXYZ
    options: dict[str, Any]
    profile: FixtureProfile | None
    is_manual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fixture_id,
            "fixture_type": self.fixture_type,
            "name": self.name,
            "group_name": self.group_name,
            "is_manual": self.is_manual,
            "universe_name": self.universe_name,
            "universe": self.universe,
            "address": self.address,
            "position_m": list(self.position_m.as_tuple()),
            "housing_rotation_deg": [
                self.housing_rotation.x_deg,
                self.housing_rotation.y_deg,
                self.housing_rotation.z_deg,
            ],
            "options": dict(self.options),
            "profile": (
                None if self.profile is None else profile_summary(self.profile)
            ),
        }


@dataclass(frozen=True, slots=True)
class PartyParrotShowImport:
    show_id: str
    slug: str
    name: str
    revision: int
    room_width_m: float
    room_depth_m: float
    room_height_m: float
    fixtures: tuple[ImportedPartyFixture, ...]
    moving_heads: tuple[FixturePatch, ...]
    warnings: tuple[str, ...]
    source_database: str

    def to_lumen_rig_dict(self) -> dict[str, Any]:
        return {
            "name": f"{self.name} (imported from Party Parrot)",
            "source": {
                "kind": "party_parrot_sqlite",
                "database": self.source_database,
                "show_id": self.show_id,
                "slug": self.slug,
                "revision": self.revision,
            },
            "room": {
                "width_m": self.room_width_m,
                "depth_m": self.room_depth_m,
                "height_m": self.room_height_m,
                "origin_description": (
                    "Party Parrot floor center; +Y points from the audience/front "
                    "toward the back/upstage"
                ),
            },
            "fixtures": [_moving_head_to_dict(item) for item in self.moving_heads],
            "auxiliary_fixtures": [
                _auxiliary_to_dict(item)
                for item in self.fixtures
                if item.profile is not None
                and item.profile.motion_kind is not MotionKind.MOVING_HEAD
            ],
            "party_parrot_fixtures": [item.to_dict() for item in self.fixtures],
            "import_warnings": list(self.warnings),
        }

    def write_lumen_rig(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_lumen_rig_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )


def import_party_parrot_show(
    database: str | Path, show_slug: str | None = None
) -> PartyParrotShowImport:
    source = Path(database).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Party Parrot database not found: {source}")

    uri = f"{source.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if show_slug:
            show = connection.execute(
                "SELECT * FROM shows WHERE slug=?", (show_slug,)
            ).fetchone()
        else:
            show = connection.execute(
                """
                SELECT * FROM shows WHERE active=1 AND archived=0
                ORDER BY updated_at DESC LIMIT 1
                """
            ).fetchone()
        if show is None:
            requested = f" {show_slug!r}" if show_slug else " active"
            raise ValueError(f"Party Parrot has no{requested} show")

        rows = connection.execute(
            """
            SELECT * FROM fixtures WHERE show_id=?
            ORDER BY order_index, address, id
            """,
            (show["id"],),
        ).fetchall()
    finally:
        connection.close()

    imported: list[ImportedPartyFixture] = []
    movers: list[FixturePatch] = []
    warnings: list[str] = []
    for row in rows:
        fixture_type = str(row["fixture_type"])
        profile = party_parrot_profile(fixture_type)
        options = json.loads(row["options"] or "{}")
        position = Vec3(float(row["x"]), -float(row["y"]), float(row["z"]))
        rotation = _party_rotation_to_lumen(
            float(row["rotation_x"]),
            float(row["rotation_y"]),
            float(row["rotation_z"]),
        )
        universe_name = str(row["universe"])
        universe = _universe_number(universe_name)
        fixture_name = str(row["name"] or (profile.label if profile else fixture_type))
        item = ImportedPartyFixture(
            fixture_id=str(row["id"]),
            fixture_type=fixture_type,
            name=fixture_name,
            group_name=(
                None if row["group_name"] in (None, "") else str(row["group_name"])
            ),
            universe_name=universe_name,
            universe=universe,
            address=int(row["address"]),
            position_m=position,
            housing_rotation=rotation,
            options=options,
            profile=profile,
            is_manual=bool(row["is_manual"]),
        )
        imported.append(item)
        if profile is None:
            warnings.append(f"No Lumen profile for Party Parrot type {fixture_type!r}")
            continue
        if profile.motion_kind is MotionKind.MOVING_HEAD:
            movers.append(
                _moving_head_patch(
                    item,
                    profile,
                    home_target=Vec3(0.0, 0.0, min(1.2, float(show["floor_height"]))),
                )
            )

    return PartyParrotShowImport(
        show_id=str(show["id"]),
        slug=str(show["slug"]),
        name=str(show["name"]),
        revision=int(show["revision"]),
        room_width_m=float(show["floor_width"]),
        room_depth_m=float(show["floor_depth"]),
        room_height_m=float(show["floor_height"]),
        fixtures=tuple(imported),
        moving_heads=tuple(movers),
        warnings=tuple(warnings),
        source_database=str(source),
    )


def _party_rotation_to_lumen(
    rotation_x_rad: float, rotation_y_rad: float, rotation_z_rad: float
) -> EulerXYZ:
    """Convert Party Parrot's +Y-toward-audience basis to Lumen +Y-back.

    Reflecting the Y basis conjugates the rotation. For intrinsic XYZ this is
    equivalent to negating the X and Z Euler angles while retaining Y.
    """

    return EulerXYZ(
        x_deg=-math.degrees(rotation_x_rad),
        y_deg=math.degrees(rotation_y_rad),
        z_deg=-math.degrees(rotation_z_rad),
    )


def _universe_number(name: str) -> int:
    if name == "default":
        return 0
    if name == "art1":
        return 1
    try:
        return max(0, int(name))
    except ValueError:
        return 0


def _moving_head_patch(
    fixture: ImportedPartyFixture,
    profile: FixtureProfile,
    home_target: Vec3,
) -> FixturePatch:
    assert profile.pan_degrees is not None
    assert profile.tilt_degrees is not None
    pan = _axis_calibration(
        fixture.options,
        axis="pan",
        total_degrees=profile.pan_degrees,
        room_low_key="room_pan_left_dmx",
        room_high_key="room_pan_right_dmx",
    )
    tilt = _axis_calibration(
        fixture.options,
        axis="tilt",
        total_degrees=profile.tilt_degrees,
        room_low_key="room_tilt_low_dmx",
        room_high_key="room_tilt_high_dmx",
    )
    home_direction = home_target - fixture.position_m
    local_home = apply_transpose(
        rotation_matrix_xyz(fixture.housing_rotation),
        home_direction.normalized(),
    )
    canonical_home_pan = math.degrees(math.atan2(local_home.y, local_home.x))
    canonical_home_tilt = math.degrees(
        math.atan2(local_home.z, math.hypot(local_home.x, local_home.y))
    )
    pan_offset = float(pan["home_deg"]) - int(pan["direction"]) * canonical_home_pan
    tilt_offset = (
        float(tilt["home_deg"]) - int(tilt["direction"]) * canonical_home_tilt
    )
    channels = profile.channels
    return FixturePatch(
        fixture_id=fixture.fixture_id,
        name=fixture.name,
        universe=fixture.universe,
        address=fixture.address,
        position_m=fixture.position_m,
        housing_rotation=fixture.housing_rotation,
        profile_key=profile.key,
        source_metadata={
            "source": "party_parrot",
            "fixture_type": fixture.fixture_type,
            "group_name": fixture.group_name,
            "universe_name": fixture.universe_name,
            "options": dict(fixture.options),
        },
        calibration=FixtureCalibration(
            pan_min_deg=pan["minimum_deg"],
            pan_max_deg=pan["maximum_deg"],
            tilt_min_deg=tilt["minimum_deg"],
            tilt_max_deg=tilt["maximum_deg"],
            pan_offset_deg=pan_offset,
            tilt_offset_deg=tilt_offset,
            pan_direction=int(pan["direction"]),
            tilt_direction=int(tilt["direction"]),
            pan_dmx_min_u16=pan["dmx_min_u16"],
            pan_dmx_max_u16=pan["dmx_max_u16"],
            tilt_dmx_min_u16=tilt["dmx_min_u16"],
            tilt_dmx_max_u16=tilt["dmx_max_u16"],
            max_pan_speed_deg_s=float(
                fixture.options.get("max_pan_speed_deg_s", 180.0)
            ),
            max_tilt_speed_deg_s=float(
                fixture.options.get("max_tilt_speed_deg_s", 180.0)
            ),
        ),
        pan_coarse_channel=int(channels.get("pan_coarse", 1)),
        pan_fine_channel=_optional_channel(channels.get("pan_fine")),
        tilt_coarse_channel=int(channels.get("tilt_coarse", 3)),
        tilt_fine_channel=_optional_channel(channels.get("tilt_fine")),
        dimmer_channel=_optional_channel(channels.get("dimmer")),
    )


def _axis_calibration(
    options: dict[str, Any],
    *,
    axis: str,
    total_degrees: float,
    room_low_key: str,
    room_high_key: str,
) -> dict[str, int | float]:
    room_low = options.get(room_low_key)
    room_high = options.get(room_high_key)
    if room_low is not None and room_high is not None:
        endpoint_a = max(0.0, min(255.0, float(room_low)))
        endpoint_b = max(0.0, min(255.0, float(room_high)))
        raw_min = min(endpoint_a, endpoint_b)
        raw_max = max(endpoint_a, endpoint_b)
        direction = 1 if endpoint_b >= endpoint_a else -1
        minimum = raw_min / 255.0 * total_degrees
        maximum = raw_max / 255.0 * total_degrees
    else:
        minimum = max(
            0.0, min(total_degrees, float(options.get(f"{axis}_lower", 0.0)))
        )
        maximum = max(
            0.0,
            min(total_degrees, float(options.get(f"{axis}_upper", total_degrees))),
        )
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        raw_min = minimum / total_degrees * 255.0
        raw_max = maximum / total_degrees * 255.0
        direction = 1

    if math.isclose(minimum, maximum):
        minimum, maximum = 0.0, total_degrees
        raw_min, raw_max = 0.0, 255.0
    if bool(options.get(f"invert_{axis}", False)):
        direction *= -1
    home_raw = max(
        0.0, min(255.0, float(options.get(f"home_{axis}_dmx", 128.0)))
    )
    return {
        "minimum_deg": minimum,
        "maximum_deg": maximum,
        "home_deg": home_raw / 255.0 * total_degrees,
        "direction": direction,
        "dmx_min_u16": round(raw_min / 255.0 * 65535.0),
        "dmx_max_u16": round(raw_max / 255.0 * 65535.0),
    }


def _optional_channel(value: int | None) -> int | None:
    return None if value is None else int(value)


def _moving_head_to_dict(fixture: FixturePatch) -> dict[str, Any]:
    calibration = asdict(fixture.calibration)
    return {
        "id": fixture.fixture_id,
        "name": fixture.name,
        "profile_key": fixture.profile_key,
        "universe": fixture.universe,
        "address": fixture.address,
        "position_m": list(fixture.position_m.as_tuple()),
        "housing_rotation_deg": [
            fixture.housing_rotation.x_deg,
            fixture.housing_rotation.y_deg,
            fixture.housing_rotation.z_deg,
        ],
        "calibration": calibration,
        "channels": {
            "pan_coarse": fixture.pan_coarse_channel,
            "pan_fine": fixture.pan_fine_channel,
            "tilt_coarse": fixture.tilt_coarse_channel,
            "tilt_fine": fixture.tilt_fine_channel,
            "dimmer": fixture.dimmer_channel,
        },
        "source_metadata": dict(fixture.source_metadata),
    }


def _auxiliary_to_dict(fixture: ImportedPartyFixture) -> dict[str, Any]:
    assert fixture.profile is not None
    return {
        "id": fixture.fixture_id,
        "name": fixture.name,
        "profile_key": fixture.profile.key,
        "universe": fixture.universe,
        "address": fixture.address,
        "position_m": list(fixture.position_m.as_tuple()),
        "housing_rotation_deg": [
            fixture.housing_rotation.x_deg,
            fixture.housing_rotation.y_deg,
            fixture.housing_rotation.z_deg,
        ],
        "options": dict(fixture.options),
        "source_metadata": {
            "source": "party_parrot",
            "fixture_type": fixture.fixture_type,
            "group_name": fixture.group_name,
            "universe_name": fixture.universe_name,
        },
    }
