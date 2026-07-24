"""Load and validate human-editable rig configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from lumen_engine.models import (
    EulerXYZ,
    FixtureCalibration,
    FixturePatch,
    Vec3,
)


@dataclass(frozen=True, slots=True)
class RoomConfig:
    width_m: float
    depth_m: float
    height_m: float
    origin_description: str

    def __post_init__(self) -> None:
        if min(self.width_m, self.depth_m, self.height_m) <= 0:
            raise ValueError("room dimensions must be positive")


@dataclass(frozen=True, slots=True)
class RigConfig:
    name: str
    room: RoomConfig
    fixtures: tuple[FixturePatch, ...]


def load_rig(path: str | Path) -> RigConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rig configuration must be a JSON object")
    return rig_from_dict(payload)


def rig_from_dict(payload: dict[str, Any]) -> RigConfig:
    room_payload = payload["room"]
    room = RoomConfig(
        width_m=float(room_payload["width_m"]),
        depth_m=float(room_payload["depth_m"]),
        height_m=float(room_payload["height_m"]),
        origin_description=str(
            room_payload.get(
                "origin_description", "floor center at the front of the room"
            )
        ),
    )
    fixtures = tuple(_fixture_from_dict(item) for item in payload["fixtures"])
    identifiers = [fixture.fixture_id for fixture in fixtures]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fixture IDs must be unique")
    _validate_patch_collisions(fixtures)
    return RigConfig(
        name=str(payload.get("name", "Unnamed rig")),
        room=room,
        fixtures=fixtures,
    )


def _fixture_from_dict(payload: dict[str, Any]) -> FixturePatch:
    calibration_payload = payload["calibration"]
    channels = payload.get("channels", {})
    return FixturePatch(
        fixture_id=str(payload["id"]),
        name=str(payload.get("name", payload["id"])),
        universe=int(payload.get("universe", 0)),
        address=int(payload["address"]),
        position_m=_vec3(payload["position_m"]),
        housing_rotation=_euler(payload.get("housing_rotation_deg", [0, 0, 0])),
        calibration=FixtureCalibration(
            pan_min_deg=float(calibration_payload["pan_min_deg"]),
            pan_max_deg=float(calibration_payload["pan_max_deg"]),
            tilt_min_deg=float(calibration_payload["tilt_min_deg"]),
            tilt_max_deg=float(calibration_payload["tilt_max_deg"]),
            pan_offset_deg=float(calibration_payload.get("pan_offset_deg", 0)),
            tilt_offset_deg=float(calibration_payload.get("tilt_offset_deg", 0)),
            pan_direction=int(calibration_payload.get("pan_direction", 1)),
            tilt_direction=int(calibration_payload.get("tilt_direction", 1)),
            pan_invert_dmx=bool(
                calibration_payload.get("pan_invert_dmx", False)
            ),
            tilt_invert_dmx=bool(
                calibration_payload.get("tilt_invert_dmx", False)
            ),
            max_pan_speed_deg_s=float(
                calibration_payload.get("max_pan_speed_deg_s", 180)
            ),
            max_tilt_speed_deg_s=float(
                calibration_payload.get("max_tilt_speed_deg_s", 180)
            ),
        ),
        pan_coarse_channel=int(channels.get("pan_coarse", 1)),
        pan_fine_channel=_optional_int(channels.get("pan_fine", 2)),
        tilt_coarse_channel=int(channels.get("tilt_coarse", 3)),
        tilt_fine_channel=_optional_int(channels.get("tilt_fine", 4)),
        dimmer_channel=_optional_int(channels.get("dimmer", 5)),
    )


def _vec3(value: list[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("a position must contain [x, y, z]")
    return Vec3(*(float(component) for component in value))


def _euler(value: list[float]) -> EulerXYZ:
    if len(value) != 3:
        raise ValueError("a rotation must contain [x, y, z]")
    return EulerXYZ(*(float(component) for component in value))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _validate_patch_collisions(fixtures: tuple[FixturePatch, ...]) -> None:
    used: dict[tuple[int, int], str] = {}
    for fixture in fixtures:
        relatives = (
            fixture.pan_coarse_channel,
            fixture.pan_fine_channel,
            fixture.tilt_coarse_channel,
            fixture.tilt_fine_channel,
            fixture.dimmer_channel,
        )
        for relative in relatives:
            if relative is None:
                continue
            channel = fixture.address + relative - 1
            if not 1 <= channel <= 512:
                raise ValueError(
                    f"{fixture.fixture_id} channel {channel} is outside its universe"
                )
            key = fixture.universe, channel
            if key in used:
                raise ValueError(
                    f"DMX collision: {fixture.fixture_id} and {used[key]} both use "
                    f"universe {fixture.universe}, channel {channel}"
                )
            used[key] = fixture.fixture_id

