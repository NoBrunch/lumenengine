"""Core data types shared across Lumen Engine subsystems.

Lumen uses one room coordinate system internally:

* X: room left to room right
* Y: front of room to back of room, with the floor center at zero
* Z: floor to ceiling

Distances are meters, angles are degrees at fixture boundaries, timestamps are
monotonic seconds for live events, and DMX channels are one-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        length = self.norm()
        if length <= 1e-12:
            raise ValueError("Cannot normalize a zero-length vector")
        return self * (1.0 / length)

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).norm()

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class EulerXYZ:
    """Housing rotation in intrinsic XYZ order, stored in degrees."""

    x_deg: float = 0.0
    y_deg: float = 0.0
    z_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class FixtureCalibration:
    """Mechanical and DMX behavior of one moving-head fixture.

    Mathematical pan zero points along fixture-local +X, mathematical tilt zero
    lies in the fixture-local XY plane, and positive tilt points toward local +Z.
    Offsets and directions map that canonical sphere onto a real fixture.
    """

    pan_min_deg: float
    pan_max_deg: float
    tilt_min_deg: float
    tilt_max_deg: float
    pan_offset_deg: float = 0.0
    tilt_offset_deg: float = 0.0
    pan_direction: int = 1
    tilt_direction: int = 1
    pan_invert_dmx: bool = False
    tilt_invert_dmx: bool = False
    # The reachable mechanical window may cover only part of the fixture's
    # full DMX travel. These endpoints preserve that calibrated subrange.
    pan_dmx_min_u16: int = 0
    pan_dmx_max_u16: int = 65535
    tilt_dmx_min_u16: int = 0
    tilt_dmx_max_u16: int = 65535
    max_pan_speed_deg_s: float = 180.0
    max_tilt_speed_deg_s: float = 180.0

    def __post_init__(self) -> None:
        if self.pan_min_deg >= self.pan_max_deg:
            raise ValueError("pan_min_deg must be less than pan_max_deg")
        if self.tilt_min_deg >= self.tilt_max_deg:
            raise ValueError("tilt_min_deg must be less than tilt_max_deg")
        if self.pan_direction not in (-1, 1):
            raise ValueError("pan_direction must be -1 or 1")
        if self.tilt_direction not in (-1, 1):
            raise ValueError("tilt_direction must be -1 or 1")
        if self.max_pan_speed_deg_s <= 0 or self.max_tilt_speed_deg_s <= 0:
            raise ValueError("fixture speeds must be positive")
        for field_name in (
            "pan_dmx_min_u16",
            "pan_dmx_max_u16",
            "tilt_dmx_min_u16",
            "tilt_dmx_max_u16",
        ):
            value = getattr(self, field_name)
            if not 0 <= value <= 65535:
                raise ValueError(f"{field_name} must be in [0, 65535]")
        if self.pan_dmx_min_u16 == self.pan_dmx_max_u16:
            raise ValueError("pan DMX endpoints must differ")
        if self.tilt_dmx_min_u16 == self.tilt_dmx_max_u16:
            raise ValueError("tilt DMX endpoints must differ")


@dataclass(frozen=True, slots=True)
class FixturePatch:
    fixture_id: str
    name: str
    universe: int
    address: int
    position_m: Vec3
    housing_rotation: EulerXYZ
    calibration: FixtureCalibration
    profile_key: str = "generic_moving_head"
    source_metadata: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )
    pan_coarse_channel: int = 1
    pan_fine_channel: int | None = 2
    tilt_coarse_channel: int = 3
    tilt_fine_channel: int | None = 4
    dimmer_channel: int | None = 5

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("fixture_id must not be empty")
        if self.universe < 0:
            raise ValueError("universe must be non-negative")
        if not 1 <= self.address <= 512:
            raise ValueError("address must be in [1, 512]")


@dataclass(frozen=True, slots=True)
class ProfileFixturePatch:
    """A patched non-conventional fixture driven by a declarative profile."""

    fixture_id: str
    name: str
    profile_key: str
    universe: int
    address: int
    position_m: Vec3
    housing_rotation: EulerXYZ
    options: dict[str, Any] = field(default_factory=dict, compare=False)
    source_metadata: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("fixture_id must not be empty")
        if self.universe < 0:
            raise ValueError("universe must be non-negative")
        if not 1 <= self.address <= 512:
            raise ValueError("address must be in [1, 512]")


@dataclass(frozen=True, slots=True)
class MediaIdentity:
    provider: str
    provider_item_id: str | None
    title: str | None
    artists: tuple[str, ...] = ()
    album: str | None = None
    duration_ms: int | None = None
    observed_position_ms: int | None = None
    observed_at_unix_ms: int | None = None
    is_playing: bool = False
    device_name: str | None = None
    context_uri: str | None = None
    confidence: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def display_name(self) -> str:
        artist = ", ".join(self.artists)
        if artist and self.title:
            return f"{artist} — {self.title}"
        return self.title or artist or "Unknown recording"


@dataclass(frozen=True, slots=True)
class MusicalObservation:
    timestamp_s: float
    loudness: float
    onset_strength: float
    low_energy: float
    mid_energy: float
    high_energy: float
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    beat_pulse: float = 0.0
    beat_confidence: float = 0.0
    bpm: float | None = None
    section: str | None = None
    section_confidence: float = 0.0
    novelty: float = 0.0

    def __post_init__(self) -> None:
        bounded = (
            "loudness",
            "onset_strength",
            "low_energy",
            "mid_energy",
            "high_energy",
            "beat_phase",
            "bar_phase",
            "beat_pulse",
            "beat_confidence",
            "section_confidence",
            "novelty",
        )
        for name in bounded:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class ExpressionState:
    energy: float = 0.0
    tension: float = 0.0
    motion: float = 0.0
    intimacy: float = 0.5
    confidence: float = 0.0


class Gesture(StrEnum):
    HOLD = "hold"
    BREATHE = "breathe"
    CONVERGE = "converge"
    EXPAND = "expand"
    SWEEP = "sweep"
    PULSE = "pulse"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class PerformanceDecision:
    timestamp_s: float
    gesture: Gesture
    expression: ExpressionState
    target: Vec3
    brightness: float
    reason: str
    confidence: float
    palette_hint: str = "auto"
    # A phrase-level semantic routine.  This is intentionally separate from
    # the broad gesture so the resolver can keep one musical idea coherent for
    # several bars while still changing individual beat accents.
    routine: str = "auto"


@dataclass(frozen=True, slots=True)
class Feedback:
    song_id: int
    position_ms: int | None
    label: str
    value: float
    note: str | None = None
    scope: str = "overall"
    fixture_id: str | None = None
    # Snapshot of what Lumen was doing when the operator pressed the control.
    # These fields make a feedback moment explainable and learnable rather than
    # a timeless scalar preference.
    gesture: str | None = None
    section: str | None = None
    energy: float | None = None
    motion: float | None = None
    tension: float | None = None
    confidence: float | None = None
    bpm: float | None = None
    routine: str | None = None
