"""DMX frame construction and isolated output adapters."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import threading
import time
from typing import Protocol

from lumen_engine.models import FixturePatch, clamp
from lumen_engine.spatial import TargetingSolution

DMX_CHANNELS = 512


class DMXFrame:
    def __init__(self) -> None:
        self._universes: dict[int, bytearray] = {}

    def set_channel(self, universe: int, channel: int, value: int) -> None:
        if universe < 0:
            raise ValueError("universe must be non-negative")
        if not 1 <= channel <= DMX_CHANNELS:
            raise ValueError("channel must be in [1, 512]")
        if not 0 <= value <= 255:
            raise ValueError("DMX value must be in [0, 255]")
        data = self._universes.setdefault(universe, bytearray(DMX_CHANNELS))
        data[channel - 1] = value

    def get_channel(self, universe: int, channel: int) -> int:
        if not 1 <= channel <= DMX_CHANNELS:
            raise ValueError("channel must be in [1, 512]")
        data = self._universes.get(universe)
        return 0 if data is None else data[channel - 1]

    def universe_data(self, universe: int) -> bytes:
        return bytes(self._universes.get(universe, bytearray(DMX_CHANNELS)))

    @property
    def universes(self) -> tuple[int, ...]:
        return tuple(sorted(self._universes))

    def copy(self) -> "DMXFrame":
        copied = DMXFrame()
        copied._universes = {
            universe: bytearray(data) for universe, data in self._universes.items()
        }
        return copied


def _angle_to_u16(value: float, minimum: float, maximum: float, invert: bool) -> int:
    normalized = clamp((value - minimum) / (maximum - minimum), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return round(normalized * 65535.0)


def _set_u16(
    frame: DMXFrame,
    universe: int,
    address: int,
    coarse_relative: int,
    fine_relative: int | None,
    value: int,
) -> None:
    frame.set_channel(universe, address + coarse_relative - 1, value >> 8)
    if fine_relative is not None:
        frame.set_channel(universe, address + fine_relative - 1, value & 0xFF)


def apply_moving_head_solution(
    frame: DMXFrame,
    fixture: FixturePatch,
    solution: TargetingSolution,
    brightness: float,
) -> None:
    calibration = fixture.calibration
    pan = _angle_to_u16(
        solution.pan_deg,
        calibration.pan_min_deg,
        calibration.pan_max_deg,
        calibration.pan_invert_dmx,
    )
    tilt = _angle_to_u16(
        solution.tilt_deg,
        calibration.tilt_min_deg,
        calibration.tilt_max_deg,
        calibration.tilt_invert_dmx,
    )
    _set_u16(
        frame,
        fixture.universe,
        fixture.address,
        fixture.pan_coarse_channel,
        fixture.pan_fine_channel,
        pan,
    )
    _set_u16(
        frame,
        fixture.universe,
        fixture.address,
        fixture.tilt_coarse_channel,
        fixture.tilt_fine_channel,
        tilt,
    )
    if fixture.dimmer_channel is not None:
        frame.set_channel(
            fixture.universe,
            fixture.address + fixture.dimmer_channel - 1,
            round(clamp(brightness, 0.0, 1.0) * 255.0),
        )


class DMXOutput(Protocol):
    def send(self, frame: DMXFrame) -> None: ...

    def close(self) -> None: ...


class VirtualDMXOutput:
    """Safe default output that stores frames without touching hardware."""

    def __init__(self) -> None:
        self.last_frame = DMXFrame()
        self.frame_count = 0
        self.last_sent_monotonic: float | None = None

    def send(self, frame: DMXFrame) -> None:
        self.last_frame = frame.copy()
        self.frame_count += 1
        self.last_sent_monotonic = time.monotonic()

    def close(self) -> None:
        return


class OutputSafetyGate:
    """Physical output requires explicit arming and a live heartbeat."""

    def __init__(self, output: DMXOutput, watchdog_timeout_s: float = 1.0) -> None:
        if watchdog_timeout_s <= 0:
            raise ValueError("watchdog_timeout_s must be positive")
        self.output = output
        self.watchdog_timeout_s = watchdog_timeout_s
        self._armed = False
        self._last_heartbeat = 0.0
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self._last_heartbeat = time.monotonic()

    def disarm(self, blackout: bool = True) -> None:
        with self._lock:
            self._armed = False
            if blackout:
                self.output.send(DMXFrame())

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()

    def send(self, frame: DMXFrame) -> bool:
        with self._lock:
            alive = (
                self._armed
                and time.monotonic() - self._last_heartbeat
                <= self.watchdog_timeout_s
            )
            if not alive:
                return False
            self.output.send(frame)
            return True

    def close(self) -> None:
        self.disarm(blackout=True)
        self.output.close()


class ArtNetOutput:
    """Minimal ArtDMX sender.

    Constructing this class does not send anything. Place it behind
    OutputSafetyGate and arm it explicitly before use.
    """

    def __init__(self, host: str, port: int = 6454) -> None:
        self.host = host
        self.port = port
        self._sequence = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame: DMXFrame) -> None:
        for universe in frame.universes:
            self._sequence = (self._sequence % 255) + 1
            data = frame.universe_data(universe)
            header = (
                b"Art-Net\x00"
                + struct.pack("<H", 0x5000)
                + struct.pack(">H", 14)
                + bytes((self._sequence, 0))
                + struct.pack("<H", universe)
                + struct.pack(">H", len(data))
            )
            self._socket.sendto(header + data, (self.host, self.port))

    def close(self) -> None:
        self._socket.close()

