"""The FT232R/Open-DMX transport proven in Party Parrot.

The adapter has no DMX frame processor. Lumen therefore owns timing: it emits
the break, mark-after-break, start code, and full universe continuously on a
dedicated thread while the rest of the engine updates an in-memory frame.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import glob
from pathlib import Path
import threading
import time
from typing import Protocol

from lumen_engine.dmx import DMXFrame

FTDI_VENDOR_ID = 0x0403
FT232R_PRODUCT_ID = 0x6001
DMX_BAUD = 250_000
DMX_BREAK_S = 0.000120
DMX_MARK_AFTER_BREAK_S = 0.000012


class OpenDmxBackend(Protocol):
    description: str

    def write_frame(self, frame: bytes) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenDmxStatus:
    backend: str
    universe: int
    frame_rate_hz: float
    frames_sent: int
    control_updates: int
    content_changes: int
    last_control_update_age_ms: float | None
    last_content_change_age_ms: float | None
    last_error: str | None


class LibFtdiBackend:
    """Native libftdi transport used by Party Parrot on this Ubuntu PC."""

    def __init__(
        self,
        vendor_id: int = FTDI_VENDOR_ID,
        product_id: int = FT232R_PRODUCT_ID,
        library: object | None = None,
    ) -> None:
        library_path = ctypes.util.find_library("ftdi1") or "libftdi1.so.2"
        self._libftdi = library or ctypes.CDLL(library_path)
        self._configure_signatures()
        self._context = self._libftdi.ftdi_new()
        self._usb_open = False
        if not self._context:
            raise RuntimeError("libftdi could not allocate an FTDI context")
        try:
            # Reattach Linux's ftdi_sio driver on close, matching the corrected
            # Party Parrot behavior that avoids a disappearing /dev/ttyUSB path.
            self._check(
                self._libftdi.ftdi_set_module_detach_mode(self._context, 2),
                "set FTDI kernel-driver reattach mode",
            )
            self._check(
                self._libftdi.ftdi_usb_open(
                    self._context, int(vendor_id), int(product_id)
                ),
                "open FTDI USB device",
            )
            self._usb_open = True
            self._check(
                self._libftdi.ftdi_set_latency_timer(self._context, 1),
                "set FTDI latency timer",
            )
            self._check(
                self._libftdi.ftdi_set_baudrate(self._context, DMX_BAUD),
                "set DMX baud rate",
            )
            self._check(
                self._libftdi.ftdi_set_line_property2(
                    self._context, 8, 2, 0, 0
                ),
                "set 8N2 line format",
            )
            self._check(
                self._libftdi.ftdi_usb_purge_buffers(self._context),
                "purge FTDI buffers",
            )
        except Exception:
            self.close()
            raise
        self.description = (
            f"libftdi Open-DMX {vendor_id:04x}:{product_id:04x}, "
            "250000 baud, full 512 slots"
        )

    def _configure_signatures(self) -> None:
        lib = self._libftdi
        # A fake library in unit tests may expose Python methods rather than
        # ctypes functions, so signature assignment is best-effort.
        signatures = {
            "ftdi_new": (None, ctypes.c_void_p),
            "ftdi_set_module_detach_mode": (
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_int,
            ),
            "ftdi_usb_open": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                ctypes.c_int,
            ),
            "ftdi_set_latency_timer": (
                [ctypes.c_void_p, ctypes.c_ubyte],
                ctypes.c_int,
            ),
            "ftdi_set_baudrate": (
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_int,
            ),
            "ftdi_set_line_property2": (
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                ctypes.c_int,
            ),
            "ftdi_usb_purge_buffers": ([ctypes.c_void_p], ctypes.c_int),
            "ftdi_write_data": (
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_ubyte),
                    ctypes.c_int,
                ],
                ctypes.c_int,
            ),
            "ftdi_usb_close": ([ctypes.c_void_p], ctypes.c_int),
            "ftdi_free": ([ctypes.c_void_p], None),
            "ftdi_get_error_string": ([ctypes.c_void_p], ctypes.c_char_p),
        }
        for name, (argument_types, result_type) in signatures.items():
            function = getattr(lib, name)
            try:
                if argument_types is not None:
                    function.argtypes = argument_types
                function.restype = result_type
            except (AttributeError, TypeError):
                pass

    def _error_string(self) -> str:
        if not self._context:
            return "no FTDI context"
        raw = self._libftdi.ftdi_get_error_string(self._context)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw or "unknown error")

    def _check(self, result: int, operation: str) -> int:
        if result < 0:
            raise OSError(f"{operation}: {self._error_string()} ({result})")
        return result

    def write_frame(self, frame: bytes) -> None:
        if len(frame) != 512:
            raise ValueError("Open-DMX native output requires exactly 512 slots")
        self._check(
            self._libftdi.ftdi_set_line_property2(
                self._context, 8, 2, 0, 1
            ),
            "start DMX break",
        )
        time.sleep(DMX_BREAK_S)
        self._check(
            self._libftdi.ftdi_set_line_property2(
                self._context, 8, 2, 0, 0
            ),
            "end DMX break",
        )
        time.sleep(DMX_MARK_AFTER_BREAK_S)
        payload = b"\x00" + frame
        buffer = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        offset = 0
        while offset < len(payload):
            pointer = ctypes.cast(
                ctypes.byref(buffer, offset), ctypes.POINTER(ctypes.c_ubyte)
            )
            written = self._check(
                self._libftdi.ftdi_write_data(
                    self._context, pointer, len(payload) - offset
                ),
                "write DMX frame",
            )
            if written == 0:
                raise OSError("write DMX frame: libftdi wrote zero bytes")
            offset += written

    def close(self) -> None:
        context = getattr(self, "_context", None)
        if not context:
            return
        if self._usb_open:
            self._libftdi.ftdi_usb_close(context)
            self._usb_open = False
        self._libftdi.ftdi_free(context)
        self._context = None


class SerialOpenDmxBackend:
    """Diagnostic tty fallback from Party Parrot; libftdi is preferred."""

    def __init__(self, port: str, serial_module: object | None = None) -> None:
        if serial_module is None:
            try:
                import serial as serial_module  # type: ignore[no-redef]
            except ImportError as error:
                raise RuntimeError(
                    "tty Open-DMX requires pyserial; native libftdi needs no "
                    "Python package"
                ) from error
        self._serial_module = serial_module
        self.serial = serial_module.Serial(
            port=port,
            baudrate=DMX_BAUD,
            bytesize=serial_module.EIGHTBITS,
            parity=serial_module.PARITY_NONE,
            stopbits=serial_module.STOPBITS_TWO,
            timeout=0,
            write_timeout=1,
        )
        self.serial.reset_output_buffer()
        self.description = f"tty Open-DMX {port}, 250000 baud, full 512 slots"

    def write_frame(self, frame: bytes) -> None:
        if len(frame) != 512:
            raise ValueError("Open-DMX tty output requires exactly 512 slots")
        self.serial.break_condition = True
        time.sleep(DMX_BREAK_S)
        self.serial.break_condition = False
        time.sleep(DMX_MARK_AFTER_BREAK_S)
        self.serial.write(b"\x00" + frame)
        self.serial.flush()

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()


class OpenDmxUsbOutput:
    """Continuously transmit the latest Lumen universe through Open-DMX."""

    def __init__(
        self,
        backend: OpenDmxBackend,
        universe: int = 0,
        frame_rate_hz: float = 40.0,
        *,
        start_thread: bool = True,
    ) -> None:
        if universe < 0:
            raise ValueError("universe must be non-negative")
        if not 1.0 <= frame_rate_hz <= 44.0:
            raise ValueError("Open-DMX frame rate must be in [1, 44] Hz")
        self.backend = backend
        self.universe = universe
        self.frame_rate_hz = frame_rate_hz
        self._frame = bytes(512)
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self.frames_sent = 0
        self.control_updates = 0
        self.content_changes = 0
        self._last_control_update_at: float | None = None
        self._last_content_change_at: float | None = None
        self.last_error: Exception | None = None
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._transmit_forever,
                name="lumen-open-dmx-transmitter",
                daemon=True,
            )
            self._thread.start()

    @classmethod
    def open(
        cls,
        *,
        driver: str = "native",
        port: str | None = None,
        universe: int = 0,
        frame_rate_hz: float = 40.0,
    ) -> "OpenDmxUsbOutput":
        normalized = driver.strip().lower()
        if normalized in ("native", "libftdi", "open_dmx"):
            backend: OpenDmxBackend = LibFtdiBackend()
        elif normalized in ("tty", "serial", "open_dmx_tty"):
            resolved_port = port or first_ftdi_tty()
            if not resolved_port:
                raise RuntimeError("no /dev/ttyUSB* or /dev/ttyACM* device found")
            backend = SerialOpenDmxBackend(resolved_port)
        else:
            raise ValueError("driver must be native or tty")
        return cls(
            backend,
            universe=universe,
            frame_rate_hz=frame_rate_hz,
        )

    def send(self, frame: DMXFrame) -> None:
        data = frame.universe_data(self.universe)
        now = time.monotonic()
        with self._frame_lock:
            changed = data != self._frame
            self._frame = data
            self.control_updates += 1
            self._last_control_update_at = now
            if changed:
                self.content_changes += 1
                self._last_content_change_at = now

    def blackout(self) -> None:
        with self._frame_lock:
            self._frame = bytes(512)

    def _transmit_once(self) -> None:
        with self._frame_lock:
            frame = self._frame
        self.backend.write_frame(frame)
        self.frames_sent += 1
        self.last_error = None

    def _transmit_forever(self) -> None:
        period = 1.0 / self.frame_rate_hz
        next_frame_at = time.monotonic()
        while not self._stop.is_set():
            try:
                self._transmit_once()
            except Exception as error:
                self.last_error = error
                self._stop.wait(0.1)
            next_frame_at += period
            delay = next_frame_at - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_frame_at = time.monotonic()

    @property
    def status(self) -> OpenDmxStatus:
        now = time.monotonic()
        with self._frame_lock:
            control_updates = self.control_updates
            content_changes = self.content_changes
            last_control_update_at = self._last_control_update_at
            last_content_change_at = self._last_content_change_at
        return OpenDmxStatus(
            backend=self.backend.description,
            universe=self.universe,
            frame_rate_hz=self.frame_rate_hz,
            frames_sent=self.frames_sent,
            control_updates=control_updates,
            content_changes=content_changes,
            last_control_update_age_ms=(
                None
                if last_control_update_at is None
                else max(0.0, (now - last_control_update_at) * 1000.0)
            ),
            last_content_change_age_ms=(
                None
                if last_content_change_at is None
                else max(0.0, (now - last_content_change_at) * 1000.0)
            ),
            last_error=None if self.last_error is None else str(self.last_error),
        )

    def close(self) -> None:
        self.blackout()
        if self._thread is not None and self._thread.is_alive():
            # Give the transmitter one normal period to publish blackout.
            self._stop.wait(1.0 / self.frame_rate_hz)
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self.backend.close()


def first_ftdi_tty() -> str | None:
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def describe_open_dmx_environment() -> dict[str, object]:
    library = ctypes.util.find_library("ftdi1")
    tty_devices = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    sysfs_ftdi: list[dict[str, str]] = []
    for vendor_path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            vendor = vendor_path.read_text(encoding="ascii").strip().lower()
            product = (vendor_path.parent / "idProduct").read_text(
                encoding="ascii"
            ).strip().lower()
        except OSError:
            continue
        if vendor == "0403" and product == "6001":
            description = ""
            product_path = vendor_path.parent / "product"
            if product_path.exists():
                try:
                    description = product_path.read_text(
                        encoding="utf-8"
                    ).strip()
                except OSError:
                    pass
            sysfs_ftdi.append(
                {
                    "usb_path": str(vendor_path.parent),
                    "vendor_id": vendor,
                    "product_id": product,
                    "description": description,
                }
            )
    return {
        "libftdi1": library,
        "tty_devices": tty_devices,
        "ft232r_devices": sysfs_ftdi,
        "native_driver_ready": library is not None,
    }
