from __future__ import annotations

import unittest

from lumen_engine.dmx import DMXFrame
from lumen_engine.usb_dmx import OpenDmxUsbOutput, SerialOpenDmxBackend


class FakeBackend:
    description = "fake FT232R"

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.closed = False

    def write_frame(self, frame: bytes) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class OpenDmxOutputTests(unittest.TestCase):
    def test_transmits_start_code_slots_as_backend_frame(self) -> None:
        backend = FakeBackend()
        output = OpenDmxUsbOutput(backend, start_thread=False)
        frame = DMXFrame()
        frame.set_channel(0, 1, 12)
        frame.set_channel(0, 19, 255)
        frame.set_channel(0, 512, 64)
        output.send(frame)
        output._transmit_once()
        self.assertEqual(len(backend.frames[0]), 512)
        self.assertEqual(backend.frames[0][0], 12)
        self.assertEqual(backend.frames[0][18], 255)
        self.assertEqual(backend.frames[0][511], 64)
        self.assertEqual(output.status.frames_sent, 1)
        output.close()
        self.assertTrue(backend.closed)

    def test_selected_universe_isolated(self) -> None:
        backend = FakeBackend()
        output = OpenDmxUsbOutput(backend, universe=1, start_thread=False)
        frame = DMXFrame()
        frame.set_channel(0, 1, 10)
        frame.set_channel(1, 1, 20)
        output.send(frame)
        output._transmit_once()
        self.assertEqual(backend.frames[0][0], 20)
        output.close()

    def test_tty_backend_matches_party_parrot_framing(self) -> None:
        class Port:
            is_open = True
            break_condition = False

            def __init__(self) -> None:
                self.writes: list[bytes] = []
                self.flushes = 0

            def reset_output_buffer(self) -> None:
                return

            def write(self, payload: bytes) -> None:
                self.writes.append(payload)

            def flush(self) -> None:
                self.flushes += 1

            def close(self) -> None:
                self.is_open = False

        class SerialModule:
            EIGHTBITS = 8
            PARITY_NONE = "N"
            STOPBITS_TWO = 2

            def __init__(self) -> None:
                self.port = Port()
                self.arguments = {}

            def Serial(self, **arguments):  # noqa: N802
                self.arguments = arguments
                return self.port

        serial_module = SerialModule()
        backend = SerialOpenDmxBackend("/dev/ttyUSB1", serial_module=serial_module)
        frame = bytearray(512)
        frame[0] = 12
        frame[18] = 255
        backend.write_frame(bytes(frame))
        self.assertEqual(serial_module.arguments["baudrate"], 250_000)
        self.assertEqual(serial_module.arguments["bytesize"], 8)
        self.assertEqual(serial_module.arguments["parity"], "N")
        self.assertEqual(serial_module.arguments["stopbits"], 2)
        payload = serial_module.port.writes[0]
        self.assertEqual(len(payload), 513)
        self.assertEqual(payload[0], 0)
        self.assertEqual(payload[1], 12)
        self.assertEqual(payload[19], 255)
        self.assertEqual(serial_module.port.flushes, 1)
        backend.close()


if __name__ == "__main__":
    unittest.main()
