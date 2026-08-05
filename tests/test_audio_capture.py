from __future__ import annotations

import threading
import time
import unittest

from lumen_engine.audio import (
    AudioCaptureConfig,
    ContinuouslyDrainedAudio,
    SourcePcm,
)


class _GatedAudioSource:
    def __init__(self, packets: list[bytes]) -> None:
        self.config = AudioCaptureConfig(
            sample_rate=10,
            channels=1,
            chunk_frames=2,
        )
        self.packets = packets
        self.release_after_first = threading.Event()
        self.drained = threading.Event()
        self.closed = threading.Event()

    def chunks(self):
        yield self.packets[0]
        self.release_after_first.wait(timeout=2.0)
        for packet in self.packets[1:]:
            if self.closed.is_set():
                break
            yield packet
        self.drained.set()

    def close(self) -> None:
        self.closed.set()
        self.release_after_first.set()


class _BlockingAudioSource:
    def __init__(self) -> None:
        self.config = AudioCaptureConfig(
            sample_rate=10,
            channels=1,
            chunk_frames=2,
        )
        self.closed = threading.Event()

    def chunks(self):
        self.closed.wait(timeout=5.0)
        if False:
            yield b""

    def close(self) -> None:
        self.closed.set()


class _DelayedStartSource:
    def __init__(self) -> None:
        self.config = AudioCaptureConfig(
            sample_rate=100,
            channels=1,
            chunk_frames=10,
        )

    def chunks(self):
        time.sleep(0.12)
        yield bytes(20)

    def close(self) -> None:
        pass


class _OverflowPauseSource:
    def __init__(self) -> None:
        self.config = AudioCaptureConfig(
            sample_rate=100,
            channels=1,
            chunk_frames=2,
        )
        self.release_burst = threading.Event()
        self.burst_complete = threading.Event()
        self.release_finish = threading.Event()
        self.closed = threading.Event()

    def chunks(self):
        yield bytes(4)
        self.release_burst.wait(timeout=2.0)
        for index in range(1, 10):
            yield bytes([index, 0, index, 0])
        self.burst_complete.set()
        self.release_finish.wait(timeout=2.0)

    def close(self) -> None:
        self.closed.set()
        self.release_burst.set()
        self.release_finish.set()


class _UncooperativeAudioSource:
    def __init__(self) -> None:
        self.config = AudioCaptureConfig(
            sample_rate=100,
            channels=1,
            chunk_frames=2,
        )
        self.started = threading.Event()
        self.close_called = threading.Event()
        self.release = threading.Event()

    def chunks(self):
        self.started.set()
        self.release.wait(timeout=5.0)
        if False:
            yield b""

    def close(self) -> None:
        self.close_called.set()


class ContinuouslyDrainedAudioTests(unittest.TestCase):
    def test_default_queue_absorbs_one_minute_without_large_ram_cost(self):
        capture = ContinuouslyDrainedAudio(_BlockingAudioSource())
        try:
            self.assertEqual(capture.diagnostics["queue_capacity"], 300)
            self.assertAlmostEqual(
                capture.diagnostics["buffer_seconds"], 60.0
            )
        finally:
            capture.close()

    def test_default_queue_absorbs_a_consumer_pause_without_audio_loss(self):
        source = _OverflowPauseSource()
        capture = ContinuouslyDrainedAudio(source)
        iterator = capture.chunks()
        try:
            self.assertEqual(next(iterator).source_start_frame, 0)
            source.release_burst.set()
            self.assertTrue(source.burst_complete.wait(timeout=1.0))
            diagnostics = capture.diagnostics
            self.assertEqual(diagnostics["dropped_packets"], 0)
            self.assertEqual(diagnostics["queue_depth"], 9)
            source.release_finish.set()
            self.assertEqual(
                [packet.source_start_frame for packet in iterator],
                list(range(2, 20, 2)),
            )
        finally:
            source.release_finish.set()
            capture.close()

    def test_sample_clock_starts_when_first_alsa_packet_arrives(self):
        capture = ContinuouslyDrainedAudio(_DelayedStartSource())
        started = time.monotonic()
        try:
            packet = next(capture.chunks())
            self.assertGreater(packet.timestamp_s - started, 0.05)
            self.assertLess(packet.timestamp_s - started, 0.20)
            self.assertAlmostEqual(
                packet.timestamp_s,
                capture.diagnostics["sample_clock_origin_s"] + 0.05,
                places=4,
            )
        finally:
            capture.close()

    def test_reader_never_backpressures_and_drops_oldest_with_exact_clock(self):
        packets = [bytes([index, 0, index, 0]) for index in range(6)]
        source = _GatedAudioSource(packets)
        capture = ContinuouslyDrainedAudio(source, max_chunks=2)
        iterator = capture.chunks()
        try:
            first = next(iterator)
            self.assertEqual(first.source_start_frame, 0)
            self.assertEqual(first.frame_count, 2)
            self.assertIsInstance(first.pcm, SourcePcm)
            self.assertEqual(first.pcm.source_start_frame, 0)

            # Leave the consumer paused. The dedicated reader must still drain
            # the entire source instead of blocking when its queue fills.
            source.release_after_first.set()
            self.assertTrue(source.drained.wait(timeout=1.0))

            remaining = list(iterator)
            self.assertEqual(
                [item.source_start_frame for item in remaining], [8, 10]
            )
            self.assertAlmostEqual(
                remaining[1].timestamp_s - remaining[0].timestamp_s,
                0.2,
                places=6,
            )
            diagnostics = capture.diagnostics
            self.assertEqual(diagnostics["packets_read"], 6)
            self.assertEqual(diagnostics["source_frames"], 12)
            self.assertEqual(diagnostics["dropped_packets"], 3)
            self.assertEqual(diagnostics["dropped_frames"], 6)
            self.assertEqual(
                diagnostics["dropped_ranges"],
                [
                    {
                        "start_frame": 2,
                        "frame_count": 6,
                        "reason": "capture_queue_overflow",
                    }
                ],
            )
        finally:
            capture.close()

    def test_overflow_collapses_backlog_to_live_low_water(self):
        source = _OverflowPauseSource()
        capture = ContinuouslyDrainedAudio(
            source,
            max_chunks=8,
            overflow_low_water_chunks=2,
        )
        iterator = capture.chunks()
        try:
            self.assertEqual(next(iterator).source_start_frame, 0)
            source.release_burst.set()
            self.assertTrue(source.burst_complete.wait(timeout=1.0))
            diagnostics = capture.diagnostics
            # Eight queued packets plus one arrival caused one atomic collapse:
            # six old packets were discarded and two recent packets retained
            # before the newest packet was appended.
            self.assertEqual(diagnostics["queue_depth"], 3)
            self.assertEqual(diagnostics["dropped_packets"], 6)
            self.assertEqual(
                diagnostics["dropped_ranges"],
                [{
                    "start_frame": 2,
                    "frame_count": 12,
                    "reason": "capture_queue_overflow",
                }],
            )
            source.release_finish.set()
            self.assertEqual(
                [packet.source_start_frame for packet in iterator],
                [14, 16, 18],
            )
        finally:
            source.release_finish.set()
            capture.close()

    def test_close_surfaces_a_reader_that_did_not_stop(self):
        source = _UncooperativeAudioSource()
        capture = ContinuouslyDrainedAudio(source)
        capture.start()
        self.assertTrue(source.started.wait(timeout=1.0))
        try:
            with self.assertRaisesRegex(
                RuntimeError, "reader did not stop"
            ):
                capture.close(timeout=0.02)
            self.assertTrue(source.close_called.is_set())
            self.assertTrue(capture.diagnostics["reader_alive"])
        finally:
            source.release.set()
            capture.close(timeout=1.0)

    def test_consumer_stop_event_interrupts_an_empty_capture(self):
        source = _BlockingAudioSource()
        capture = ContinuouslyDrainedAudio(source, max_chunks=2)
        stop = threading.Event()
        finished = threading.Event()

        def consume() -> None:
            list(capture.chunks(stop_event=stop))
            finished.set()

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        time.sleep(0.05)
        stop.set()
        self.assertTrue(finished.wait(timeout=1.0))
        self.assertTrue(source.closed.is_set())
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        capture.close()


if __name__ == "__main__":
    unittest.main()
