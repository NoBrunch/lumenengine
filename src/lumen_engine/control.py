"""Local operator application for Lumen Engine.

The terminal remains useful for diagnostics, but the supported operator surface
is the browser application served by this module.  It deliberately uses only
the Python standard library so the dedicated lighting computer does not need a
second service stack to start the UI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser

from lumen_engine.audio import AudioCaptureConfig, AlsaLineIn, RealtimeAudioAnalyzer
from lumen_engine.config import RigConfig, load_rig, rig_from_dict
from lumen_engine.dmx import DMXFrame, DMXOutput, VirtualDMXOutput
from lumen_engine.expression import ExpressionEngine, ExpressionPolicy
from lumen_engine.media import (
    SpotifyNowPlayingProvider,
    SpotifyOAuthPKCE,
    SpotifyTokenCache,
)
from lumen_engine.memory import SongMemoryStore
from lumen_engine.models import (
    Feedback,
    MediaIdentity,
    MusicalObservation,
    PerformanceDecision,
    Vec3,
    clamp,
)
from lumen_engine.profiles import PARTY_PARROT_PROFILES, profile_summary
from lumen_engine.runtime import PerformanceRuntime, RuntimeFrame
from lumen_engine.spatial import SpatialTargetingEngine, UnreachableTargetError
from lumen_engine.usb_dmx import OpenDmxUsbOutput, describe_open_dmx_environment


PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_DIR / "web"
DEFAULT_RIG_PATH = PROJECT_DIR / "config" / "party-parrot-active.json"
DEFAULT_MEMORY_PATH = PROJECT_DIR / "state" / "lumen.sqlite3"
DEFAULT_SETTINGS_PATH = PROJECT_DIR / "state" / "settings.json"
DEFAULT_SPOTIFY_TOKEN = (
    Path.home() / ".local" / "state" / "lumenengine" / "spotify-token.json"
)


@dataclass(slots=True)
class OperatorControls:
    master: float = 0.78
    intensity: float = 0.50
    motion: float = 0.46
    focus: float = 0.58
    warmth: float = 0.42
    influence: float = 0.62
    blackout: bool = False
    palette: str = "midnight_teal"

    def patch(self, values: dict[str, Any]) -> None:
        for name in (
            "master",
            "intensity",
            "motion",
            "focus",
            "warmth",
            "influence",
        ):
            if name in values:
                setattr(self, name, clamp(float(values[name]), 0.0, 1.0))
        if "blackout" in values:
            self.blackout = bool(values["blackout"])
        if "palette" in values:
            palette = str(values["palette"]).strip()
            if palette:
                self.palette = palette[:64]


class GatedOutput:
    """Apply operator blackout at the transport boundary."""

    def __init__(self, output: DMXOutput, controls: OperatorControls) -> None:
        self.output = output
        self.controls = controls
        self.last_frame = DMXFrame()
        self.frame_count = 0

    def send(self, frame: DMXFrame) -> None:
        self.last_frame = frame.copy()
        self.frame_count += 1
        if self.controls.blackout:
            self.output.send(DMXFrame())
        else:
            self.output.send(frame)

    def refresh_gate(self) -> None:
        if self.controls.blackout:
            if isinstance(self.output, OpenDmxUsbOutput):
                self.output.blackout()
            else:
                self.output.send(DMXFrame())
        else:
            self.output.send(self.last_frame)

    def close(self) -> None:
        self.output.close()


class OperatorExpressionEngine(ExpressionEngine):
    """Blend the owner's live influence into the authored expression baseline."""

    def __init__(
        self,
        controls: OperatorControls,
        policy: ExpressionPolicy,
    ) -> None:
        super().__init__(policy)
        self.controls = controls

    def decide(self, observation: MusicalObservation) -> PerformanceDecision:
        influence = self.controls.influence
        intensity_scale = 1.0 + (self.controls.intensity - 0.5) * 1.35 * influence
        motion_scale = 1.0 + (self.controls.motion - 0.5) * 1.60 * influence
        influenced = replace(
            observation,
            loudness=clamp(observation.loudness * intensity_scale, 0.0, 1.0),
            onset_strength=clamp(
                observation.onset_strength * motion_scale, 0.0, 1.0
            ),
            novelty=clamp(observation.novelty * motion_scale, 0.0, 1.0),
        )
        decision = super().decide(influenced)
        state = decision.expression
        warm_bias = (self.controls.warmth - 0.5) * 0.42 * influence
        motion_bias = (self.controls.motion - 0.5) * 0.50 * influence
        state = replace(
            state,
            energy=clamp(state.energy * intensity_scale, 0.0, 1.0),
            tension=clamp(state.tension + warm_bias, 0.0, 1.0),
            motion=clamp(state.motion + motion_bias, 0.0, 1.0),
            intimacy=clamp(state.intimacy - warm_bias * 0.35, 0.0, 1.0),
        )
        focus = self.controls.focus
        target = replace(
            decision.target,
            x=decision.target.x * (1.35 - focus),
            z=clamp(
                decision.target.z + (0.5 - focus) * 0.35,
                0.15,
                self.policy.room_high.z,
            ),
        )
        brightness = clamp(
            decision.brightness
            * self.controls.master
            * (0.74 + self.controls.intensity * 0.52),
            0.0,
            1.0,
        )
        reason = decision.reason
        if influence >= 0.12:
            reason += (
                f" Operator influence is {round(influence * 100)}%, biasing "
                f"intensity {round(self.controls.intensity * 100)} and "
                f"motion {round(self.controls.motion * 100)}."
            )
        return replace(
            decision,
            expression=state,
            target=target,
            brightness=brightness,
            reason=reason,
        )


class LumenApplication:
    """Own the runtime, persistent rig, song memory, and operator state."""

    def __init__(
        self,
        rig_path: str | Path = DEFAULT_RIG_PATH,
        memory_path: str | Path = DEFAULT_MEMORY_PATH,
        *,
        audio_device: str = "default",
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    ) -> None:
        self.rig_path = Path(rig_path)
        self.memory_path = Path(memory_path)
        self.settings_path = Path(settings_path)
        self._settings = self._read_settings()
        self.audio_device = (
            audio_device
            if audio_device != "default"
            else str(self._settings.get("audio_device", "default"))
        )
        self.spotify_client_id = str(
            os.environ.get("LUMEN_SPOTIFY_CLIENT_ID")
            or self._settings.get("spotify_client_id", "")
        ).strip()
        self._spotify_login_phase = "connected" if DEFAULT_SPOTIFY_TOKEN.exists() else "disconnected"
        self._spotify_login_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._output: GatedOutput | None = None
        self._runtime: PerformanceRuntime | None = None
        self.controls = OperatorControls()
        self.rig = load_rig(self.rig_path)
        self._rig_payload = self._read_rig_payload()
        self.memory = SongMemoryStore(self.memory_path)
        self.engine_mode = "standby"
        self.engine_phase = "ready"
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.observation = _silence_observation()
        self.frame: RuntimeFrame | None = None
        self.media: MediaIdentity | None = None
        self.song_id: int | None = None
        self._last_media_key: str | None = None
        self._last_gesture: str | None = None
        self.selected_target = Vec3(0.0, 0.0, 1.2)
        self.target_solutions: list[dict[str, Any]] = []
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self._status_sequence = 0
        self._last_media_poll = 0.0
        self._spotify_error: str | None = None
        self._add_event("system", f"Loaded {self.rig.name}")
        self.solve_target(self.selected_target)

    def _read_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("operator settings must be a JSON object")
        return payload

    def _save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._settings, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.settings_path)

    def _read_rig_payload(self) -> dict[str, Any]:
        payload = json.loads(self.rig_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("rig configuration must be a JSON object")
        return payload

    def _add_event(self, kind: str, message: str) -> None:
        self.events.appendleft(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "kind": kind,
                "message": message,
            }
        )

    def start(self, mode: str) -> dict[str, Any]:
        normalized = mode.strip().lower()
        if normalized not in {"monitor", "live", "demo"}:
            raise ValueError("mode must be monitor, live, or demo")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self.engine_mode == normalized:
                    return self.snapshot()
                raise RuntimeError(
                    f"Lumen is already running in {self.engine_mode} mode; stop it first"
                )
            self._stop.clear()
            self.engine_mode = normalized
            self.engine_phase = "starting"
            self.last_error = None
            self.started_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._run,
                args=(normalized,),
                name=f"lumen-{normalized}",
                daemon=True,
            )
            self._thread.start()
            self._add_event("engine", f"Starting {normalized} mode")
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop.set()
            if self.engine_mode != "standby":
                self.engine_phase = "stopping"
                self._add_event("engine", f"Stopping {self.engine_mode} mode")
        return self.snapshot()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        output = self._output
        if output is not None:
            try:
                output.close()
            except Exception:
                pass

    def _run(self, mode: str) -> None:
        raw_output: VirtualDMXOutput | OpenDmxUsbOutput
        runtime: PerformanceRuntime | None = None
        gated: GatedOutput | None = None
        try:
            raw_output = (
                OpenDmxUsbOutput.open()
                if mode == "live"
                else VirtualDMXOutput()
            )
            gated = GatedOutput(raw_output, self.controls)
            runtime = self._runtime_for_rig(gated)
            with self._lock:
                self._output = gated
                self._runtime = runtime
                self.engine_phase = "performing" if mode != "monitor" else "listening"
                self._add_event(
                    "output",
                    (
                        f"DMX connected through {raw_output.status.backend}"
                        if isinstance(raw_output, OpenDmxUsbOutput)
                        else "Virtual output active"
                    ),
                )
            if mode == "demo":
                self._run_demo(runtime)
            else:
                self._run_audio(runtime)
        except Exception as error:
            with self._lock:
                self.last_error = str(error)
                self.engine_phase = "fault"
                self._add_event("fault", str(error))
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as error:
                    with self._lock:
                        self.last_error = self.last_error or str(error)
            elif gated is not None:
                try:
                    gated.close()
                except Exception:
                    pass
            with self._lock:
                self._output = None
                self._runtime = None
                if self.engine_phase != "fault":
                    self.engine_phase = "ready"
                    self.engine_mode = "standby"
                    self._add_event("engine", "Engine stopped")
                self._status_sequence += 1

    def _run_audio(self, runtime: PerformanceRuntime) -> None:
        capture_config = AudioCaptureConfig(device=self.audio_device)
        analyzer = RealtimeAudioAnalyzer(
            capture_config.sample_rate, capture_config.channels
        )
        with AlsaLineIn(capture_config) as capture:
            for pcm in capture.chunks():
                if self._stop.is_set():
                    break
                observation = analyzer.analyze_pcm16(pcm)
                self._accept_runtime_frame(observation, runtime.step(observation))

    def _run_demo(self, runtime: PerformanceRuntime) -> None:
        index = 0
        while not self._stop.wait(0.12):
            observation = _demo_observation(index)
            self._accept_runtime_frame(observation, runtime.step(observation))
            index += 1

    def _accept_runtime_frame(
        self,
        observation: MusicalObservation,
        frame: RuntimeFrame,
    ) -> None:
        with self._lock:
            self.observation = observation
            self.frame = frame
            gesture = frame.decision.gesture.value
            if gesture != self._last_gesture:
                self._last_gesture = gesture
                self._add_event(
                    "gesture",
                    f"{gesture.title()}: {frame.decision.reason.split('.')[0]}.",
                )
                self.memory.log_decision(
                    frame.decision,
                    song_id=self.song_id,
                    position_ms=self._media_position_ms(),
                )
            self._status_sequence += 1
        self._poll_spotify_if_due()

    def _runtime_for_rig(self, output: GatedOutput) -> PerformanceRuntime:
        width_target = max(0.25, self.rig.room.width_m * 0.35)
        center_height = min(1.2, self.rig.room.height_m * 0.45)
        policy = ExpressionPolicy(
            room_center=Vec3(0.0, 0.0, center_height),
            room_high=Vec3(
                0.0, 0.0, min(self.rig.room.height_m * 0.82, 2.4)
            ),
            room_wide=Vec3(
                width_target,
                0.0,
                min(self.rig.room.height_m * 0.52, 1.4),
            ),
        )
        return PerformanceRuntime(
            self.rig.fixtures,
            output,
            auxiliary_fixtures=self.rig.auxiliary_fixtures,
            expression=OperatorExpressionEngine(self.controls, policy),
        )

    def patch_controls(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            previous_blackout = self.controls.blackout
            self.controls.patch(values)
            if previous_blackout != self.controls.blackout:
                self._add_event(
                    "control",
                    "Blackout engaged"
                    if self.controls.blackout
                    else "Blackout released",
                )
            output = self._output
            if output is not None:
                output.refresh_gate()
            self._status_sequence += 1
        return self.snapshot()

    def apply_preset(self, preset: str) -> dict[str, Any]:
        presets = {
            "restrained": {
                "intensity": 0.26,
                "motion": 0.18,
                "focus": 0.75,
                "warmth": 0.32,
                "influence": 0.78,
            },
            "balanced": {
                "intensity": 0.50,
                "motion": 0.46,
                "focus": 0.58,
                "warmth": 0.42,
                "influence": 0.62,
            },
            "open": {
                "intensity": 0.62,
                "motion": 0.40,
                "focus": 0.20,
                "warmth": 0.55,
                "influence": 0.72,
            },
            "drive": {
                "intensity": 0.82,
                "motion": 0.84,
                "focus": 0.38,
                "warmth": 0.68,
                "influence": 0.86,
            },
        }
        if preset not in presets:
            raise ValueError(f"unknown influence preset {preset!r}")
        with self._lock:
            self.controls.patch(presets[preset])
            self._add_event("control", f"Influence preset: {preset.title()}")
            self._status_sequence += 1
        return self.snapshot()

    def request_fresh_gesture(self) -> dict[str, Any]:
        with self._lock:
            if self._runtime is None:
                raise RuntimeError("start the engine before requesting a fresh gesture")
            self._runtime.expression.request_fresh_gesture()
            self._add_event("control", "Operator requested a fresh gesture")
            self._status_sequence += 1
        return self.snapshot()

    def patch_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if "audio_device" in payload:
                if self._thread is not None and self._thread.is_alive():
                    raise RuntimeError(
                        "stop the engine before changing the audio input"
                    )
                audio_device = str(payload["audio_device"]).strip()
                if not audio_device:
                    raise ValueError("audio_device must not be empty")
                self.audio_device = audio_device[:160]
                self._settings["audio_device"] = self.audio_device
                self._add_event(
                    "system", f"Audio input changed to {self.audio_device}"
                )
            if "spotify_client_id" in payload:
                client_id = str(payload["spotify_client_id"]).strip()
                if not client_id:
                    raise ValueError("spotify_client_id must not be empty")
                self.spotify_client_id = client_id[:256]
                self._settings["spotify_client_id"] = self.spotify_client_id
            self._save_settings()
            self._status_sequence += 1
        return {
            "settings": self.operator_settings(),
            "status": self.snapshot(),
        }

    def operator_settings(self) -> dict[str, Any]:
        return {
            "audio_device": self.audio_device,
            "spotify_client_id_masked": _masked_identifier(
                self.spotify_client_id
            ),
        }

    def connect_spotify(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_id = str(
            payload.get("client_id") or self.spotify_client_id
        ).strip()
        if not client_id:
            raise ValueError("paste the Spotify developer client ID first")
        with self._lock:
            if (
                self._spotify_login_thread is not None
                and self._spotify_login_thread.is_alive()
            ):
                raise RuntimeError("Spotify connection is already in progress")
            self.spotify_client_id = client_id[:256]
            self._settings["spotify_client_id"] = self.spotify_client_id
            self._save_settings()
            self._spotify_login_phase = "connecting"
            self._spotify_error = None
            self._spotify_login_thread = threading.Thread(
                target=self._complete_spotify_login,
                name="lumen-spotify-login",
                daemon=True,
            )
            self._spotify_login_thread.start()
            self._add_event(
                "media", "Opening Spotify authorization in the desktop browser"
            )
            self._status_sequence += 1
        return {
            "phase": self._spotify_login_phase,
            "message": "Spotify authorization is opening in the desktop browser.",
        }

    def _complete_spotify_login(self) -> None:
        try:
            oauth = SpotifyOAuthPKCE(
                client_id=self.spotify_client_id,
                cache=SpotifyTokenCache(DEFAULT_SPOTIFY_TOKEN),
            )
            oauth.login(open_browser=True)
            with self._lock:
                self._spotify_login_phase = "connected"
                self._spotify_error = None
                self._last_media_poll = 0.0
                self._add_event("media", "Spotify playback identity connected")
                self._status_sequence += 1
        except Exception as error:
            with self._lock:
                self._spotify_login_phase = "fault"
                self._spotify_error = str(error)
                self._add_event("fault", f"Spotify connection: {error}")
                self._status_sequence += 1

    def solve_target(self, target: Vec3) -> list[dict[str, Any]]:
        solver = SpatialTargetingEngine()
        solutions: list[dict[str, Any]] = []
        for fixture in self.rig.fixtures:
            try:
                solution = solver.solve(fixture, target)
                solutions.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "fixture_name": fixture.name,
                        "reachable": True,
                        "pan_deg": solution.pan_deg,
                        "tilt_deg": solution.tilt_deg,
                        "distance_m": solution.distance_m,
                        "aim_error_deg": solution.aim_error_deg,
                        "branch": solution.branch,
                    }
                )
            except UnreachableTargetError as error:
                solutions.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "fixture_name": fixture.name,
                        "reachable": False,
                        "error": str(error),
                    }
                )
        with self._lock:
            self.selected_target = target
            self.target_solutions = solutions
            self._status_sequence += 1
        return solutions

    def patch_fixture(self, payload: dict[str, Any]) -> dict[str, Any]:
        fixture_id = str(payload.get("fixture_id", ""))
        if not fixture_id:
            raise ValueError("fixture_id is required")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("stop the engine before changing the rig")
            candidates = list(self._rig_payload.get("fixtures", [])) + list(
                self._rig_payload.get("auxiliary_fixtures", [])
            )
            fixture = next(
                (item for item in candidates if str(item.get("id")) == fixture_id),
                None,
            )
            if fixture is None:
                raise ValueError(f"fixture {fixture_id!r} is not in the active rig")
            if "name" in payload:
                fixture["name"] = str(payload["name"]).strip() or fixture["name"]
            if "address" in payload:
                fixture["address"] = int(payload["address"])
            if "universe" in payload:
                fixture["universe"] = int(payload["universe"])
            if "position_m" in payload:
                fixture["position_m"] = _number_triplet(
                    payload["position_m"], "position_m"
                )
            if "housing_rotation_deg" in payload:
                fixture["housing_rotation_deg"] = _number_triplet(
                    payload["housing_rotation_deg"], "housing_rotation_deg"
                )
            calibration = payload.get("calibration")
            if calibration is not None:
                if "calibration" not in fixture:
                    raise ValueError("this fixture does not use spatial calibration")
                for key, value in dict(calibration).items():
                    if key in {
                        "pan_direction",
                        "tilt_direction",
                        "pan_dmx_min_u16",
                        "pan_dmx_max_u16",
                        "tilt_dmx_min_u16",
                        "tilt_dmx_max_u16",
                    }:
                        fixture["calibration"][key] = int(value)
                    elif key in {"pan_invert_dmx", "tilt_invert_dmx"}:
                        fixture["calibration"][key] = bool(value)
                    else:
                        fixture["calibration"][key] = float(value)
            validated = rig_from_dict(self._rig_payload)
            temporary = self.rig_path.with_suffix(self.rig_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self._rig_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.rig_path)
            self.rig = validated
            self.solve_target(self.selected_target)
            self._add_event("rig", f"Saved fixture {fixture['name']}")
            self._status_sequence += 1
        return self.bootstrap()

    def add_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        label = str(payload.get("label", "")).strip()
        if not label:
            raise ValueError("feedback label is required")
        value = float(payload.get("value", 1.0 if label == "liked_this" else -1.0))
        note = str(payload.get("note", "")).strip() or None
        with self._lock:
            if self.song_id is None:
                media = self.media or MediaIdentity(
                    provider="line-in",
                    provider_item_id=f"unidentified:{datetime.now():%Y-%m-%d}",
                    title="Unidentified line-in session",
                    artists=(),
                    is_playing=self.engine_mode != "standby",
                )
                self.song_id = self.memory.remember_media(media)
            feedback_id = self.memory.add_feedback(
                Feedback(
                    song_id=self.song_id,
                    position_ms=self._media_position_ms(),
                    label=label[:64],
                    value=clamp(value, -1.0, 1.0),
                    note=note,
                )
            )
            self._add_event("feedback", f"Recorded feedback: {label.replace('_', ' ')}")
            self._status_sequence += 1
        return {"feedback_id": feedback_id, "song_id": self.song_id}

    def _media_position_ms(self) -> int | None:
        media = self.media
        if media is None:
            if self.started_at is None:
                return None
            return round((time.monotonic() - self.started_at) * 1000)
        if media.observed_position_ms is None:
            return None
        if not media.is_playing or media.observed_at_unix_ms is None:
            return media.observed_position_ms
        elapsed = max(0, int(time.time() * 1000) - media.observed_at_unix_ms)
        position = media.observed_position_ms + elapsed
        if media.duration_ms is not None:
            position = min(position, media.duration_ms)
        return position

    def _poll_spotify_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_media_poll < 5.0:
            return
        self._last_media_poll = now
        client_id = self.spotify_client_id
        if not client_id or not DEFAULT_SPOTIFY_TOKEN.exists():
            return
        try:
            oauth = SpotifyOAuthPKCE(
                client_id=client_id,
                cache=SpotifyTokenCache(DEFAULT_SPOTIFY_TOKEN),
            )
            media = SpotifyNowPlayingProvider(oauth.valid_token).now_playing()
            with self._lock:
                self._spotify_error = None
                self.media = media
                if media is not None:
                    key = f"{media.provider}:{media.provider_item_id}"
                    count_play = key != self._last_media_key
                    self.song_id = self.memory.remember_media(
                        media, count_play=count_play
                    )
                    if count_play:
                        self._last_media_key = key
                        self._add_event("media", f"Now playing {media.display_name}")
        except Exception as error:
            with self._lock:
                self._spotify_error = str(error)

    def scan_system(self) -> dict[str, Any]:
        dmx = describe_open_dmx_environment()
        audio: dict[str, Any] = {
            "arecord": shutil.which("arecord"),
            "cards": [],
            "report": "",
        }
        if audio["arecord"]:
            result = subprocess.run(
                ["arecord", "-l"],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            report = (result.stdout + result.stderr).strip()
            audio["report"] = report
            audio["cards"] = [
                line.strip()
                for line in report.splitlines()
                if line.strip().lower().startswith("card ")
            ]
        return {
            "dmx": dmx,
            "audio": audio,
            "network": {
                "host_name": socket.gethostname(),
                "addresses": _lan_addresses(),
            },
            "spotify": {
                "client_id_configured": bool(self.spotify_client_id),
                "client_id_masked": _masked_identifier(
                    self.spotify_client_id
                ),
                "token_present": DEFAULT_SPOTIFY_TOKEN.exists(),
                "error": self._spotify_error,
                "phase": self._spotify_login_phase,
            },
        }

    def bootstrap(self) -> dict[str, Any]:
        with self._lock:
            return {
                "project": {
                    "name": "Lumen Engine",
                    "version": "0.2.0",
                    "role": "Spatial music-lighting control",
                },
                "rig": self._rig_payload,
                "profiles": [
                    profile_summary(profile)
                    for profile in PARTY_PARROT_PROFILES.values()
                ],
                "status": self._snapshot_unlocked(),
                "memory": self.memory.summary(limit=30),
                "system": self.scan_system(),
                "settings": self.operator_settings(),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        observation = asdict(self.observation)
        decision: dict[str, Any] | None = None
        solutions: list[dict[str, Any]] = []
        dmx: dict[str, Any] = {"universes": [], "active_channels": []}
        if self.frame is not None:
            decision = asdict(self.frame.decision)
            decision["gesture"] = self.frame.decision.gesture.value
            solutions = [
                {
                    "fixture_id": solution.fixture_id,
                    "target": asdict(solution.target),
                    "pan_deg": solution.pan_deg,
                    "tilt_deg": solution.tilt_deg,
                    "distance_m": solution.distance_m,
                    "movement_cost_deg": solution.movement_cost_deg,
                    "aim_error_deg": solution.aim_error_deg,
                    "branch": solution.branch,
                }
                for solution in self.frame.solutions
            ]
            active_channels: list[dict[str, int]] = []
            for universe in self.frame.dmx.universes:
                data = self.frame.dmx.universe_data(universe)
                active_channels.extend(
                    {
                        "universe": universe,
                        "channel": index + 1,
                        "value": value,
                    }
                    for index, value in enumerate(data)
                    if value
                )
            dmx = {
                "universes": list(self.frame.dmx.universes),
                "active_channels": active_channels,
            }
        output_status: dict[str, Any] | None = None
        if self._output is not None:
            raw = self._output.output
            if isinstance(raw, OpenDmxUsbOutput):
                output_status = asdict(raw.status)
            elif isinstance(raw, VirtualDMXOutput):
                output_status = {
                    "backend": "Virtual DMX",
                    "universe": 0,
                    "frame_rate_hz": None,
                    "frames_sent": raw.frame_count,
                    "last_error": None,
                }
        media: dict[str, Any] | None = None
        if self.media is not None:
            media = asdict(self.media)
            media.pop("raw", None)
            media["display_name"] = self.media.display_name
            media["live_position_ms"] = self._media_position_ms()
        return {
            "sequence": self._status_sequence,
            "engine": {
                "mode": self.engine_mode,
                "phase": self.engine_phase,
                "running": self._thread is not None and self._thread.is_alive(),
                "uptime_s": (
                    None
                    if self.started_at is None
                    else max(0.0, time.monotonic() - self.started_at)
                ),
                "audio_device": self.audio_device,
                "error": self.last_error,
            },
            "controls": asdict(self.controls),
            "observation": observation,
            "decision": decision,
            "solutions": solutions,
            "dmx": dmx,
            "output": output_status,
            "media": media,
            "song_id": self.song_id,
            "selected_target": asdict(self.selected_target),
            "target_solutions": list(self.target_solutions),
            "events": list(self.events),
        }


class LumenRequestHandler(BaseHTTPRequestHandler):
    server: "LumenHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/bootstrap":
            self._json(HTTPStatus.OK, self.server.application.bootstrap())
            return
        if path == "/api/status":
            self._json(HTTPStatus.OK, self.server.application.snapshot())
            return
        if path == "/api/system":
            self._json(HTTPStatus.OK, self.server.application.scan_system())
            return
        if path == "/api/memory":
            self._json(
                HTTPStatus.OK,
                self.server.application.memory.summary(limit=100),
            )
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            app = self.server.application
            if path == "/api/engine/start":
                result = app.start(str(payload.get("mode", "monitor")))
            elif path == "/api/engine/stop":
                result = app.stop()
            elif path == "/api/service/shutdown":
                app.stop()
                self._json(
                    HTTPStatus.OK,
                    {
                        "shutting_down": True,
                        "message": "Lumen is shutting down.",
                    },
                )
                threading.Thread(
                    target=self._shutdown_server,
                    name="lumen-service-shutdown",
                    daemon=True,
                ).start()
                return
            elif path == "/api/control":
                result = app.patch_controls(payload)
            elif path == "/api/preset":
                result = app.apply_preset(str(payload.get("preset", "")))
            elif path == "/api/feedback":
                result = app.add_feedback(payload)
            elif path == "/api/gesture/fresh":
                result = app.request_fresh_gesture()
            elif path == "/api/settings":
                result = app.patch_settings(payload)
            elif path == "/api/spotify/connect":
                result = app.connect_spotify(payload)
            elif path == "/api/target":
                target = Vec3(
                    float(payload["x"]),
                    float(payload["y"]),
                    float(payload["z"]),
                )
                result = {
                    "target": asdict(target),
                    "solutions": app.solve_target(target),
                }
            elif path == "/api/rig/fixture":
                result = app.patch_fixture(payload)
            else:
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"unknown endpoint {path}"},
                )
                return
            self._json(HTTPStatus.OK, result)
        except (KeyError, TypeError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except RuntimeError as error:
            self._json(HTTPStatus.CONFLICT, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request is too large")
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _serve_static(self, path: str) -> None:
        route = "/index.html" if path in {"/", "/remote"} else path
        files = {
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        asset = files.get(route)
        if asset is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        source = WEB_DIR / asset[0]
        try:
            body = source.read_bytes()
        except OSError as error:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"interface asset unavailable: {error}"},
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", asset[1])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _shutdown_server(self) -> None:
        time.sleep(0.15)
        self.server.shutdown()


class LumenHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        application: LumenApplication,
    ) -> None:
        self.application = application
        super().__init__(address, LumenRequestHandler)

    def server_close(self) -> None:
        self.application.close()
        super().server_close()


def serve(
    *,
    host: str = "0.0.0.0",
    port: int = 4042,
    rig_path: str | Path = DEFAULT_RIG_PATH,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    audio_device: str = "default",
    open_browser: bool = False,
) -> None:
    application = LumenApplication(
        rig_path=rig_path,
        memory_path=memory_path,
        audio_device=audio_device,
    )
    server = LumenHTTPServer((host, port), application)
    desktop_url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(desktop_url,)).start()
    print(f"Lumen operator console: {desktop_url}")
    addresses = _lan_addresses()
    if addresses:
        print(f"Phone remote: http://{addresses[0]}:{port}/remote")
    else:
        print(f"Phone remote: http://<this-computer-ip>:{port}/remote")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _number_triplet(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return [float(component) for component in value]


def _masked_identifier(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "•" * len(value)
    return f"{value[:6]}…{value[-4:]}"


def _silence_observation() -> MusicalObservation:
    return MusicalObservation(
        timestamp_s=0.0,
        loudness=0.0,
        onset_strength=0.0,
        low_energy=0.0,
        mid_energy=0.0,
        high_energy=0.0,
    )


def _demo_observation(index: int) -> MusicalObservation:
    timestamp = index * 0.12
    cycle = (index % 320) / 319.0
    beat = index % 4 == 0
    if cycle < 0.22:
        section = "intro"
        energy = 0.18 + cycle
    elif cycle < 0.58:
        section = "build"
        energy = 0.36 + (cycle - 0.22) * 1.28
    elif cycle < 0.64:
        section = "drop"
        energy = 0.94
    else:
        section = "chorus"
        energy = 0.76 + 0.12 * math.sin(index * 0.24)
    return MusicalObservation(
        timestamp_s=timestamp,
        loudness=clamp(energy, 0.0, 1.0),
        onset_strength=0.88 if beat else 0.22 + energy * 0.24,
        low_energy=clamp(0.35 + energy * 0.60, 0.0, 1.0),
        mid_energy=clamp(0.40 + energy * 0.42, 0.0, 1.0),
        high_energy=clamp(0.18 + cycle * 0.72, 0.0, 1.0),
        beat_phase=(index % 4) / 4.0,
        beat_confidence=0.88,
        bpm=125.0,
        section=section,
        section_confidence=0.86,
        novelty=0.92 if section == "drop" and beat else 0.24 + energy * 0.34,
    )


def _lan_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            parsed = ipaddress.ip_address(address)
            if not parsed.is_loopback and parsed.is_private:
                addresses.add(address)
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            parsed = ipaddress.ip_address(address)
            if not parsed.is_loopback:
                addresses.add(address)
        finally:
            probe.close()
    except OSError:
        pass
    return sorted(addresses)
