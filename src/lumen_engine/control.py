"""Local operator application for Lumen Engine.

The terminal remains useful for diagnostics, but the supported operator surface
is the browser application served by this module.  It deliberately uses only
the Python standard library so the dedicated lighting computer does not need a
second service stack to start the UI.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

from lumen_engine.audio import (
    AudioCaptureConfig,
    AudioInputMetrics,
    AlsaLineIn,
    RealtimeAudioAnalyzer,
)
from lumen_engine.config import RigConfig, load_rig, rig_from_dict
from lumen_engine.choreography import SequencePreferenceModel
from lumen_engine.dmx import DMXFrame, DMXOutput, VirtualDMXOutput
from lumen_engine.expression import ExpressionEngine, ExpressionPolicy
from lumen_engine.media import (
    SpotifyNowPlayingProvider,
    SpotifyOAuthPKCE,
    SpotifyTokenCache,
    SpotifyWebAPI,
    media_identity_from_spotify,
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
from lumen_engine.motion import (
    DEFAULT_MOTION_TUNINGS,
    MotionTuning,
    merged_motion_tunings,
    preview_paths,
    required_axis_speeds,
)
from lumen_engine.offline import (
    EDMFORMER_JOB,
    OfflineResearchWorker,
    ResearchJobCoordinator,
    SONGFORMER_JOB,
    STUDENT_TRAIN_JOB,
    enqueue_student_training,
    training_readiness,
)
from lumen_engine.profiles import (
    PARTY_PARROT_PROFILES,
    party_parrot_profile,
    profile_summary,
)
from lumen_engine.research import ResearchManager
from lumen_engine.fixture_output import PALETTE_FAMILIES
from lumen_engine.runtime import PerformanceRuntime, RuntimeFrame
from lumen_engine.spatial import SpatialTargetingEngine, UnreachableTargetError
from lumen_engine.student import (
    StableStructureDecoder,
    StreamingStructureStudent,
    semantic_frame_features,
)
from lumen_engine.training import (
    TrainingCaptureConfig,
    TrainingDataRecorder,
    export_training_dataset,
)
from lumen_engine.usb_dmx import OpenDmxUsbOutput, describe_open_dmx_environment


PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_DIR / "web"
DEFAULT_RIG_PATH = PROJECT_DIR / "config" / "party-parrot-active.json"
DEFAULT_MEMORY_PATH = PROJECT_DIR / "state" / "lumen.sqlite3"
DEFAULT_SETTINGS_PATH = PROJECT_DIR / "state" / "settings.json"
DEFAULT_SPOTIFY_TOKEN = (
    Path.home() / ".local" / "state" / "lumenengine" / "spotify-token.json"
)

REHEARSAL_ROUTINES: tuple[dict[str, str], ...] = (
    {
        "id": "breathe",
        "name": "Breathe",
        "description": "Slow spacious arcs for quiet or intimate passages.",
    },
    {
        "id": "fan_sweep",
        "name": "Fan sweep",
        "description": "Broad coordinated pan with a smaller tilt breathe.",
    },
    {
        "id": "figure_eight",
        "name": "Figure eight",
        "description": "Looping pan with a faster vertical harmonic.",
    },
    {
        "id": "opposing_chase",
        "name": "Opposing chase",
        "description": "The two sides trade position across the beat grid.",
    },
    {
        "id": "beat_nod",
        "name": "Beat nod",
        "description": "Alternating tilt accents anchored to the beat.",
    },
    {
        "id": "counter_rotate",
        "name": "Counter-rotate",
        "description": "Opposed circular motion with independent side arms.",
    },
)
REHEARSAL_ROUTINE_IDS = frozenset(item["id"] for item in REHEARSAL_ROUTINES)


@dataclass(slots=True)
class OperatorControls:
    master: float = 0.86
    intensity: float = 0.62
    motion: float = 0.68
    focus: float = 0.50
    warmth: float = 0.44
    influence: float = 0.74
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
            if palette in PALETTE_FAMILIES:
                self.palette = palette[:64]


@dataclass(slots=True)
class RehearsalControls:
    routine: str = "figure_eight"
    scope: str = "movers"
    output: str = "virtual"
    bpm: float = 120.0
    intensity: float = 0.68
    size: float = 1.0
    palette: str = "party_vivid"
    strobe: float = 0.0
    isolate: bool = True
    tour: bool = False

    def patch(self, values: dict[str, Any]) -> None:
        if "routine" in values:
            routine = str(values["routine"]).strip().lower()
            if routine not in REHEARSAL_ROUTINE_IDS:
                raise ValueError("unknown rehearsal routine")
            self.routine = routine
        if "scope" in values:
            scope = str(values["scope"]).strip().lower()
            if scope not in {"overall", "movers", "center"} and not scope.startswith("fixture:"):
                raise ValueError("unknown rehearsal fixture scope")
            self.scope = scope
        if "output" in values:
            output = str(values["output"]).strip().lower()
            if output not in {"virtual", "live"}:
                raise ValueError("rehearsal output must be virtual or live")
            self.output = output
        if "bpm" in values:
            self.bpm = clamp(float(values["bpm"]), 40.0, 240.0)
        if "intensity" in values:
            self.intensity = clamp(float(values["intensity"]), 0.0, 1.0)
        if "size" in values:
            self.size = clamp(float(values["size"]), 0.0, 1.0)
        if "strobe" in values:
            self.strobe = clamp(float(values["strobe"]), 0.0, 1.0)
        if "palette" in values:
            palette = str(values["palette"]).strip()
            if palette not in PALETTE_FAMILIES:
                raise ValueError("unknown rehearsal palette")
            self.palette = palette
        if "isolate" in values:
            self.isolate = bool(values["isolate"])
        if "tour" in values:
            self.tour = bool(values["tour"])


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
            # Influence controls are centered biases, not replacement state.
            # The previous blend forced energy and motion toward the slider
            # positions on every frame, flattening quiet/loud contrast.
            energy=clamp(
                state.energy
                + (self.controls.intensity - 0.5) * 0.40 * influence,
                0.0,
                1.0,
            ),
            tension=clamp(state.tension + warm_bias, 0.0, 1.0),
            motion=clamp(
                state.motion
                + (self.controls.motion - 0.5) * 0.50 * influence
                + motion_bias * 0.15,
                0.0,
                1.0,
            ),
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
            palette_hint=self.controls.palette,
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
        self.motion_path = self.settings_path.parent / "motion-routines.json"
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
        self._spotify_poll_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        # Persistence/model updates may be comparatively slow. Serialize them
        # independently so feedback from several clients cannot block the
        # audio/DMX publication lock or corrupt reversible model events.
        self._feedback_lock = threading.Lock()
        self._media_lock = threading.Lock()
        # Export preparation reconstructs and validates captured audio. Keep it
        # single-flight even though the HTTP server handles requests in
        # parallel, so a double-click cannot build two manifests at once.
        self._training_export_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._output: GatedOutput | None = None
        self._runtime: PerformanceRuntime | None = None
        self.controls = OperatorControls()
        self.rehearsal = RehearsalControls()
        self.motion_tunings = self._load_motion_tunings()
        self.rig = load_rig(self.rig_path)
        self._rig_payload = self._read_rig_payload()
        self.memory = SongMemoryStore(self.memory_path)
        default_memory = DEFAULT_MEMORY_PATH.resolve(strict=False)
        requested_memory = self.memory_path.resolve(strict=False)
        state_root = (
            self.memory_path.parent
            if requested_memory == default_memory
            else self.memory_path.parent
            / f".{self.memory_path.stem}-lumen-state"
        )
        self.training_root = state_root / "training"
        self._choreography_model_path = (
            state_root / "models" / "choreography-preferences.json"
        )
        self._choreography_model = self._load_choreography_model()
        self.research = ResearchManager(
            self.training_root / "research",
            store=self.memory,
        )
        self._student_model_path = (
            self.training_root
            / "research"
            / "models"
            / "lumen-structure-student.npz"
        )
        self._student_model: StreamingStructureStudent | None = None
        self._student_decoder = StableStructureDecoder()
        self._student_model_error: str | None = None
        self._student_prediction: dict[str, Any] | None = None
        self._load_student_model()
        self.training_capture_enabled = bool(
            self._settings.get("training_capture_enabled", True)
        )
        self.training_max_gb = clamp(
            float(self._settings.get("training_max_gb", 100.0)),
            1.0,
            800.0,
        )
        self._training_recorder: TrainingDataRecorder | None = None
        self._training_audio_frame: int | None = None
        self._training_linked_feedback = 0
        self._training_annotations = 0
        self._training_history = self.memory.training_summary()
        self._last_training_export: str | None = None
        self._training_prepare_thread: threading.Thread | None = None
        self._training_prepare_pending = False
        self._research_worker_thread: threading.Thread | None = None
        self._research_worker_last: dict[str, Any] | None = None
        self._research_worker_progress: dict[str, Any] = {
            "processed": 0,
            "failed": 0,
            "current_job_type": None,
        }
        self._research_recovered: list[dict[str, Any]] = []
        self._research_cancel = threading.Event()
        self._trace_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)
        self._trace_stop = threading.Event()
        self._trace_thread = threading.Thread(
            target=self._trace_worker,
            name="lumen-performance-trace",
            daemon=True,
        )
        self._trace_thread.start()
        self._session_id = "standby"
        self._last_trace_timestamp: float | None = None
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
        self._research_recovered = self.memory.recover_abandoned_analysis_jobs()
        if self._research_recovered:
            self._add_event(
                "memory",
                (
                    "Recovered "
                    f"{len(self._research_recovered)} interrupted offline "
                    "analysis job(s); they are ready to resume"
                ),
            )
        self._status_sequence = 0
        self._last_media_poll = 0.0
        self._spotify_error: str | None = None
        self._spotify_last_command: dict[str, Any] | None = None
        self._spotify_console_cache: dict[str, Any] = {}
        self._spotify_rate_limited_until = 0.0
        self._feedback_biases: dict[str, dict[str, float]] = {}
        self._calibration_overrides: dict[str, dict[str, float]] = {}
        self._audio_metrics = AudioInputMetrics.silence()
        self._audio_packets = 0
        self._audio_frames = 0
        self._audio_bytes = 0
        self._audio_capture_started_at: float | None = None
        self._audio_last_packet_at: float | None = None
        self._audio_packet_times: deque[float] = deque(maxlen=96)
        self._analysis_history: deque[dict[str, Any]] = deque(maxlen=240)
        self._last_analysis_history_at: float | None = None
        self._analysis_generation = 0
        self._add_event("system", f"Loaded {self.rig.name}")
        self._rebuild_feedback_biases()
        self.solve_target(self.selected_target)

    def _load_student_model(self) -> None:
        self._student_model = None
        self._student_model_error = None
        self._student_decoder.reset()
        if not self._student_model_path.is_file():
            return
        try:
            self._student_model = StreamingStructureStudent.load(
                self._student_model_path
            )
        except (OSError, ValueError) as error:
            self._student_model_error = str(error)

    def _apply_student_structure(
        self,
        observation: MusicalObservation,
        metrics: AudioInputMetrics,
    ) -> MusicalObservation:
        model = self._student_model
        if model is None:
            self._student_prediction = None
            return observation
        prediction = model.predict(
            semantic_frame_features(
                self._semantic_audio_payload(observation, metrics)
            ),
            timestamp_s=observation.timestamp_s,
        )
        stable = self._student_decoder.update(
            prediction, observation.timestamp_s
        )
        selected_section = observation.section
        selected_axis = "live_analyzer"
        selected_confidence = observation.section_confidence
        energy_confidence = float(stable["confidence"]["energy"])
        functional_confidence = float(stable["confidence"]["functional"])
        content_confidence = float(stable["confidence"]["content"])
        accepted_axes = {
            "energy": bool(
                stable["energy"] not in {None, "unknown"}
                and energy_confidence >= 0.52
            ),
            "functional": bool(
                stable["functional"] not in {None, "unknown"}
                and functional_confidence >= 0.60
            ),
            "content": bool(
                stable["content"] not in {None, "unknown"}
                and content_confidence >= 0.55
            ),
        }
        if (
            stable["energy"] not in {None, "unknown", "sustained"}
            and energy_confidence >= 0.52
        ):
            selected_section = str(stable["energy"])
            selected_axis = "student_energy"
            selected_confidence = energy_confidence
        self._student_prediction = {
            "functional": stable["functional"] or prediction.functional,
            "energy": stable["energy"] or prediction.energy,
            "content": stable["content"] or prediction.content,
            "confidence": stable["confidence"],
            "raw": {
                "functional": prediction.functional,
                "energy": prediction.energy,
                "content": prediction.content,
                "confidence": prediction.confidence,
            },
            "stable": stable,
            "boundary_probability": prediction.boundary_probability,
            "accepted_axes": accepted_axes,
            "selected_axis": selected_axis,
            "selected_section": selected_section,
            "model_path": str(self._student_model_path),
            "training_examples": model.training_examples,
            "target_provenance": "lumen_streaming_structure_student",
        }
        if selected_axis == "live_analyzer":
            return observation
        return replace(
            observation,
            section=selected_section,
            section_confidence=selected_confidence,
        )

    @staticmethod
    def _semantic_audio_payload(
        observation: MusicalObservation,
        metrics: AudioInputMetrics,
    ) -> dict[str, Any]:
        """Fill the causal student contract from measurements we record."""
        observation_payload = asdict(observation)
        spectral_total = max(
            1e-9,
            observation.low_energy
            + observation.mid_energy
            + observation.high_energy,
        )
        observation_payload.update(
            {
                "spectral_flux": (
                    observation.spectral_flux or observation.novelty
                ),
                "spectral_brightness": (
                    observation.spectral_brightness
                    or clamp(
                        (
                            observation.high_energy
                            + 0.5 * observation.mid_energy
                        )
                        / spectral_total,
                        0.0,
                        1.0,
                    )
                ),
                "tempo_confidence": observation.beat_confidence,
                "silence_confidence": (
                    1.0
                    if observation.section == "silence"
                    else clamp(
                        1.0 - observation.loudness / 0.04,
                        0.0,
                        1.0,
                    )
                ),
            }
        )
        audio_payload = asdict(metrics)
        audio_payload["clipping"] = clamp(
            metrics.clipped_samples
            / max(1, metrics.frame_count * len(metrics.channel_peak)),
            0.0,
            1.0,
        )
        return {
            "observation": observation_payload,
            "audio_metrics": audio_payload,
        }

    def _student_runtime_context(self) -> dict[str, Any]:
        """Expose only confidence-gated structure labels to choreography.

        The complete prediction remains visible for diagnosis, but a weak
        categorical guess must not become a routine-selection token or motion
        expansion merely because it happened to win a softmax frame.
        """

        prediction = self._student_prediction or {}
        accepted = prediction.get("accepted_axes") or {}
        confidence = prediction.get("confidence") or {}
        energy_accepted = bool(accepted.get("energy"))
        return {
            "functional": (
                str(prediction.get("functional") or "unknown")
                if accepted.get("functional")
                else "unknown"
            ),
            "energy": (
                str(prediction.get("energy") or "unknown")
                if energy_accepted
                else "unknown"
            ),
            "content": (
                str(prediction.get("content") or "unknown")
                if accepted.get("content")
                else "unknown"
            ),
            "confidence": (
                float(confidence.get("energy", 0.0))
                if energy_accepted
                else 0.0
            ),
            "boundary_probability": (
                float(prediction.get("boundary_probability", 0.0))
                if energy_accepted
                else 0.0
            ),
        }

    def _load_choreography_model(self) -> SequencePreferenceModel:
        path = self._choreography_model_path
        if not path.is_file():
            return SequencePreferenceModel()
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            return SequencePreferenceModel.from_state_dict(state)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Keep the invalid file for diagnosis; never replace it merely
            # because application startup could not parse it.
            return SequencePreferenceModel()

    def _save_choreography_model(self) -> None:
        path = self._choreography_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(
            json.dumps(
                self._choreography_model.state_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _remember_sequence_learning(
        self,
        learning: dict[str, Any],
        *,
        label: str,
        scope: str,
        song_id: int | None = None,
    ) -> None:
        target_song_id = self.song_id if song_id is None else song_id
        if target_song_id is None:
            return
        for role in ("performed_sequence", "preferred_sequence"):
            sequence = learning.get(role)
            if not isinstance(sequence, dict):
                continue
            steps = sequence.get("steps")
            if not isinstance(steps, list) or not steps:
                continue
            sequence_name = str(sequence.get("sequence_id") or role)
            self.memory.save_choreography_sequence(
                sequence_id=(
                    f"online:{target_song_id}:{role}:{sequence_name}"
                ),
                song_id=target_song_id,
                source=(
                    "operator_preferred_sequence"
                    if role == "preferred_sequence"
                    else "performed_sequence_feedback"
                ),
                confidence=clamp(
                    float(learning.get("urgency", 0.0)), 0.0, 1.0
                ),
                context={
                    "feedback_label": label,
                    "scope": scope,
                    "model_revision": learning.get("model_revision"),
                    "effective_strength": learning.get(
                        "effective_strength"
                    ),
                },
                steps=[
                    {
                        "start_beat": step["start_beat"],
                        "duration_beats": step["duration_beats"],
                        "fixture_scope": step["fixture_scope"],
                        "routine": step["routine"],
                        "intensity": step.get("intensity", 1.0),
                        "palette": step.get("palette"),
                        "parameters": {
                            "strobe": step.get("strobe", 0.0),
                            "beat_sync": step.get("beat_sync", 1.0),
                        },
                    }
                    for step in steps
                ],
            )

    def _save_feedback_routine(
        self, song_id: int, song_feedback: list[dict[str, Any]]
    ) -> None:
        """Persist the currently reversible feedback view for one song."""
        self.memory.save_routine(
            song_id,
            routine_version=len(song_feedback),
            payload={
                "kind": "semantic_feedback_routine",
                "moments": [
                    {
                        "position_ms": row.get("position_ms"),
                        "label": row.get("label"),
                        "gesture": row.get("gesture"),
                        "routine": row.get("routine"),
                        "section": row.get("section"),
                        "scope": row.get("scope"),
                        "fixture_id": row.get("fixture_id"),
                    }
                    for row in song_feedback[-128:]
                ],
                "palette_family": self.controls.palette,
            },
        )

    @staticmethod
    def _feedback_effect(label: str) -> tuple[float, float, float, float]:
        return {
            "increase_movement": (0.28, 0.0, 0.0, 0.0), "decrease_movement": (-0.28, 0.0, 0.0, 0.0),
            "too_busy": (-0.28, 0.0, 0.0, 0.0), "not_busy_enough": (0.28, 0.0, 0.0, 0.0),
            "calm_down": (-0.35, -0.08, -0.6, 0.0), "pick_it_up": (0.35, 0.10, 0.0, 0.0),
            "too_bright": (0.0, -0.25, 0.0, 0.0), "too_dim": (0.0, 0.25, 0.0, 0.0),
            "great_timing": (0.16, 0.06, 0.05, 0.0), "perfect_motion": (0.14, 0.0, 0.0, 0.0),
            "more_like_this": (0.12, 0.08, 0.0, 0.0), "great_transition": (0.10, 0.06, 0.0, 0.0),
            "no_strobes": (0.0, 0.0, -0.8, 0.0), "less_flashing": (0.0, 0.0, -0.6, 0.0),
            "strobe": (0.0, 0.0, 0.8, 0.0), "faster_strobe": (0.0, 0.0, 0.35, 0.0),
            "slower_strobe": (0.0, 0.0, -0.30, 0.0),
            "faster": (0.28, 0.0, 0.0, 0.0), "slower": (-0.28, 0.0, 0.0, 0.0),
            "brighter": (0.0, 0.25, 0.0, 0.0), "dimmer": (0.0, -0.25, 0.0, 0.0),
            "slower_side_arms": (-0.18, 0.0, 0.0, 0.0), "faster_side_arms": (0.18, 0.0, 0.0, 0.0),
            "cool_blue_purple": (0.0, 0.0, 0.0, -0.7), "warmer_color": (0.0, 0.0, 0.0, 0.7),
            "liked_this": (0.08, 0.04, 0.0, 0.0), "hold_this": (0.06, 0.02, 0.0, 0.0),
            "bad_timing": (-0.08, 0.0, 0.0, 0.0),
        }.get(label, (0.0, 0.0, 0.0, 0.0))

    @staticmethod
    def _feedback_note_effect(note: str | None) -> tuple[float, float, float, float]:
        text = (note or "").casefold()
        motion = intensity = strobe = palette = 0.0
        if any(term in text for term in ("no strobe", "stop flashing", "less flash")):
            strobe -= 0.60
        if any(term in text for term in ("slow", "calm", "too fast")):
            motion -= 0.18
        if any(term in text for term in ("faster", "pick it up", "more movement")):
            motion += 0.18
        if any(term in text for term in ("strobe", "flash")) and not any(
            term in text for term in ("no strobe", "stop flashing", "less flash")
        ):
            strobe += 0.35
        if any(term in text for term in ("blue", "purple", "cool")):
            palette -= 0.70
        if any(term in text for term in ("warm", "red", "amber")):
            palette += 0.70
        if any(term in text for term in ("brighter", "too dim")):
            intensity += 0.25
        if any(term in text for term in ("dimmer", "too bright")):
            intensity -= 0.25
        return motion, intensity, strobe, palette

    @staticmethod
    def _feedback_gesture_effect(label: str, gesture: str | None) -> dict[str, float]:
        preferred = {
            "calm_down": "breathe", "decrease_movement": "breathe",
            "too_busy": "breathe", "pick_it_up": "release",
            "increase_movement": "sweep", "not_busy_enough": "sweep",
        }.get(label)
        if preferred:
            return {preferred: 0.42}
        if gesture and label in {"liked_this", "hold_this", "great_timing", "perfect_motion", "more_like_this", "great_transition"}:
            return {gesture: 0.42}
        if gesture and label in {"bad_timing", "too_busy", "wrong_look"}:
            return {gesture: -0.42}
        return {}

    @staticmethod
    def _feedback_routine_effect(label: str, routine: str | None) -> dict[str, float]:
        preferred = {
            "calm_down": "breathe", "decrease_movement": "breathe",
            "too_busy": "breathe", "pick_it_up": "opposing_chase",
            "increase_movement": "opposing_chase", "not_busy_enough": "opposing_chase",
            "faster_side_arms": "counter_rotate", "slower_side_arms": "fan_sweep",
        }.get(label)
        if preferred:
            return {preferred: 0.50}
        if routine and label in {"liked_this", "hold_this", "great_timing", "perfect_motion", "more_like_this", "great_transition"}:
            return {routine: 0.50}
        if routine and label in {"bad_timing", "too_busy", "wrong_look"}:
            return {routine: -0.50}
        return {}

    def _rebuild_feedback_biases(self) -> None:
        """Reconstruct preferences with recency decay and agreement confidence."""
        rows = self.memory.all_feedback()
        now = time.time() * 1000.0
        buckets: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for row in rows:
            scope = row.get("scope", "overall")
            song_id = row.get("song_id")
            song_key = f"song:{song_id}" if song_id is not None else None
            song = self.memory.get_song(int(song_id)) if song_id is not None else None
            artists = [
                str(artist).casefold().strip()
                for artist in (song or {}).get("artists", ())
                if str(artist).strip()
            ]
            section = row.get("section")
            if scope == "overall":
                context_keys = ["overall"]
                if song_key:
                    context_keys.append(song_key)
                    if section:
                        context_keys.append(f"{song_key}:section:{section}")
                context_keys.extend(f"artist:{artist}" for artist in artists)
            else:
                if scope == "group" and row.get("fixture_id") == "movers":
                    target_ids = [fixture.fixture_id for fixture in self.rig.fixtures]
                else:
                    target_ids = [str(row.get("fixture_id") or "")]
                target_ids = [target for target in target_ids if target]
                context_keys = []
                for target in target_ids:
                    context_keys.append(target)
                    if song_key:
                        context_keys.append(f"{song_key}:fixture:{target}")
                        if section:
                            context_keys.append(
                                f"{song_key}:section:{section}:fixture:{target}"
                            )
                    context_keys.extend(
                        f"artist:{artist}:fixture:{target}" for artist in artists
                    )
            if not context_keys:
                continue
            age_days = max(0.0, (now - float(row.get("created_unix_ms") or now)) / 86_400_000.0)
            decay = math.exp(-age_days / 21.0)
            motion, intensity, strobe, palette = self._feedback_effect(str(row.get("label", "")))
            note_effect = self._feedback_note_effect(row.get("note"))
            motion += note_effect[0]
            intensity += note_effect[1]
            strobe += note_effect[2]
            palette += note_effect[3]
            raw_value = row.get("value")
            weight = decay * min(
                1.0,
                abs(float(1.0 if raw_value is None else raw_value)),
            )
            for key in context_keys:
                bucket = buckets.setdefault(key, {"motion": 0.0, "intensity": 0.0, "weight": 0.0, "gestures": {}, "routines": {}})
                bucket["motion"] += motion * weight
                bucket["intensity"] += intensity * weight
                bucket["strobe"] = bucket.get("strobe", 0.0) + strobe * weight
                bucket["palette"] = bucket.get("palette", 0.0) + palette * weight
                bucket["weight"] += weight
                gesture = str(row.get("gesture") or "")
                gesture_deltas = self._feedback_gesture_effect(str(row.get("label", "")), gesture)
                if gesture_deltas:
                    gestures = bucket.setdefault("gestures", {})
                    for gesture_name, gesture_delta in gesture_deltas.items():
                        gestures[gesture_name] = gestures.get(gesture_name, 0.0) + gesture_delta * weight
                routine = str(row.get("routine") or "")
                if not routine:
                    routine = {
                        "breathe": "breathe", "release": "opposing_chase",
                        "sweep": "fan_sweep", "expand": "fan_sweep",
                        "pulse": "beat_nod",
                    }.get(gesture, "")
                routine_deltas = self._feedback_routine_effect(
                    str(row.get("label", "")), routine
                )
                routines = bucket.setdefault("routines", {})
                for routine_name, routine_delta in routine_deltas.items():
                    routines[routine_name] = routines.get(routine_name, 0.0) + routine_delta * weight
                counts[key] = counts.get(key, 0) + 1
        rebuilt: dict[str, dict[str, Any]] = {}
        for key, bucket in buckets.items():
            confidence = min(1.0, math.sqrt(counts[key]) / 2.0)
            normalizer = max(float(bucket.get("weight", 0.0)), 1e-9)
            rebuilt[key] = {
                "motion": clamp(bucket["motion"] / normalizer * confidence, -1.0, 1.0),
                "intensity": clamp(bucket["intensity"] / normalizer * confidence, -1.0, 1.0),
                "strobe": clamp(bucket.get("strobe", 0.0) / normalizer * confidence, -1.0, 1.0),
                "palette": clamp(bucket.get("palette", 0.0) / normalizer * confidence, -1.0, 1.0),
                "gestures": {
                    name: clamp(value / normalizer * confidence, -1.0, 1.0)
                    for name, value in bucket.get("gestures", {}).items()
                },
                "routines": {
                    name: clamp(value / normalizer * confidence, -1.0, 1.0)
                    for name, value in bucket.get("routines", {}).items()
                },
            }
        with self._lock:
            self._feedback_biases = rebuilt

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

    def _load_motion_tunings(self) -> dict[str, MotionTuning]:
        try:
            payload = json.loads(self.motion_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except (OSError, ValueError, TypeError):
            payload = {}
        return merged_motion_tunings(payload if isinstance(payload, dict) else {})

    def _save_motion_tunings(self) -> None:
        self.motion_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.motion_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    routine: tuning.as_dict()
                    for routine, tuning in self.motion_tunings.items()
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.motion_path)

    def patch_motion_routine(self, payload: dict[str, Any]) -> dict[str, Any]:
        routine = str(payload.get("routine", "")).strip().lower()
        if routine not in DEFAULT_MOTION_TUNINGS:
            raise ValueError("unknown motion routine")
        with self._lock:
            if str(payload.get("action", "")).lower() == "reset":
                self.motion_tunings[routine] = DEFAULT_MOTION_TUNINGS[routine]
            else:
                values = payload.get("values", payload)
                if not isinstance(values, dict):
                    raise ValueError("motion values must be an object")
                self.motion_tunings[routine] = self.motion_tunings[routine].patch(values)
            self._save_motion_tunings()
            if self._runtime is not None:
                self._runtime.set_motion_tunings(self.motion_tunings)
            self._add_event("rehearsal", f"Tuned {routine.replace('_', ' ')} motion")
            self._status_sequence += 1
        return self.snapshot()

    def _motion_editor_snapshot(self) -> dict[str, Any]:
        routine = self.rehearsal.routine
        tuning = self.motion_tunings[routine]
        diagnostics = []
        feasible = True
        for index, fixture in enumerate(self.rig.fixtures):
            calibration = fixture.calibration
            pan_speed, tilt_speed = required_axis_speeds(
                routine,
                tuning,
                bpm=self.rehearsal.bpm,
                fixture_index=index,
                fixture_count=len(self.rig.fixtures),
                pan_range_deg=calibration.pan_max_deg - calibration.pan_min_deg,
                tilt_range_deg=calibration.tilt_max_deg - calibration.tilt_min_deg,
                size=self.rehearsal.size,
            )
            within = (
                pan_speed <= calibration.max_pan_speed_deg_s
                and tilt_speed <= calibration.max_tilt_speed_deg_s
            )
            feasible = feasible and within
            diagnostics.append({
                "fixture_id": fixture.fixture_id,
                "address": fixture.address,
                "required_pan_deg_s": pan_speed,
                "required_tilt_deg_s": tilt_speed,
                "maximum_pan_deg_s": calibration.max_pan_speed_deg_s,
                "maximum_tilt_deg_s": calibration.max_tilt_speed_deg_s,
                "within_velocity": within,
            })
        return {
            "values": tuning.as_dict(),
            "defaults": DEFAULT_MOTION_TUNINGS[routine].as_dict(),
            "modified": tuning != DEFAULT_MOTION_TUNINGS[routine],
            "paths": preview_paths(routine, tuning),
            "path": str(self.motion_path),
            "velocity_feasible": feasible,
            "velocity": diagnostics,
        }

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
        if normalized not in {"monitor", "live", "demo", "rehearsal"}:
            raise ValueError("mode must be monitor, live, demo, or rehearsal")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self.engine_mode == normalized:
                    return self.snapshot()
                raise RuntimeError(
                    f"Lumen is already running in {self.engine_mode} mode; stop it first"
                )
            research_worker = self._research_worker_thread
            if research_worker is not None and research_worker.is_alive():
                action = (
                    "wait for it to pause"
                    if self._research_cancel.is_set()
                    else "pause it in Audio Laboratory first"
                )
                raise RuntimeError(
                    f"offline analysis is still running; {action}"
                )
            preparation = self._training_prepare_thread
            if preparation is not None and preparation.is_alive():
                raise RuntimeError(
                    "Lumen is preparing the last audio capture; wait for "
                    "Audio Laboratory to report that preparation is complete"
                )
            self._stop.clear()
            self.engine_mode = normalized
            self.engine_phase = "starting"
            self.last_error = None
            self.started_at = time.monotonic()
            self._session_id = f"{int(time.time() * 1000)}:{normalized}"
            self._last_trace_timestamp = None
            self._reset_audio_diagnostics()
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
        self._research_cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Audio chunks are short; allow the owner thread to finalize its
            # recorder before closing the DMX object it is still using.
            thread.join(timeout=15.0)
        output = self._output
        if output is not None:
            try:
                output.close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        recorder = self._training_recorder
        if recorder is not None:
            recorder.stop()
        self._trace_stop.set()
        self._trace_thread.join(timeout=10.0)
        preparation = self._training_prepare_thread
        if preparation is not None and preparation.is_alive():
            preparation.join(timeout=15.0)
        research_worker = self._research_worker_thread
        if research_worker is not None and research_worker.is_alive():
            research_worker.join(timeout=15.0)

    def _trace_worker(self) -> None:
        while not self._trace_stop.is_set() or not self._trace_queue.empty():
            try:
                item = self._trace_queue.get(timeout=0.20)
            except queue.Empty:
                continue
            try:
                kind = item.pop("_kind", "performance")
                if kind == "decision":
                    self.memory.log_decision(**item)
                else:
                    self.memory.log_performance_sample(**item)
            except Exception as error:
                with self._lock:
                    self._add_event("memory", f"Performance trace write failed: {error}")
            finally:
                self._trace_queue.task_done()

    def _run(self, mode: str) -> None:
        raw_output: VirtualDMXOutput | OpenDmxUsbOutput
        runtime: PerformanceRuntime | None = None
        gated: GatedOutput | None = None
        try:
            raw_output = (
                OpenDmxUsbOutput.open()
                if mode == "live"
                or (
                    mode == "rehearsal"
                    and self.rehearsal.output == "live"
                )
                else VirtualDMXOutput()
            )
            gated = GatedOutput(raw_output, self.controls)
            runtime = self._runtime_for_rig(gated)
            with self._lock:
                self._output = gated
                self._runtime = runtime
                self.engine_phase = (
                    "listening"
                    if mode == "monitor"
                    else "rehearsing"
                    if mode == "rehearsal"
                    else "performing"
                )
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
            elif mode == "rehearsal":
                self._run_rehearsal(runtime)
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
        self._prepare_dedicated_line_input()
        capture_config = AudioCaptureConfig(device=self.audio_device)
        analyzer = RealtimeAudioAnalyzer(
            capture_config.sample_rate, capture_config.channels
        )
        analysis_generation = self._analysis_generation
        sample_clock_origin = time.monotonic()
        captured_frames = 0
        recorder: TrainingDataRecorder | None = None
        if self.training_capture_enabled:
            recorder = TrainingDataRecorder(
                self.memory,
                session_id=self._session_id,
                mode=self.engine_mode,
                config=TrainingCaptureConfig(
                    root=self.training_root,
                    sample_rate=capture_config.sample_rate,
                    channels=capture_config.channels,
                    segment_seconds=60,
                    feature_rate_hz=10.0,
                    max_bytes=round(self.training_max_gb * 1024**3),
                ),
                metadata={
                    "schema": "lumen_training_session_v1",
                    "audio_device": self.audio_device,
                    "rig_name": self.rig.name,
                    "rig_snapshot": self._rig_payload,
                    "semantic_target_notice": (
                        "Runtime output is baseline context. Operator feedback "
                        "and preferred actions provide training supervision."
                    ),
                },
            )
            recorder.start()
            with self._lock:
                self._training_recorder = recorder
                self._training_audio_frame = None
                self._training_linked_feedback = 0
                self._training_annotations = 0
                state = recorder.status()
                if state["state"] == "recording":
                    self._add_event(
                        "memory",
                        "Lossless training audio capture started",
                    )
                else:
                    self._add_event(
                        "memory",
                        state["error"] or "Training capture did not start",
                    )
        try:
            with AlsaLineIn(capture_config) as capture:
                for pcm in capture.chunks():
                    if self._stop.is_set():
                        break
                    if analysis_generation != self._analysis_generation:
                        analyzer.reset()
                        if self._student_model is not None:
                            self._student_model.reset()
                            self._student_decoder.reset()
                        analysis_generation = self._analysis_generation
                    packet_frames = len(pcm) // (
                        2 * capture_config.channels
                    )
                    timestamp = (
                        sample_clock_origin
                        + (captured_frames + packet_frames / 2.0)
                        / capture_config.sample_rate
                    )
                    observation = analyzer.analyze_pcm16(
                        pcm, timestamp_s=timestamp
                    )
                    audio_metrics = analyzer.last_metrics
                    observation = self._apply_student_structure(
                        observation, audio_metrics
                    )
                    if self._student_prediction is not None:
                        runtime.set_structure_context(
                            **self._student_runtime_context()
                        )
                    captured_frames += packet_frames
                    frame = runtime.step(observation)
                    audio_frame: int | None = None
                    if recorder is not None:
                        audio_frame = recorder.submit(
                            pcm,
                            song_id=self.song_id,
                            position_ms=self._media_position_ms(),
                            payload=lambda: self._training_frame_payload(
                                observation, frame, audio_metrics
                            ),
                        )
                    self._accept_runtime_frame(
                        observation,
                        frame,
                        audio_metrics=audio_metrics,
                        audio_bytes=len(pcm),
                        training_audio_frame=audio_frame,
                    )
        finally:
            if recorder is not None:
                recorder.stop()
                with self._lock:
                    final_status = recorder.status()
                    self._training_history = self.memory.training_summary()
                    self._training_recorder = None
                    self._training_audio_frame = None
                    self._training_linked_feedback = 0
                    self._training_annotations = 0
                    self._add_event(
                        "memory",
                        (
                            "Training audio finalized"
                            if final_status["state"] == "complete"
                            else final_status["error"]
                            or f"Training capture ended: {final_status['state']}"
                        ),
                    )
                if (
                    final_status["state"] in {"complete", "quota"}
                    and int(final_status.get("frames_written", 0)) > 0
                ):
                    self._start_training_preparation()

    def _start_training_preparation(self) -> None:
        """Build verified recording identities and queue teachers off-thread."""
        with self._lock:
            running = self._training_prepare_thread
            if running is not None and running.is_alive():
                self._training_prepare_pending = True
                self._add_event(
                    "memory",
                    "Offline preparation is already processing a prior capture",
                )
                return
            thread = threading.Thread(
                target=self._prepare_training_capture,
                name="lumen-offline-preparation",
                daemon=True,
            )
            self._training_prepare_thread = thread
            thread.start()

    def _prepare_training_capture(self) -> None:
        try:
            result = export_training_dataset(
                self.memory, self.training_root
            )
            research = ResearchJobCoordinator(
                self.memory,
                training_root=self.training_root,
                research_root=self.training_root / "research",
            ).prepare_export(result["path"], queue_songformer=True)
            with self._lock:
                self._last_training_export = result["path"]
                self._training_history = self.memory.training_summary()
                self._add_event(
                    "memory",
                    (
                        "Prepared captured songs and queued "
                        f"{research['jobs_queued']} offline teacher job(s)"
                    ),
                )
                self._status_sequence += 1
        except Exception as error:
            with self._lock:
                self._add_event(
                    "memory",
                    f"Offline capture preparation failed: {error}",
                )
                self._status_sequence += 1
        finally:
            with self._lock:
                rerun = self._training_prepare_pending
                self._training_prepare_pending = False
                self._training_prepare_thread = None
            if rerun and not self._stop.is_set():
                self._start_training_preparation()

    def _training_frame_payload(
        self,
        observation: MusicalObservation,
        frame: RuntimeFrame,
        audio_metrics: AudioInputMetrics | None = None,
    ) -> dict[str, Any]:
        decision = asdict(frame.decision)
        decision["gesture"] = frame.decision.gesture.value
        media: dict[str, Any] | None = None
        if self.media is not None:
            media = {
                "provider": self.media.provider,
                "provider_item_id": self.media.provider_item_id,
                "title": self.media.title,
                "artists": list(self.media.artists),
                "album": self.media.album,
                "duration_ms": self.media.duration_ms,
                "is_playing": self.media.is_playing,
                "device_name": self.media.device_name,
            }
        semantic_payload = (
            self._semantic_audio_payload(observation, audio_metrics)
            if audio_metrics is not None
            else {"observation": asdict(observation)}
        )
        return {
            "schema": "lumen_semantic_frame_v1",
            "observation": semantic_payload["observation"],
            "audio_metrics": semantic_payload.get("audio_metrics", {}),
            "decision": decision,
            "controls": asdict(self.controls),
            "media": media,
            "structure_model": self._student_prediction,
            "solutions": [
                {
                    "fixture_id": solution.fixture_id,
                    "pan_deg": solution.pan_deg,
                    "tilt_deg": solution.tilt_deg,
                    "branch": solution.branch,
                }
                for solution in frame.solutions
            ],
            "fixture_dmx": self._fixture_dmx_snapshot(frame),
            "choreography_runtime": (
                self._runtime.choreography_snapshot()
                if self._runtime is not None
                else None
            ),
            "target_provenance": (
                "heuristic_runtime_baseline_not_ground_truth"
            ),
        }

    def _fixture_dmx_snapshot(
        self, frame: RuntimeFrame
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for fixture in (*self.rig.fixtures, *self.rig.auxiliary_fixtures):
            profile = party_parrot_profile(fixture.profile_key)
            footprint = profile.dmx_footprint if profile is not None else 1
            universe = frame.dmx.universe_data(fixture.universe)
            start = fixture.address - 1
            result.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "profile_key": fixture.profile_key,
                    "universe": fixture.universe,
                    "address": fixture.address,
                    "channels": list(
                        universe[start : start + footprint]
                    ),
                }
            )
        return result

    def _prepare_dedicated_line_input(self) -> None:
        """Apply this controller PC's known-good Realtek line-input baseline."""

        if self.audio_device != "default" or shutil.which("amixer") is None:
            return
        commands = (
            ["amixer", "-q", "-c", "0", "sset", "Input Source", "Line"],
            ["amixer", "-q", "-c", "0", "sset", "Capture", "0dB"],
        )
        for command in commands:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()
                with self._lock:
                    self._add_event(
                        "audio",
                        "Mixer baseline could not be applied"
                        + (f": {detail}" if detail else ""),
                    )
                return
        with self._lock:
            self._add_event(
                "audio",
                "Prepared Realtek Line input at 0 dB capture gain",
            )

    def _run_demo(self, runtime: PerformanceRuntime) -> None:
        index = 0
        while not self._stop.wait(0.12):
            observation = _demo_observation(index)
            self._accept_runtime_frame(observation, runtime.step(observation))
            index += 1

    def _run_rehearsal(self, runtime: PerformanceRuntime) -> None:
        """Audition authored routines against a stable generated beat clock."""
        started = time.monotonic()
        active_routine: str | None = None
        while not self._stop.wait(1.0 / 24.0):
            elapsed = time.monotonic() - started
            with self._lock:
                bpm = self.rehearsal.bpm
                intensity = self.rehearsal.intensity
                requested = self.rehearsal.routine
                if self.rehearsal.tour:
                    beats = elapsed * bpm / 60.0
                    routine_index = int(beats // 8) % len(REHEARSAL_ROUTINES)
                    requested = REHEARSAL_ROUTINES[routine_index]["id"]
                    self.rehearsal.routine = requested
                if requested != active_routine:
                    active_routine = requested
                    self._add_event(
                        "rehearsal",
                        f"Auditioning {requested.replace('_', ' ')}",
                    )
                self._apply_rehearsal_to_runtime(runtime)
            observation = _rehearsal_observation(elapsed, bpm, intensity)
            self._accept_runtime_frame(observation, runtime.step(observation))

    def _accept_runtime_frame(
        self,
        observation: MusicalObservation,
        frame: RuntimeFrame,
        *,
        audio_metrics: AudioInputMetrics | None = None,
        audio_bytes: int = 0,
        training_audio_frame: int | None = None,
    ) -> None:
        with self._lock:
            if audio_metrics is not None:
                packet_time = time.monotonic()
                if self._audio_capture_started_at is None:
                    self._audio_capture_started_at = packet_time
                self._audio_last_packet_at = packet_time
                self._audio_packet_times.append(packet_time)
                self._audio_packets += 1
                self._audio_frames += audio_metrics.frame_count
                self._audio_bytes += audio_bytes
                self._audio_metrics = audio_metrics
            self._training_audio_frame = training_audio_frame
            self.observation = observation
            self.frame = frame
            if (
                self._last_analysis_history_at is None
                or observation.timestamp_s - self._last_analysis_history_at >= 0.095
            ):
                self._last_analysis_history_at = observation.timestamp_s
                self._analysis_history.append(
                    {
                        "timestamp_s": observation.timestamp_s,
                        "dbfs": (
                            audio_metrics.dbfs
                            if audio_metrics is not None
                            else -120.0
                        ),
                        "loudness": observation.loudness,
                        "onset": observation.onset_strength,
                        "section": observation.section,
                        "section_confidence": observation.section_confidence,
                        "bpm": observation.bpm,
                        "beat_confidence": observation.beat_confidence,
                        "energy": frame.decision.expression.energy,
                        "motion": frame.decision.expression.motion,
                        "gesture": frame.decision.gesture.value,
                        "routine": frame.decision.routine,
                    }
                )
            if (
                self._last_trace_timestamp is None
                or observation.timestamp_s - self._last_trace_timestamp >= 0.48
            ):
                self._last_trace_timestamp = observation.timestamp_s
                trace_decision = asdict(frame.decision)
                trace_decision["gesture"] = frame.decision.gesture.value
                try:
                    self._trace_queue.put_nowait(
                        {
                            "_kind": "performance",
                            "session_id": self._session_id,
                            "song_id": self.song_id,
                            "position_ms": self._media_position_ms(),
                            "payload": {
                                "observation": asdict(observation),
                                "decision": trace_decision,
                                "controls": asdict(self.controls),
                                "solutions": [
                                    {
                                        "fixture_id": solution.fixture_id,
                                        "pan_deg": solution.pan_deg,
                                        "tilt_deg": solution.tilt_deg,
                                        "branch": solution.branch,
                                    }
                                    for solution in frame.solutions
                                ],
                                "choreography_runtime": (
                                    self._runtime.choreography_snapshot()
                                    if self._runtime is not None
                                    else None
                                ),
                            },
                        }
                    )
                except queue.Full:
                    pass
            gesture = frame.decision.gesture.value
            if gesture != self._last_gesture:
                self._last_gesture = gesture
                self._add_event(
                    "gesture",
                    f"{gesture.title()}: {frame.decision.reason.split('.')[0]}.",
                )
                if self.song_id is not None:
                    try:
                        self._trace_queue.put_nowait(
                            {
                                "_kind": "decision",
                                "decision": frame.decision,
                                "song_id": self.song_id,
                                "position_ms": self._media_position_ms(),
                                "observation": observation,
                            }
                        )
                    except queue.Full:
                        pass
            self._status_sequence += 1
        self._schedule_spotify_poll()

    def _schedule_spotify_poll(self) -> None:
        """Keep network metadata work off the real-time audio/DMX loop."""
        thread = self._spotify_poll_thread
        if thread is not None and thread.is_alive():
            return
        if time.monotonic() - self._last_media_poll < 5.0:
            return
        self._spotify_poll_thread = threading.Thread(
            target=self._poll_spotify_if_due,
            name="lumen-spotify-poll",
            daemon=True,
        )
        self._spotify_poll_thread.start()

    def _reset_audio_diagnostics(self) -> None:
        self._audio_metrics = AudioInputMetrics.silence()
        self._audio_packets = 0
        self._audio_frames = 0
        self._audio_bytes = 0
        self._audio_capture_started_at = None
        self._audio_last_packet_at = None
        self._audio_packet_times.clear()
        self._analysis_history.clear()
        self._last_analysis_history_at = None

    def _audio_snapshot_unlocked(self, running: bool) -> dict[str, Any]:
        now = time.monotonic()
        last_age_ms = (
            None
            if self._audio_last_packet_at is None
            else max(0.0, (now - self._audio_last_packet_at) * 1000.0)
        )
        packet_rate_hz = 0.0
        if len(self._audio_packet_times) >= 2:
            elapsed = self._audio_packet_times[-1] - self._audio_packet_times[0]
            if elapsed > 0:
                packet_rate_hz = (
                    len(self._audio_packet_times) - 1
                ) / elapsed

        if running and self.engine_mode == "rehearsal":
            state = "simulated"
            label = "REHEARSAL — GENERATED CLOCK"
            detail = (
                "A stable beat clock is driving one selected lighting routine; "
                "line input is intentionally not part of this audition."
            )
        elif running and self.engine_mode == "demo":
            state = "simulated"
            label = "DEMO — NO PHYSICAL INPUT"
            detail = "The interface is being driven by generated observations."
        elif not running:
            state = "inactive"
            label = "INPUT TEST NOT RUNNING"
            detail = "Start Monitor to test the physical line input without DMX output."
        elif self._audio_packets == 0:
            state = "waiting"
            label = "WAITING FOR PCM"
            detail = f"ALSA is opening {self.audio_device}; no packet has arrived yet."
        elif last_age_ms is not None and last_age_ms > 750:
            state = "stale"
            label = "PCM STREAM STALLED"
            detail = f"The last audio packet arrived {last_age_ms / 1000.0:.1f}s ago."
        elif self._audio_metrics.clipped_samples:
            state = "clipping"
            label = "SIGNAL CLIPPING"
            detail = (
                f"{self._audio_metrics.clipped_samples} samples hit full scale "
                "in the latest packet."
            )
        elif self._audio_metrics.dbfs > -55.0:
            state = "signal"
            label = "PHYSICAL SIGNAL DETECTED"
            detail = (
                "Fresh PCM is arriving from the selected ALSA line input and "
                "is being analyzed."
            )
        elif self.media is not None and self.media.is_playing:
            state = "missing"
            label = "SPOTIFY PLAYING — NO LINE SIGNAL"
            detail = (
                "Spotify reports active playback, but the physical line input "
                f"is only {self._audio_metrics.dbfs:.1f} dBFS. Lumen cannot "
                "synchronize lighting until the splitter signal reaches this input."
            )
        else:
            state = "quiet"
            label = "PCM LIVE — INPUT QUIET"
            detail = (
                "Fresh PCM is arriving, but its current level is below "
                "−55 dBFS."
            )
        return {
            "state": state,
            "label": label,
            "detail": detail,
            "source": "demo" if self.engine_mode == "demo" else "line-in",
            "packets_received": self._audio_packets,
            "frames_received": self._audio_frames,
            "bytes_received": self._audio_bytes,
            "packet_rate_hz": packet_rate_hz,
            "expected_packet_rate_hz": 48_000 / 2_048,
            "last_packet_age_ms": last_age_ms,
            "capture_uptime_s": (
                None
                if self._audio_capture_started_at is None
                else max(0.0, now - self._audio_capture_started_at)
            ),
            "metrics": asdict(self._audio_metrics),
        }

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
        runtime = PerformanceRuntime(
            self.rig.fixtures,
            output,
            auxiliary_fixtures=self.rig.auxiliary_fixtures,
            expression=OperatorExpressionEngine(self.controls, policy),
            motion_extents=Vec3(
                max(0.8, self.rig.room.width_m * 0.44),
                max(1.8, self.rig.room.depth_m * 0.33),
                min(2.6, max(2.2, self.rig.room.height_m * 0.45)),
            ),
            choreography_model=self._choreography_model,
            motion_tunings=self.motion_tunings,
        )
        runtime.replace_feedback(self._feedback_biases)
        runtime.set_media_context(
            self.song_id, self.observation.section,
            self.media.artists[0] if self.media and self.media.artists else None,
        )
        for fixture_id, override in self._calibration_overrides.items():
            runtime.set_calibration_override(fixture_id, active=True, **override)
        return runtime

    def calibration_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        fixture_id = str(payload.get("fixture_id", "")).strip()
        if not any(f.fixture_id == fixture_id for f in self.rig.fixtures):
            raise ValueError("calibration requires a moving-head fixture")
        active = bool(payload.get("active", True))
        if active:
            override = {
                "pan_dmx": clamp(float(payload.get("pan_dmx", 128)), 0, 255) * 257.0,
                "tilt_dmx": clamp(float(payload.get("tilt_dmx", 128)), 0, 255) * 257.0,
                "speed": clamp(float(payload.get("speed", 192)), 0, 255),
            }
            self._calibration_overrides[fixture_id] = override
        else:
            override = self._calibration_overrides.pop(fixture_id, None) or {}
        runtime = self._runtime
        if runtime is not None:
            runtime.set_calibration_override(fixture_id, active=active, **override)
        self._add_event("rig", f"Calibration {'active' if active else 'stopped'}: {fixture_id[:8]}")
        self._status_sequence += 1
        return {"active": active, "fixture_id": fixture_id, **override}

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

    def patch_rehearsal(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            candidate = replace(self.rehearsal)
            candidate.patch(values)
            if candidate.scope.startswith("fixture:"):
                fixture_id = candidate.scope[8:]
                if not any(
                    fixture.fixture_id == fixture_id
                    for fixture in (*self.rig.fixtures, *self.rig.auxiliary_fixtures)
                ):
                    raise ValueError("rehearsal fixture is not in the active rig")
            running_rehearsal = (
                self._thread is not None
                and self._thread.is_alive()
                and self.engine_mode == "rehearsal"
            )
            if (
                running_rehearsal
                and candidate.output != self.rehearsal.output
            ):
                raise RuntimeError(
                    "stop rehearsal before changing preview/live output"
                )
            self.rehearsal = candidate
            if self._runtime is not None and running_rehearsal:
                self._apply_rehearsal_to_runtime(self._runtime)
            self._add_event(
                "rehearsal",
                (
                    f"{self.rehearsal.routine.replace('_', ' ').title()} · "
                    f"{self.rehearsal.scope} · {round(self.rehearsal.bpm)} BPM"
                ),
            )
            self._status_sequence += 1
        return self.snapshot()

    def _apply_rehearsal_to_runtime(
        self, runtime: PerformanceRuntime
    ) -> None:
        runtime.set_rehearsal(
            self.rehearsal.routine,
            scope=self.rehearsal.scope,
            intensity=self.rehearsal.intensity,
            size=self.rehearsal.size,
            palette=self.rehearsal.palette,
            strobe=self.rehearsal.strobe,
            isolate=self.rehearsal.isolate,
        )

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
                "intensity": 0.62,
                "motion": 0.68,
                "focus": 0.50,
                "warmth": 0.44,
                "influence": 0.74,
            },
            "open": {
                "intensity": 0.62,
                "motion": 0.56,
                "focus": 0.20,
                "warmth": 0.55,
                "influence": 0.72,
            },
            "drive": {
                "intensity": 0.90,
                "motion": 0.94,
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
            if (
                "training_capture_enabled" in payload
                or "training_max_gb" in payload
            ):
                if self._thread is not None and self._thread.is_alive():
                    raise RuntimeError(
                        "stop the engine before changing training capture settings"
                    )
                if "training_capture_enabled" in payload:
                    self.training_capture_enabled = bool(
                        payload["training_capture_enabled"]
                    )
                    self._settings["training_capture_enabled"] = (
                        self.training_capture_enabled
                    )
                if "training_max_gb" in payload:
                    training_max_gb = float(payload["training_max_gb"])
                    if not 1.0 <= training_max_gb <= 800.0:
                        raise ValueError(
                            "training storage limit must be between 1 and 800 GB"
                        )
                    self.training_max_gb = training_max_gb
                    self._settings["training_max_gb"] = training_max_gb
                self._add_event(
                    "memory",
                    (
                        "Training audio capture enabled"
                        if self.training_capture_enabled
                        else "Training audio capture disabled"
                    ),
                )
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
            "training_capture_enabled": self.training_capture_enabled,
            "training_max_gb": self.training_max_gb,
            "training_path": str(self.training_root),
        }

    def export_training_data(self) -> dict[str, Any]:
        with self._lock:
            recorder = self._training_recorder
            if recorder is not None and recorder.status()["recording"]:
                raise RuntimeError(
                    "stop Monitor or Live mode before building a training export"
                )
            preparation = self._training_prepare_thread
            if (
                preparation is not None
                and preparation.is_alive()
                and preparation is not threading.current_thread()
            ):
                raise RuntimeError(
                    "the last audio capture is still being prepared; try again "
                    "when Audio Laboratory reports that preparation is complete"
                )
        if not self._training_export_lock.acquire(blocking=False):
            raise RuntimeError(
                "a training dataset build is already in progress"
            )
        try:
            result = export_training_dataset(self.memory, self.training_root)
            research = ResearchJobCoordinator(
                self.memory,
                training_root=self.training_root,
                research_root=self.training_root / "research",
            ).prepare_export(result["path"], queue_songformer=True)
            result["research"] = research
            with self._lock:
                self._last_training_export = result["path"]
                self._training_history = self.memory.training_summary()
                self._add_event(
                    "memory", "Built neural-training dataset manifest"
                )
                self._status_sequence += 1
            return result
        finally:
            self._training_export_lock.release()

    def research_status(self) -> dict[str, Any]:
        result = self.research.status()
        training = training_readiness(
            self.memory,
            research_root=self.training_root / "research",
        )
        with self._lock:
            worker = self._research_worker_thread
            worker_running = bool(worker is not None and worker.is_alive())
            preparation = self._training_prepare_thread
            preparation_running = bool(
                preparation is not None and preparation.is_alive()
            )
            running_jobs = [
                job
                for job in self.memory.list_analysis_jobs(limit=100_000)
                if job["status"] == "running"
            ]
            current_job = running_jobs[0] if running_jobs else None
            progress = dict(self._research_worker_progress)
            progress["current_job_type"] = (
                current_job["job_type"] if current_job is not None else None
            )
            progress["current_recording_id"] = (
                current_job.get("payload", {}).get("recording_id")
                if current_job is not None
                else None
            )
            progress["resources"] = (
                deepcopy(current_job.get("result") or {})
                if current_job is not None
                else {}
            )
            cancel_requested = bool(
                worker_running and self._research_cancel.is_set()
            )
            result["worker"] = {
                "running": worker_running,
                "phase": (
                    "pausing"
                    if cancel_requested
                    else "running"
                    if worker_running
                    else "idle"
                ),
                "cancel_requested": cancel_requested,
                "last_result": deepcopy(self._research_worker_last),
                "recovered_jobs": deepcopy(self._research_recovered),
                "progress": progress,
            }
            result["preparation"] = {
                "running": preparation_running,
                "pending": bool(self._training_prepare_pending),
            }
            runtime_state = (
                "active"
                if self._student_model is not None
                else "error"
                if self._student_model_error
                else "awaiting_training"
            )
            model = training.setdefault("model", {})
            model["artifact_present"] = bool(model.get("active"))
            model["active"] = self._student_model is not None
            model["runtime_state"] = runtime_state
            model["runtime_error"] = self._student_model_error
        result["training"] = training
        return result

    def _recover_abandoned_research_jobs(self) -> list[dict[str, Any]]:
        """Make interrupted durable jobs resumable before queue inspection."""
        recovered = self.memory.recover_abandoned_analysis_jobs()
        if not recovered:
            return []
        with self._lock:
            for item in recovered:
                if item not in self._research_recovered:
                    self._research_recovered.append(deepcopy(item))
            self._add_event(
                "memory",
                (
                    f"Recovered {len(recovered)} interrupted offline "
                    "analysis job(s); they are ready to resume"
                ),
            )
            self._status_sequence += 1
        return recovered

    def analyze_training_data(self) -> dict[str, Any]:
        """Prepare all captures, queue both teachers, and resume the batch."""
        # Reject a duplicate/unsafe request before reconstructing WAV files or
        # rebuilding the export.  The browser normally disables this button,
        # but the HTTP server is threaded and a second client (or double tap)
        # can still submit the endpoint directly.
        with self._lock:
            engine = self._thread
            if (
                self.engine_phase not in {"ready", "fault"}
                or (engine is not None and engine.is_alive())
            ):
                raise RuntimeError(
                    "stop Monitor or Live before running an offline teacher"
                )
            worker = self._research_worker_thread
            if worker is not None and worker.is_alive():
                raise RuntimeError("an offline research job is already running")
            preparation = self._training_prepare_thread
            if preparation is not None and preparation.is_alive():
                raise RuntimeError(
                    "the last audio capture is still being prepared; wait for "
                    "Audio Laboratory to report that preparation is complete"
                )
        # Recovery must precede export preparation and the queued-job count;
        # otherwise a crash-stranded `running` row can make Analyze appear to
        # have no resumable work.
        self._recover_abandoned_research_jobs()
        export = self.export_training_data()
        queued = sum(
            job["status"] == "queued"
            and job["job_type"] in {EDMFORMER_JOB, SONGFORMER_JOB}
            for job in self.memory.list_analysis_jobs(limit=100_000)
        )
        if not queued:
            return {
                "export": export,
                "research": self.research_status(),
                "started": False,
                "message": (
                    "No new or retryable full-song recordings are queued for "
                    "structure analysis."
                ),
            }
        research = self.start_research_worker(
            {"job_types": [EDMFORMER_JOB, SONGFORMER_JOB]}
        )
        return {
            "export": export,
            "research": research,
            "started": True,
        }

    def start_research_worker(
        self, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = payload or {}
        requested_types = payload.get("job_types") or [
            EDMFORMER_JOB,
            SONGFORMER_JOB,
        ]
        allowed_types = {EDMFORMER_JOB, SONGFORMER_JOB, STUDENT_TRAIN_JOB}
        job_types = tuple(str(item) for item in requested_types)
        if not job_types or any(item not in allowed_types for item in job_types):
            raise ValueError("invalid offline research job type")
        maximum_jobs = payload.get("maximum_jobs")
        if maximum_jobs is not None:
            maximum_jobs = int(maximum_jobs)
            if not 1 <= maximum_jobs <= 10_000:
                raise ValueError("maximum_jobs must be between 1 and 10000")
        with self._lock:
            engine = self._thread
            if (
                self.engine_phase not in {"ready", "fault"}
                or (engine is not None and engine.is_alive())
            ):
                raise RuntimeError(
                    "stop Monitor or Live before running an offline teacher"
                )
            worker = self._research_worker_thread
            if worker is not None and worker.is_alive():
                raise RuntimeError("an offline research job is already running")
            preparation = self._training_prepare_thread
            if preparation is not None and preparation.is_alive():
                raise RuntimeError(
                    "the last audio capture is still being prepared; wait for "
                    "Audio Laboratory to report that preparation is complete"
                )
            self._recover_abandoned_research_jobs()
            available = sum(
                job["status"] == "queued" and job["job_type"] in job_types
                for job in self.memory.list_analysis_jobs(limit=100_000)
            )
            if not available:
                raise RuntimeError("there are no queued matching research jobs")
            self._research_worker_last = None
            self._research_worker_progress = {
                "processed": 0,
                "failed": 0,
                "planned": min(available, maximum_jobs or available),
                "current_job_type": None,
                "job_types": list(job_types),
            }
            self._research_cancel.clear()
            thread = threading.Thread(
                target=self._run_research_batch,
                args=(job_types, maximum_jobs),
                name="lumen-research-worker",
                daemon=True,
            )
            self._research_worker_thread = thread
            thread.start()
            self._add_event(
                "memory",
                f"Started offline analysis batch ({min(available, maximum_jobs or available)} job(s))",
            )
            self._status_sequence += 1
        return self.research_status()

    def cancel_research_worker(self) -> dict[str, Any]:
        worker = self._research_worker_thread
        if worker is None or not worker.is_alive():
            raise RuntimeError("no offline research batch is running")
        self._research_cancel.set()
        with self._lock:
            self._add_event("memory", "Pausing offline analysis after cancellation")
            self._status_sequence += 1
        return self.research_status()

    def train_structure_student(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        epochs = int(payload.get("epochs", 30))
        if not 1 <= epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        with self._lock:
            if self.engine_phase not in {"ready", "fault"}:
                raise RuntimeError(
                    "stop Monitor or Live before training the CPU student"
                )
            worker = self._research_worker_thread
            if worker is not None and worker.is_alive():
                raise RuntimeError("an offline research job is already running")
        queued = enqueue_student_training(
            self.memory,
            research_root=self.training_root / "research",
            epochs=epochs,
        )
        status = self.start_research_worker(
            {"job_types": [STUDENT_TRAIN_JOB], "maximum_jobs": 1}
        )
        return {"queued": queued, "research": status}

    def _run_research_batch(
        self,
        job_types: tuple[str, ...],
        maximum_jobs: int | None,
    ) -> None:
        results: list[dict[str, Any]] = []
        recovered_jobs: list[dict[str, Any]] = []
        fatal_error: str | None = None
        try:
            worker = OfflineResearchWorker(
                self.memory,
                research_root=self.training_root / "research",
                cancel_event=self._research_cancel,
            )
            while maximum_jobs is None or len(results) < maximum_jobs:
                if self._research_cancel.is_set():
                    break
                result = worker.run_once(job_types)
                with self._lock:
                    for recovered in getattr(worker, "last_recovery", ()):
                        if recovered not in recovered_jobs:
                            recovered_jobs.append(deepcopy(recovered))
                        if recovered not in self._research_recovered:
                            self._research_recovered.append(deepcopy(recovered))
                if result is None:
                    break
                results.append(result)
                with self._lock:
                    self._research_worker_progress["processed"] = len(results)
                    self._research_worker_progress["failed"] = sum(
                        item["status"] == "failed" for item in results
                    )
                    self._research_worker_progress["current_job_type"] = result[
                        "job_type"
                    ]
                if result["status"] == "canceled":
                    break
        except Exception as error:
            fatal_error = f"{type(error).__name__}: {error}"
        result = results[-1] if results else None
        with self._lock:
            self._research_worker_last = {
                "jobs": results,
                "processed": len(results),
                "failed": (
                    sum(item["status"] == "failed" for item in results)
                    + int(fatal_error is not None)
                ),
                "canceled": self._research_cancel.is_set(),
                "error": fatal_error,
                "recovered_jobs": recovered_jobs,
            }
            if (
                result is not None
                and result["status"] == "complete"
                and result["job_type"] == STUDENT_TRAIN_JOB
                and bool((result.get("result") or {}).get("activated", True))
            ):
                self._load_student_model()
            if fatal_error is not None:
                message = f"Offline analysis stopped unexpectedly: {fatal_error}"
            elif result is None:
                message = "Offline research queue was empty"
            elif result["status"] == "canceled":
                message = "Offline research job canceled and returned to queue"
            elif any(item["status"] == "failed" for item in results):
                message = (
                    f"Offline analysis batch finished with {self._research_worker_last['failed']} failure(s)"
                )
            else:
                message = f"Offline analysis batch complete: {len(results)} job(s)"
            self._add_event("memory", message)
            self._status_sequence += 1

    def provision_research_sources(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        raw_components = payload.get("components")
        if raw_components is None:
            components: list[str] = []
        elif isinstance(raw_components, list):
            components = [str(value) for value in raw_components]
        else:
            raise ValueError("components must be a list")
        result = self.research.provision_sources(components)
        with self._lock:
            self._add_event(
                "memory", "Research annotation sources provisioned"
            )
            self._status_sequence += 1
        return result

    def import_research_annotations(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            if self.engine_phase not in {"ready", "fault"}:
                raise RuntimeError(
                    "stop Monitor or Live before importing research annotations"
                )
        raw_components = payload.get("components")
        if raw_components is None:
            components: list[str] = []
        elif isinstance(raw_components, list):
            components = [str(value) for value in raw_components]
        else:
            raise ValueError("components must be a list")
        result = self.research.import_annotations(components)
        imported = [
            item["component_id"]
            for item in result["results"]
            if item["state"] == "imported"
        ]
        action_required = [
            item["component_id"]
            for item in result["results"]
            if item["state"] != "imported"
        ]
        with self._lock:
            if imported:
                message = (
                    "Imported research annotations: " + ", ".join(imported)
                )
            else:
                message = "No research annotations were imported"
            if action_required:
                message += "; action required for " + ", ".join(action_required)
            self._add_event("memory", message)
            self._status_sequence += 1
        return result

    def _training_snapshot_unlocked(self) -> dict[str, Any]:
        recorder = self._training_recorder
        current = recorder.status() if recorder is not None else None
        try:
            disk = shutil.disk_usage(self.training_root.parent)
            disk_free_bytes = disk.free
        except OSError:
            disk_free_bytes = None
        history = dict(self._training_history)
        historical_bytes = int(history.get("bytes", 0))
        current_bytes = int(current.get("bytes_written", 0)) if current else 0
        return {
            "enabled": self.training_capture_enabled,
            "path": str(self.training_root),
            "max_bytes": round(self.training_max_gb * 1024**3),
            "disk_free_bytes": disk_free_bytes,
            "current": current,
            "current_linked_feedback": self._training_linked_feedback,
            "current_annotations": self._training_annotations,
            "history": history,
            "total_bytes": historical_bytes + current_bytes,
            "last_export": self._last_training_export,
            "capture_policy": (
                "Lossless line-in PCM is recorded in Monitor and Live modes. "
                "Demo mode is never recorded."
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

    def spotify_console(
        self,
        query: str = "",
        playlist_id: str = "",
    ) -> dict[str, Any]:
        if not self.spotify_client_id or not DEFAULT_SPOTIFY_TOKEN.exists():
            return {
                "connected": False,
                "control_authorized": False,
                "library_authorized": False,
                "granted_scopes": [],
                "profile": None,
                "playback": None,
                "devices": [],
                "playlists": [],
                "selected_playlist": None,
                "playlist_tracks": [],
                "playlist_error": None,
                "results": [],
                "query": query,
                "playlist_id": playlist_id,
                "message": (
                    "Connect a private Spotify developer app in System to "
                    "activate this console."
                ),
            }
        cache_key = f"{query[:200]}|{playlist_id[:128]}"
        if time.time() < self._spotify_rate_limited_until and cache_key in self._spotify_console_cache:
            cached = dict(self._spotify_console_cache[cache_key])
            cached["stale"] = True
            cached["message"] = "Spotify is rate limited; showing the last known player state."
            return cached
        try:
            oauth = SpotifyOAuthPKCE(
                client_id=self.spotify_client_id,
                cache=SpotifyTokenCache(DEFAULT_SPOTIFY_TOKEN),
            )
            client = SpotifyWebAPI(oauth.valid_token)
            console = client.console(
                query=query[:200],
                playlist_id=playlist_id[:128],
            )
            self._remember_spotify_payload(client.last_playback_payload)
            with self._lock:
                self._spotify_error = None
                self._status_sequence += 1
                playback = console.get("playback") or {}
                active_device = playback.get("device") or {}
                console["diagnostics"] = {
                    "route_policy": "follow_spotify_active_device",
                    "active_device": active_device.get("name"),
                    "available_device_count": len(console.get("devices", [])),
                    "available_device_names": [
                        device.get("name")
                        for device in console.get("devices", [])
                        if device.get("name")
                    ],
                    "last_command": self._spotify_last_command,
                    "api_note": (
                        "Spotify does not return every device model through "
                        "its Web API. Choose Chromecast in Spotify itself; "
                        "Lumen will follow that active route."
                    ),
                }
            self._spotify_console_cache[cache_key] = dict(console)
            return console
        except Exception as error:
            if "rate limited" in str(error).lower() and cache_key in self._spotify_console_cache:
                self._spotify_rate_limited_until = time.time() + 15.0
                cached = dict(self._spotify_console_cache[cache_key])
                cached["stale"] = True
                cached["message"] = "Spotify API rate limited; showing cached player state for 15 seconds."
                return cached
            with self._lock:
                self._spotify_error = str(error)
                self._status_sequence += 1
            raise RuntimeError(str(error)) from error

    def spotify_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.spotify_client_id or not DEFAULT_SPOTIFY_TOKEN.exists():
            raise RuntimeError("connect Spotify in System first")
        action = str(payload.get("action", "")).strip().lower()
        if not action:
            raise ValueError("Spotify action is required")
        oauth = SpotifyOAuthPKCE(
            client_id=self.spotify_client_id,
            cache=SpotifyTokenCache(DEFAULT_SPOTIFY_TOKEN),
        )
        try:
            SpotifyWebAPI(oauth.valid_token).command(action, payload)
        except Exception as error:
            with self._lock:
                self._spotify_error = str(error)
                self._spotify_last_command = {
                    "action": action,
                    "ok": False,
                    "message": str(error),
                    "at_unix_ms": round(time.time() * 1000),
                }
                self._status_sequence += 1
            raise RuntimeError(str(error)) from error
        with self._lock:
            self._spotify_error = None
            self._spotify_last_command = {
                "action": action,
                "ok": True,
                "message": "Spotify accepted the command.",
                "at_unix_ms": round(time.time() * 1000),
            }
            self._last_media_poll = 0.0
            self._add_event("media", f"Spotify control: {action}")
            self._status_sequence += 1
        return {"accepted": True, "action": action}

    def _remember_spotify_payload(
        self,
        payload: dict[str, Any] | None,
    ) -> None:
        self._remember_media_identity(
            media_identity_from_spotify(payload or {})
        )

    def _remember_media_identity(
        self, media: MediaIdentity | None
    ) -> None:
        # Spotify console and polling threads can report the same transition.
        # Serialize identity persistence without holding the audio publication
        # lock during SQLite work.
        with self._media_lock:
            if media is None:
                with self._lock:
                    self.media = None
                return
            key = f"{media.provider}:{media.provider_item_id}"
            with self._lock:
                count_play = key != self._last_media_key
            song_id = self.memory.remember_media(
                media, count_play=count_play
            )
            with self._lock:
                self.media = media
                self.song_id = song_id
                runtime = self._runtime
                section = self.observation.section
                if count_play:
                    self._analysis_generation += 1
                    self._last_media_key = key
                    self._add_event(
                        "media", f"Now playing {media.display_name}"
                    )
            if runtime is not None:
                runtime.set_media_context(
                    song_id,
                    section,
                    media.artists[0] if media.artists else None,
                )

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
                        "pan_left_dmx",
                        "pan_right_dmx",
                        "tilt_high_dmx",
                        "tilt_low_dmx",
                        "home_pan_dmx",
                        "home_tilt_dmx",
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
        # Labels carry direction ("dimmer", "faster", and so on); value is
        # only their magnitude. A neutral value must remain neutral.
        value = clamp(float(payload.get("value", 1.0)), -1.0, 1.0)
        note = str(payload.get("note", "")).strip() or None
        scope = str(payload.get("scope", "overall")).strip().lower()
        if scope not in {"overall", "fixture", "group"}:
            raise ValueError("feedback scope must be overall, group, or fixture")
        raw_fixture_id = payload.get("fixture_id")
        fixture_id = str(raw_fixture_id).strip() if raw_fixture_id is not None else None
        group_id = str(payload.get("group_id", "")).strip() or None
        if scope == "fixture" and not fixture_id:
            raise ValueError("fixture feedback requires a fixture_id")
        if scope == "group" and group_id != "movers":
            raise ValueError("unknown feedback group")
        if fixture_id and not any(
            fixture.fixture_id == fixture_id
            for fixture in (*self.rig.fixtures, *self.rig.auxiliary_fixtures)
        ):
            raise ValueError("feedback fixture_id is not in the active rig")
        with self._feedback_lock:
            with self._lock:
                song_id = self.song_id
                media_snapshot = self.media
            if song_id is None:
                media = media_snapshot or MediaIdentity(
                    provider="line-in",
                    provider_item_id=f"unidentified:{datetime.now():%Y-%m-%d}",
                    title="Unidentified line-in session",
                    artists=(),
                    is_playing=self.engine_mode != "standby",
                )
                remembered_song_id = self.memory.remember_media(media)
                with self._lock:
                    if self.song_id is None:
                        self.song_id = remembered_song_id
                    song_id = self.song_id
            assert song_id is not None
            with self._lock:
                context_frame = self.frame
                context_observation = self.observation
                capture_audio_frame = self._training_audio_frame
                capture_session_id = (
                    self._session_id
                    if capture_audio_frame is not None
                    else None
                )
                position_ms = self._media_position_ms()
                runtime = self._runtime
            feedback_id = self.memory.add_feedback(
                Feedback(
                    song_id=song_id,
                    position_ms=position_ms,
                    label=label[:64],
                    value=clamp(value, -1.0, 1.0),
                    note=note,
                    scope=scope,
                    fixture_id=group_id if scope == "group" else fixture_id,
                    gesture=(context_frame.decision.gesture.value if context_frame else None),
                    section=context_observation.section,
                    energy=(context_frame.decision.expression.energy if context_frame else None),
                    motion=(context_frame.decision.expression.motion if context_frame else None),
                    tension=(context_frame.decision.expression.tension if context_frame else None),
                    confidence=(context_frame.decision.confidence if context_frame else None),
                    bpm=context_observation.bpm,
                    routine=(context_frame.decision.routine if context_frame else None),
                    capture_session_id=capture_session_id,
                    audio_frame_index=capture_audio_frame,
                )
            )
            # Keep a compact semantic routine alongside the raw feedback. It
            # is deliberately made from moments and decisions, not DMX bytes,
            # so it can be resolved against a changed rig later.
            song_feedback = self.memory.list_feedback(song_id)
            recent_cutoff = int(time.time() * 1000) - 5_000
            feedback_fixture = group_id if scope == "group" else fixture_id
            occurrences = sum(
                1
                for row in song_feedback
                if row.get("label") == label
                and row.get("scope") == scope
                and row.get("fixture_id") == feedback_fixture
                and int(row.get("created_unix_ms") or 0) >= recent_cutoff
            )
            learned_sequence: dict[str, Any] | None = None
            if runtime is not None and abs(value) > 0.0:
                preferred = self._feedback_routine_effect(label, None)
                learned_sequence = runtime.learn_choreography_feedback(
                    label=label,
                    value=clamp(value, -1.0, 1.0),
                    urgency=clamp(0.45 + 0.12 * (occurrences - 1), 0.0, 1.0),
                    # Each database event contributes once. Repeated presses
                    # raise urgency without quadratically counting 1+2+3...
                    occurrences=1,
                    scope=scope,
                    fixture_id=feedback_fixture,
                    preferred_routine=(
                        max(preferred, key=preferred.get)
                        if preferred
                        else None
                    ),
                    created_unix_ms=int(time.time() * 1000),
                    event_id=f"feedback:{feedback_id}",
                )
                if learned_sequence is not None:
                    self._save_choreography_model()
                    self._remember_sequence_learning(
                        learned_sequence,
                        label=label,
                        scope=scope,
                        song_id=song_id,
                    )
            self._save_feedback_routine(song_id, song_feedback)
            self._rebuild_feedback_biases()
            with self._lock:
                current_runtime = self._runtime
                biases = deepcopy(self._feedback_biases)
                if capture_audio_frame is not None:
                    self._training_linked_feedback += 1
                self._add_event(
                    "feedback",
                    f"Recorded feedback: {label.replace('_', ' ')}",
                )
                self._status_sequence += 1
            if current_runtime is not None:
                current_runtime.replace_feedback(biases)
        return {
            "feedback_id": feedback_id,
            "song_id": song_id,
            "feedback_occurrences": max(1, occurrences),
            "sequence_learning": learned_sequence,
        }

    def add_training_annotation(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip().lower()
        allowed = {
            "musical_context": {
                "intro", "verse", "pre_chorus", "chorus", "build",
                "release", "breakdown", "bridge", "solo", "outro",
                "vocal_focus", "instrumental", "transition",
            },
            "preferred_action": {
                "keep_current", "breathe", "fan_sweep", "figure_eight",
                "opposing_chase", "beat_nod", "counter_rotate",
                "hold_position", "blackout_accent", "palette_change",
                "more_contrast", "less_contrast",
            },
        }
        label = str(payload.get("label", "")).strip().lower()
        if kind not in allowed or label not in allowed[kind]:
            raise ValueError("unknown training annotation")
        scope = str(payload.get("scope", "overall")).strip().lower()
        if scope not in {"overall", "group", "fixture"}:
            raise ValueError("annotation scope must be overall, group, or fixture")
        fixture_id = str(payload.get("fixture_id", "")).strip() or None
        group_id = str(payload.get("group_id", "")).strip() or None
        if scope == "group" and group_id != "movers":
            raise ValueError("unknown annotation group")
        if scope == "fixture" and (
            fixture_id is None
            or not any(
                fixture.fixture_id == fixture_id
                for fixture in (
                    *self.rig.fixtures,
                    *self.rig.auxiliary_fixtures,
                )
            )
        ):
            raise ValueError("annotation fixture is not in the active rig")
        note = str(payload.get("note", "")).strip() or None
        intensity = clamp(float(payload.get("intensity", 1.0)), 0.1, 1.0)
        with self._feedback_lock:
            with self._lock:
                song_id = self.song_id
                media_snapshot = self.media
            if song_id is None:
                media = media_snapshot or MediaIdentity(
                    provider="line-in",
                    provider_item_id=f"unidentified:{datetime.now():%Y-%m-%d}",
                    title="Unidentified line-in session",
                    artists=(),
                    is_playing=self.engine_mode != "standby",
                )
                remembered_song_id = self.memory.remember_media(media)
                with self._lock:
                    if self.song_id is None:
                        self.song_id = remembered_song_id
                    song_id = self.song_id
            assert song_id is not None
            with self._lock:
                frame_snapshot = self.frame
                observation_snapshot = self.observation
                controls_snapshot = asdict(self.controls)
                capture_audio_frame = self._training_audio_frame
                capture_session_id = (
                    self._session_id
                    if capture_audio_frame is not None
                    else None
                )
                position_ms = self._media_position_ms()
                runtime = self._runtime
            decision: dict[str, Any] | None = None
            if frame_snapshot is not None:
                decision = asdict(frame_snapshot.decision)
                decision["gesture"] = frame_snapshot.decision.gesture.value
            annotation_id = self.memory.add_training_annotation(
                song_id=song_id,
                position_ms=position_ms,
                kind=kind,
                label=label,
                scope=scope,
                fixture_id=group_id if scope == "group" else fixture_id,
                intensity=intensity,
                note=note,
                capture_session_id=capture_session_id,
                audio_frame_index=capture_audio_frame,
                context={
                    "observation": asdict(observation_snapshot),
                    "decision": decision,
                    "controls": controls_snapshot,
                },
            )
            sequence_learning: dict[str, Any] | None = None
            if kind == "preferred_action" and runtime is not None:
                sequence_learning = (
                    runtime.learn_choreography_feedback(
                        label="more_like_this",
                        value=intensity,
                        urgency=intensity,
                        occurrences=1,
                        scope=scope,
                        fixture_id=(
                            group_id if scope == "group" else fixture_id
                        ),
                        preferred_routine=(
                            None if label == "keep_current" else label
                        ),
                        created_unix_ms=int(time.time() * 1000),
                        event_id=f"annotation:{annotation_id}",
                    )
                )
                if sequence_learning is not None:
                    self._save_choreography_model()
                    self._remember_sequence_learning(
                        sequence_learning,
                        label=label,
                        scope=scope,
                        song_id=song_id,
                    )
            with self._lock:
                if capture_audio_frame is not None:
                    self._training_annotations += 1
                self._add_event(
                    "feedback",
                    f"Training label: {label.replace('_', ' ')}",
                )
                self._status_sequence += 1
        return {
            "annotation_id": annotation_id,
            "song_id": song_id,
            "kind": kind,
            "label": label,
            "linked_to_audio": capture_audio_frame is not None,
            "sequence_learning": sequence_learning,
        }

    def delete_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback_id = int(payload.get("feedback_id", 0))
        if feedback_id < 1:
            raise ValueError("feedback_id is required")
        with self._feedback_lock:
            existing = next(
                (
                    row
                    for row in self.memory.all_feedback()
                    if int(row.get("id") or 0) == feedback_id
                ),
                None,
            )
            deleted = self.memory.delete_feedback(feedback_id)
            if not deleted:
                raise ValueError("feedback entry was not found")
            sequence_update_removed = self._choreography_model.forget(
                f"feedback:{feedback_id}"
            )
            if sequence_update_removed:
                self._save_choreography_model()
            if existing is not None and existing.get("song_id") is not None:
                song_id = int(existing["song_id"])
                self._save_feedback_routine(
                    song_id, self.memory.list_feedback(song_id)
                )
            self._rebuild_feedback_biases()
            with self._lock:
                runtime = self._runtime
                biases = deepcopy(self._feedback_biases)
                self._add_event(
                    "feedback", f"Removed feedback #{feedback_id}"
                )
                self._status_sequence += 1
            if runtime is not None:
                runtime.replace_feedback(biases)
        return {
            "deleted": True,
            "feedback_id": feedback_id,
            "sequence_update_removed": sequence_update_removed,
        }

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
            self._remember_media_identity(media)
            with self._lock:
                self._spotify_error = None
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
                    "version": "0.6.0",
                    "role": "Spatial music-lighting control",
                },
                "rig": self._rig_payload,
                "profiles": [
                    profile_summary(profile)
                    for profile in PARTY_PARROT_PROFILES.values()
                ],
                "status": self._snapshot_unlocked(),
                "memory": self.memory.summary(limit=30),
                "research": self.research_status(),
                "system": self.scan_system(),
                "settings": self.operator_settings(),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        observation = asdict(self.observation)
        running = self._thread is not None and self._thread.is_alive()
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
                "running": running,
                "uptime_s": (
                    None
                    if self.started_at is None
                    else max(0.0, time.monotonic() - self.started_at)
                ),
                "audio_device": self.audio_device,
                "error": self.last_error,
            },
            "controls": asdict(self.controls),
            "rehearsal": {
                **asdict(self.rehearsal),
                "routines": list(REHEARSAL_ROUTINES),
                "motion_editor": self._motion_editor_snapshot(),
            },
            "audio": self._audio_snapshot_unlocked(running),
            "observation": observation,
            "analysis_history": list(self._analysis_history),
            "structure_model": {
                "state": (
                    "active"
                    if self._student_model is not None
                    else "error"
                    if self._student_model_error
                    else "awaiting_training"
                ),
                "model_path": str(self._student_model_path),
                "error": self._student_model_error,
                "prediction": self._student_prediction,
            },
            "training": self._training_snapshot_unlocked(),
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
        parsed = urlparse(self.path)
        path = parsed.path
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
        if path == "/api/research":
            self._json(
                HTTPStatus.OK,
                self.server.application.research_status(),
            )
            return
        if path == "/api/spotify":
            query_values = parse_qs(parsed.query)
            query = query_values.get("q", [""])[0]
            playlist_id = query_values.get("playlist_id", [""])[0]
            try:
                self._json(
                    HTTPStatus.OK,
                    self.server.application.spotify_console(
                        query,
                        playlist_id,
                    ),
                )
            except RuntimeError as error:
                self._json(HTTPStatus.CONFLICT, {"error": str(error)})
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
            elif path == "/api/rehearsal":
                result = app.patch_rehearsal(payload)
            elif path == "/api/motion-routine":
                result = app.patch_motion_routine(payload)
            elif path == "/api/preset":
                result = app.apply_preset(str(payload.get("preset", "")))
            elif path == "/api/feedback":
                result = app.add_feedback(payload)
            elif path == "/api/feedback/delete":
                result = app.delete_feedback(payload)
            elif path == "/api/training/annotation":
                result = app.add_training_annotation(payload)
            elif path == "/api/calibration":
                result = app.calibration_control(payload)
            elif path == "/api/gesture/fresh":
                result = app.request_fresh_gesture()
            elif path == "/api/settings":
                result = app.patch_settings(payload)
            elif path == "/api/training/export":
                result = app.export_training_data()
            elif path == "/api/research/provision-sources":
                result = app.provision_research_sources(payload)
            elif path == "/api/research/import-annotations":
                result = app.import_research_annotations(payload)
            elif path == "/api/research/run-job":
                result = app.start_research_worker(payload)
            elif path == "/api/research/analyze":
                result = app.analyze_training_data()
            elif path == "/api/research/cancel":
                result = app.cancel_research_worker()
            elif path == "/api/research/train-student":
                result = app.train_structure_student(payload)
            elif path == "/api/spotify/connect":
                result = app.connect_spotify(payload)
            elif path == "/api/spotify/control":
                result = app.spotify_command(payload)
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
        bar_phase=(index % 16) / 16.0,
        beat_pulse=1.0 if beat else 0.18,
        beat_confidence=0.88,
        bpm=125.0,
        section=section,
        section_confidence=0.86,
        novelty=0.92 if section == "drop" and beat else 0.24 + energy * 0.34,
    )


def _rehearsal_observation(
    timestamp_s: float,
    bpm: float,
    intensity: float,
) -> MusicalObservation:
    """Build an exact, low-jitter musical clock for motion auditioning."""
    beat_position = timestamp_s * bpm / 60.0
    beat_phase = beat_position % 1.0
    bar_phase = (beat_position % 4.0) / 4.0
    distance_to_beat = min(beat_phase, 1.0 - beat_phase)
    beat_pulse = clamp(1.0 - distance_to_beat / 0.16, 0.0, 1.0)
    energy = clamp(intensity, 0.0, 1.0)
    return MusicalObservation(
        timestamp_s=timestamp_s,
        loudness=max(0.12, energy),
        onset_strength=0.18 + 0.82 * beat_pulse,
        low_energy=clamp(0.34 + energy * 0.58, 0.0, 1.0),
        mid_energy=clamp(0.28 + energy * 0.48, 0.0, 1.0),
        high_energy=clamp(0.16 + energy * 0.38, 0.0, 1.0),
        beat_phase=beat_phase,
        bar_phase=bar_phase,
        beat_pulse=beat_pulse,
        beat_confidence=1.0,
        bpm=bpm,
        section="groove",
        section_confidence=1.0,
        novelty=0.72 if beat_pulse > 0.7 else 0.12,
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
