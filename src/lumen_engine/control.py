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
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse
import webbrowser

from lumen_engine.audio import (
    AudioCaptureConfig,
    AudioInputMetrics,
    AlsaLineIn,
    ContinuouslyDrainedAudio,
    RealtimeAudioAnalyzer,
)
from lumen_engine.config import RigConfig, load_rig, rig_from_dict
from lumen_engine.choreography import (
    CHOREOGRAPHY_LANES,
    ChoreographySequence,
    ChoreographyStep,
    SequencePreferenceModel,
    choreography_lanes_for_scope,
)
from lumen_engine.dmx import DMXFrame, DMXOutput, VirtualDMXOutput
from lumen_engine.expression import ExpressionEngine, ExpressionPolicy
from lumen_engine.media import (
    SpotifyNowPlayingProvider,
    SpotifyOAuthPKCE,
    SpotifyTokenCache,
    SpotifyWebAPI,
    media_identity_from_spotify,
    spotify_playback_summary,
)
from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.link import LumenLinkCoordinator
from lumen_engine.models import (
    Feedback,
    MediaIdentity,
    MusicalObservation,
    PerformanceDecision,
    Vec3,
    clamp,
)
from lumen_engine.motion import (
    DEFAULT_CENTER_MOTION_TUNINGS,
    DEFAULT_MOTION_TUNINGS,
    CenterMotionTuning,
    GroupMotionTunings,
    MotionTuning,
    canonical_motion_scope,
    merged_group_motion_tunings,
    preview_paths,
    required_axis_speeds,
)
from lumen_engine.offline import (
    EDMFORMER_JOB,
    OfflineResearchWorker,
    ResearchJobCoordinator,
    SONGFORMER_JOB,
    STUDENT_ACTIVATION_GATE_VERSION,
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
from lumen_engine.structure import (
    CANONICAL_TECHNO_SECTIONS,
    ContentRole,
    TransitionEvent,
)
from lumen_engine.student import (
    StableStructureDecoder,
    StreamingStructureStudent,
    semantic_frame_features,
)
from lumen_engine.training import (
    TrainingCaptureConfig,
    TrainingDataRecorder,
    export_research_session_index,
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
SPOTIFY_MEDIA_POLL_INTERVAL_S = 2.0

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
            self.scope = canonical_motion_scope(values["scope"])
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


@dataclass(frozen=True, slots=True)
class _AnalyzedControlFrame:
    """One immutable musical result handed to the fixed-rate show clock."""

    observation: MusicalObservation
    raw_observation: MusicalObservation
    audio_metrics: AudioInputMetrics
    audio_bytes: int
    training_audio_frame: int | None
    runtime_context: dict[str, Any]
    analysis_started_perf_s: float
    analysis_stages_ms: dict[str, float]
    timeline_discontinuity: bool = False
    audio_discontinuity: bool = False


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
        # The browser dashboard can be expensive to serialize and several
        # phones may poll it concurrently. Live audio publication therefore
        # never waits for the general application/interface lock. This small
        # lock protects only atomically replaced live diagnostics and frame
        # references; interface rendering reads their latest published value.
        self._live_state_lock = threading.Lock()
        # Persistence/model updates may be comparatively slow. Serialize them
        # independently so feedback from several clients cannot block the
        # audio/DMX publication lock or corrupt reversible model events.
        self._feedback_lock = threading.Lock()
        self._media_lock = threading.Lock()
        self._configuration_lock = threading.Lock()
        # Export preparation reconstructs and validates captured audio. Keep it
        # single-flight even though the HTTP server handles requests in
        # parallel, so a double-click cannot build two manifests at once.
        self._training_export_lock = threading.Lock()
        self._teaching_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._output: GatedOutput | None = None
        self._runtime: PerformanceRuntime | None = None
        self.controls = OperatorControls()
        self.rehearsal = RehearsalControls()
        group_motion_tunings = self._load_motion_tunings()
        self.motion_tunings = group_motion_tunings.movers
        self.center_motion_tunings = group_motion_tunings.center
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
        # Several phones may submit feedback in the same musical moment.
        # Model state remains updated synchronously in memory, but the large
        # reversible JSON snapshot is coalesced on a persistence worker so
        # four acknowledgements do not perform four competing HDD writes.
        self._model_save_condition = threading.Condition()
        self._model_save_requested = 0
        self._model_save_completed = 0
        self._model_save_stop = False
        self._model_save_error: str | None = None
        self._model_save_thread = threading.Thread(
            target=self._choreography_model_save_worker,
            name="lumen-choreography-persistence",
            daemon=True,
        )
        self._model_save_thread.start()
        self.research = ResearchManager(
            self.training_root / "research",
            store=self.memory,
        )
        # Lumen Link owns only offline job leases and immutable transfers.
        # It has no reference to the runtime, audio capture, or DMX output.
        self.lumen_link = LumenLinkCoordinator(
            self.memory,
            research_root=self.training_root / "research",
            state_root=state_root / "lumen-link",
            config_path=state_root / "lumen-link" / "config.json",
            can_import=lambda: (
                self.engine_mode == "standby"
                and not (self._thread and self._thread.is_alive())
            ),
        )
        # The exact readiness audit verifies checksums and parses every trusted
        # teacher-example file.  On the installed library that is roughly a
        # gigabyte of reads, so it must never run synchronously in bootstrap or
        # an HTTP status poll.  Exact Analyze/Train operations still perform
        # the audit; the console reads this durable last-known result while a
        # single background refresh catches up after an older installation.
        self._research_readiness_cache_path = (
            self.training_root
            / "research"
            / "cache"
            / "operator-readiness.json"
        )
        self._research_readiness_lock = threading.Lock()
        self._research_readiness_cache = (
            self._load_research_readiness_cache()
        )
        self._research_readiness_thread: threading.Thread | None = None
        self._research_readiness_error: str | None = None
        self._student_model_path = (
            self.training_root
            / "research"
            / "models"
            / "lumen-structure-student.npz"
        )
        self._student_model: StreamingStructureStudent | None = None
        self._student_model_signature: tuple[int, int] | None = None
        self._student_decoder = StableStructureDecoder()
        self._student_model_error: str | None = None
        self._student_model_notice: str | None = None
        self._student_model_state = "awaiting_training"
        self._student_model_gate_reasons: list[str] = []
        self._student_prediction: dict[str, Any] | None = None
        self._effective_structure: dict[str, Any] = {
            "source": "live_analyzer",
            "section": "silence",
            "confidence": 0.0,
            "beat_timing_authority": "audio_sample_clock",
        }
        self._physical_quiet_since_s: float | None = None
        self._physical_signal_since_s: float | None = None
        self._physical_silence_active = False
        self._last_raw_observation = _silence_observation()
        self._cached_structure_prediction: dict[str, Any] | None = None
        self._cached_structure_key: str | None = None
        self._cached_structure_checked_at = 0.0
        self._recalled_choreography_checked_at = 0.0
        self._recalled_choreography_ids: tuple[str, ...] = ()
        self._recalled_choreography_runtime: PerformanceRuntime | None = None
        self._prepared_recalled_choreography: tuple[
            ChoreographySequence, ...
        ] = ()
        self._prepared_recalled_ids: tuple[str, ...] = ()
        self._memory_context_thread: threading.Thread | None = None
        self._memory_context_last_poll = 0.0
        self._memory_context_last_duration_ms: float | None = None
        self._memory_context_error: str | None = None
        self._runtime_choreography_snapshot: dict[str, Any] | None = None
        self._teaching_snapshot_cache: dict[str, Any] | None = None
        self._teaching_snapshot_checked_at = 0.0
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
        self._training_disk_free_bytes = self._read_training_disk_free()
        self._last_training_export: str | None = None
        self._training_prepare_thread: threading.Thread | None = None
        self._training_prepare_pending = False
        self._training_prepare_pending_session: str | None = None
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
        self._trace_queue_drops = 0
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
        self._spotify_console_cached_at: dict[str, float] = {}
        self._spotify_console_lock = threading.Lock()
        self._spotify_rate_limited_until = 0.0
        self._feedback_biases: dict[str, dict[str, Any]] = {}
        self._calibration_overrides: dict[str, dict[str, float]] = {}
        self._audio_metrics = AudioInputMetrics.silence()
        self._audio_packets = 0
        self._audio_frames = 0
        self._audio_bytes = 0
        self._audio_capture_started_at: float | None = None
        self._audio_last_packet_at: float | None = None
        self._audio_packet_times: deque[float] = deque(maxlen=96)
        self._audio_queue_depth = 0
        self._audio_queue_max_depth = 0
        self._audio_queue_delay_ms = 0.0
        self._audio_capture_diagnostics: dict[str, Any] = {}
        self._active_audio_capture: ContinuouslyDrainedAudio | None = None
        self._control_queue_depth = 0
        self._control_queue_max_depth = 0
        self._control_queue_drops = 0
        self._control_ticks = 0
        self._control_interpolated_ticks = 0
        self._control_maximum_late_ms = 0.0
        self._tempo_diagnostics: dict[str, Any] = {}
        self._live_pipeline_timing: dict[str, Any] = {
            "packets": 0,
            "last_total_ms": 0.0,
            "maximum_total_ms": 0.0,
            "deadline_misses": 0,
            "stages_ms": {},
            "maximum_stages_ms": {},
        }
        self._analysis_history: deque[dict[str, Any]] = deque(maxlen=240)
        self._last_analysis_history_at: float | None = None
        self._analysis_generation = 0
        self._add_event("system", f"Loaded {self.rig.name}")
        self._rebuild_feedback_biases()
        self._feedback_refresh_condition = threading.Condition()
        self._feedback_refresh_requested = 0
        self._feedback_refresh_completed = 0
        self._feedback_refresh_lanes: set[str] = set()
        self._feedback_refresh_stop = False
        self._feedback_refresh_error: str | None = None
        self._feedback_refresh_thread = threading.Thread(
            target=self._feedback_bias_refresh_worker,
            name="lumen-feedback-refresh",
            daemon=True,
        )
        self._feedback_refresh_thread.start()
        self.solve_target(self.selected_target)
        # Start remote polling only after all engine state exists and stale
        # local worker leases have been recovered. A restarted link can then
        # idempotently resume the same immutable remote job.
        self.lumen_link.start()

    def _load_student_model(self) -> None:
        self._student_model = None
        self._student_model_error = None
        self._student_model_notice = None
        self._student_model_state = "awaiting_training"
        self._student_model_gate_reasons = []
        self._reset_student_stream()
        if not self._student_model_path.is_file():
            self._student_model_signature = None
            return
        try:
            evaluation_path = self._student_model_path.with_name(
                self._student_model_path.stem + ".evaluation.json"
            )
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            if (
                evaluation.get("teacher_normalization_version")
                != TEACHER_NORMALIZATION_VERSION
            ):
                self._student_model_notice = (
                    "The previous active student is disabled because it "
                    "uses an obsolete teacher normalization. Current "
                    "candidates are evaluated separately."
                )
                self._student_model_state = "obsolete"
            elif (
                evaluation.get("edmformer_preprocessing_version")
                != EDMFORMER_PREPROCESSING_VERSION
            ):
                self._student_model_notice = (
                    "The previous active student is disabled because it "
                    "predates the current full-song EDMFormer pipeline. "
                    "Current candidates are evaluated separately."
                )
                self._student_model_state = "obsolete"
            elif (
                evaluation.get("activation_gate_version")
                != STUDENT_ACTIVATION_GATE_VERSION
            ):
                self._student_model_notice = (
                    "The previous active student is disabled because it "
                    "predates the current unseen-song and tolerant-boundary "
                    "qualification gate. Current candidates are evaluated "
                    "separately."
                )
                self._student_model_state = "obsolete"
            elif evaluation.get("activated") is not True:
                axis_reasons = evaluation.get("axis_gate_reasons") or {}
                self._student_model_gate_reasons = [
                    str(reason)
                    for reasons in axis_reasons.values()
                    if isinstance(reasons, list)
                    for reason in reasons
                ]
                self._student_model_state = "quarantined"
            else:
                self._student_model = StreamingStructureStudent.load(
                    self._student_model_path
                )
                if not self._student_model.approved_axes:
                    raise ValueError(
                        "activated student model contains no approved axes"
                    )
                self._student_model_state = "active"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._student_model_error = str(error)
            self._student_model_state = "error"
        try:
            stat = self._student_model_path.stat()
            self._student_model_signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self._student_model_signature = None

    def _reset_student_stream(
        self, *, reset_physical_silence: bool = True
    ) -> None:
        """Begin a clean causal structure stream at a listening boundary."""
        if self._student_model is not None:
            self._student_model.reset()
        self._student_decoder.reset()
        self._student_prediction = None
        if reset_physical_silence:
            self._physical_quiet_since_s = None
            self._physical_signal_since_s = None
            self._physical_silence_active = False
        observation = getattr(self, "observation", _silence_observation())
        self._effective_structure = {
            "source": "live_analyzer",
            "section": observation.section,
            "confidence": observation.section_confidence,
            "beat_timing_authority": "audio_sample_clock",
        }

    def _student_artifact_changed(self) -> bool:
        try:
            stat = self._student_model_path.stat()
        except OSError:
            return self._student_model_signature is not None
        return (stat.st_mtime_ns, stat.st_size) != self._student_model_signature

    def _apply_student_structure(
        self,
        observation: MusicalObservation,
        metrics: AudioInputMetrics,
    ) -> MusicalObservation:
        model = self._student_model
        if model is None:
            self._student_prediction = None
            self._effective_structure = {
                "source": "live_analyzer",
                "section": observation.section,
                "confidence": observation.section_confidence,
                "beat_timing_authority": "audio_sample_clock",
            }
            return observation
        prediction = model.predict(
            semantic_frame_features(
                self._semantic_audio_payload(observation, metrics)
            ),
            timestamp_s=observation.timestamp_s,
        )
        approved_axes = set(model.approved_axes)
        decoder_prediction = prediction
        if "boundary" not in approved_axes:
            decoder_prediction = replace(
                prediction, boundary_probability=0.0
            )
        stable = self._student_decoder.update(
            decoder_prediction, observation.timestamp_s
        )
        selected_section = observation.section
        selected_axis = "live_analyzer"
        selected_confidence = observation.section_confidence
        energy_confidence = float(stable["confidence"]["energy"])
        functional_confidence = float(stable["confidence"]["functional"])
        content_confidence = float(stable["confidence"]["content"])
        physical_silence_supported = bool(
            observation.loudness < 0.02
            and metrics.dbfs <= -50.0
            and metrics.rms <= 0.006
            and metrics.peak <= 0.02
        )
        energy_label_valid = bool(
            stable["energy"] not in {None, "unknown"}
            and (
                stable["energy"] != "silence"
                or physical_silence_supported
            )
        )
        accepted_axes = {
            "energy": bool(
                "energy" in approved_axes
                and energy_label_valid
                and energy_confidence >= 0.52
            ),
            "functional": bool(
                "functional" in approved_axes
                and
                stable["functional"] not in {None, "unknown"}
                and functional_confidence >= 0.60
            ),
            "content": bool(
                "content" in approved_axes
                and
                stable["content"] not in {None, "unknown"}
                and content_confidence >= 0.55
            ),
            "boundary": bool(
                "boundary" in approved_axes
                and prediction.boundary_probability >= 0.55
            ),
        }
        if (
            "energy" in approved_axes
            and energy_label_valid
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
            "approved_axes": sorted(approved_axes),
            "accepted_axes": accepted_axes,
            "selected_axis": selected_axis,
            "selected_section": selected_section,
            "silence_gate": {
                "physical_silence_supported": physical_silence_supported,
                "student_silence_rejected": bool(
                    stable["energy"] == "silence"
                    and not physical_silence_supported
                ),
                "dbfs": metrics.dbfs,
                "rms": metrics.rms,
                "peak": metrics.peak,
            },
            "model_path": str(self._student_model_path),
            "training_examples": model.training_examples,
            "target_provenance": "lumen_streaming_structure_student",
        }
        if selected_axis == "live_analyzer":
            self._effective_structure = {
                "source": "live_analyzer",
                "section": observation.section,
                "confidence": observation.section_confidence,
                "student_energy_rejected": not accepted_axes["energy"],
                "beat_timing_authority": "audio_sample_clock",
            }
            return observation
        self._effective_structure = {
            "source": "streaming_student",
            "section": selected_section,
            "confidence": selected_confidence,
            "approved_axis": "energy",
            "beat_timing_authority": "audio_sample_clock",
        }
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
        # The 128-point waveform is a display aid already derived from the
        # lossless WAV. Persisting it at ten hertz inflated every semantic and
        # diagnostic row without adding a training feature.
        audio_payload.pop("waveform", None)
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

    @staticmethod
    def _structure_axis(
        label: str,
        confidence: float,
        source: str,
        reason: str,
        *,
        provenance: Any = None,
        timeline_id: Any = None,
        model_confidence: float | None = None,
        operator_trust: float = 0.0,
        recall_authority: str | None = None,
        cue_key: str | None = None,
    ) -> dict[str, Any]:
        normalized_confidence = clamp(float(confidence), 0.0, 1.0)
        normalized_trust = clamp(float(operator_trust), 0.0, 1.0)
        return {
            "label": label,
            "confidence": normalized_confidence,
            "model_confidence": (
                normalized_confidence
                if model_confidence is None
                else clamp(float(model_confidence), 0.0, 1.0)
            ),
            "operator_trust": normalized_trust,
            "decision_confidence": max(
                normalized_confidence, normalized_trust
            ),
            "recall_authority": recall_authority,
            "source": source,
            "accepted_reason": reason,
            "provenance": deepcopy(provenance),
            "timeline_id": timeline_id,
            "cue_key": cue_key,
        }

    def _resolve_structure(
        self,
        raw_observation: MusicalObservation,
        metrics: AudioInputMetrics,
    ) -> MusicalObservation:
        """Resolve each musical axis once, with physical audio authoritative.

        Cached teachers, the streaming student, and the live analyzer used to
        overwrite one another sequentially.  That made the final section and
        runtime context depend on call order.  This resolver selects every
        axis explicitly and publishes the same decision to Live, capture, and
        the performance trace.
        """

        timestamp = raw_observation.timestamp_s
        quiet_packet = bool(
            raw_observation.loudness <= 0.001
            and metrics.rms <= 0.006
            and metrics.peak <= 0.025
        )
        if quiet_packet:
            self._physical_signal_since_s = None
            if self._physical_quiet_since_s is None:
                self._physical_quiet_since_s = timestamp
            quiet_for = max(0.0, timestamp - self._physical_quiet_since_s)
            if quiet_for >= 0.55:
                self._physical_silence_active = True
        else:
            if self._physical_signal_since_s is None:
                self._physical_signal_since_s = timestamp
            signal_for = max(0.0, timestamp - self._physical_signal_since_s)
            if not self._physical_silence_active or signal_for >= 0.12:
                self._physical_quiet_since_s = None
                self._physical_silence_active = False
            quiet_for = 0.0

        cached = self._cached_structure_prediction or {}
        cached_axes = cached.get("axes") or {}
        student = self._student_prediction or {}
        student_accepted = student.get("accepted_axes") or {}
        student_confidence = student.get("confidence") or {}

        def cached_axis(axis: str, threshold: float) -> dict[str, Any] | None:
            value = cached_axes.get(axis) or {}
            label = str(value.get("label") or "unknown")
            confidence = float(value.get("confidence") or 0.0)
            operator_trust = float(value.get("operator_trust") or 0.0)
            if label == "unknown" or (
                confidence < threshold and operator_trust <= 0.0
            ):
                return None
            if (
                axis == "energy"
                and label == "silence"
                and not self._physical_silence_active
            ):
                return None
            provenance = value.get("provenance")
            operator_consensus = bool(
                value.get("recall_authority") == "operator_consensus"
                or (
                    isinstance(provenance, dict)
                    and "operator" in str(
                        provenance.get("source") or ""
                    ).casefold()
                )
            )
            return self._structure_axis(
                label,
                confidence,
                (
                    "operator_approved_timeline"
                    if operator_trust > 0.0
                    else "operator_consensus_timeline"
                    if operator_consensus
                    else "cached_offline_teacher"
                ),
                (
                    "operator-approved exact-recording timeline"
                    if operator_trust > 0.0
                    else "participant-consensus song correction"
                    if operator_consensus
                    else "confident cached teacher axis"
                ),
                provenance=provenance,
                timeline_id=value.get("timeline_id"),
                model_confidence=float(
                    value.get("model_confidence", confidence) or 0.0
                ),
                operator_trust=operator_trust,
                recall_authority=value.get("recall_authority"),
                cue_key=(
                    f"timeline:{value.get('timeline_id')}:"
                    f"{int(value.get('start_ms') or 0)}"
                    if value.get("timeline_id") else None
                ),
            )

        def student_axis(axis: str) -> dict[str, Any] | None:
            if not student_accepted.get(axis):
                return None
            label = str(student.get(axis) or "unknown")
            if label == "unknown":
                return None
            if (
                axis == "energy"
                and label == "silence"
                and not self._physical_silence_active
            ):
                return None
            return self._structure_axis(
                label,
                float(student_confidence.get(axis) or 0.0),
                "streaming_student",
                "approved confidence-gated student axis",
                provenance=student.get("model_path"),
            )

        if self._physical_silence_active:
            energy = self._structure_axis(
                "silence",
                1.0,
                "live_audio_silence",
                "physical line-input silence overrides learned structure",
            )
        else:
            energy = (
                cached_axis("energy", 0.62)
                or student_axis("energy")
                or self._structure_axis(
                    raw_observation.section,
                    raw_observation.section_confidence,
                    "live_analyzer",
                    "first-play live audio fallback",
                )
            )
        functional = (
            cached_axis("functional", 0.55)
            or student_axis("functional")
            or self._structure_axis(
                "unknown", 0.0, "unresolved", "no accepted functional axis"
            )
        )
        content = (
            cached_axis("content", 0.52)
            or student_axis("content")
            or self._structure_axis(
                "unknown", 0.0, "unresolved", "no accepted content axis"
            )
        )
        cached_boundary = cached.get("boundary") or {}
        cached_boundary_confidence = float(
            cached_boundary.get("current_confidence") or 0.0
        )
        cached_boundary_authority = float(
            cached_boundary.get("current_authority") or 0.0
        )
        cached_boundary_consensus = (
            cached_boundary.get("recall_authority")
            == "operator_consensus"
        )
        if self._physical_silence_active:
            boundary_probability = 0.0
            boundary_source = "live_audio_silence"
            boundary_reason = "boundaries are suppressed during physical silence"
        elif (
            cached_boundary_confidence >= 0.55
            or cached_boundary_authority > 0.0
        ):
            boundary_probability = cached_boundary_confidence
            boundary_source = (
                "operator_approved_timeline"
                if cached_boundary_authority > 0.0
                else "operator_consensus_timeline"
                if cached_boundary_consensus
                else "cached_offline_teacher"
            )
            boundary_reason = (
                "operator-approved exact-recording boundary"
                if cached_boundary_authority > 0.0
                else "participant-consensus boundary"
                if cached_boundary_consensus
                else "confident cached boundary"
            )
        elif student_accepted.get("boundary"):
            boundary_probability = float(
                student.get("boundary_probability") or 0.0
            )
            boundary_source = "streaming_student"
            boundary_reason = "approved student boundary"
        else:
            boundary_probability = 0.0
            boundary_source = "unresolved"
            boundary_reason = "no accepted structural boundary"
        boundary_provenance = (
            cached_boundary.get("provenance")
            if boundary_source in {
                "cached_offline_teacher", "operator_approved_timeline",
                "operator_consensus_timeline",
            }
            else student.get("model_path")
            if boundary_source == "streaming_student"
            else None
        )
        boundary_timeline_id = (
            cached_boundary.get("timeline_id")
            if boundary_source in {
                "cached_offline_teacher", "operator_approved_timeline",
                "operator_consensus_timeline",
            }
            else None
        )
        boundary = self._structure_axis(
            (
                "boundary"
                if max(boundary_probability, cached_boundary_authority) >= 0.5
                else "none"
            ),
            boundary_probability,
            boundary_source,
            boundary_reason,
            provenance=boundary_provenance,
            timeline_id=boundary_timeline_id,
            model_confidence=cached_boundary_confidence,
            operator_trust=(
                cached_boundary_authority
                if boundary_source == "operator_approved_timeline"
                else 0.0
            ),
            recall_authority=cached_boundary.get("recall_authority"),
        )
        self._last_raw_observation = raw_observation
        self._effective_structure = {
            "schema": "lumen_structure_resolution_v2",
            "source": energy["source"],
            "section": energy["label"],
            "confidence": energy["decision_confidence"],
            "axes": {
                "functional": functional,
                "energy": energy,
                "content": content,
                "boundary": boundary,
            },
            "audio_gate": {
                "quiet_packet": quiet_packet,
                "quiet_for_s": quiet_for,
                "silence_active": self._physical_silence_active,
                "rms": metrics.rms,
                "dbfs": metrics.dbfs,
                "peak": metrics.peak,
            },
            "beat_timing_authority": "audio_sample_clock",
        }
        return replace(
            raw_observation,
            section=str(energy["label"]),
            section_confidence=float(energy["decision_confidence"]),
        )

    def _resolved_runtime_context(self) -> dict[str, Any]:
        axes = self._effective_structure.get("axes") or {}
        energy = axes.get("energy") or {}
        boundary = axes.get("boundary") or {}
        return {
            "functional": str(
                (axes.get("functional") or {}).get("label") or "unknown"
            ),
            "energy": str(energy.get("label") or "unknown"),
            "content": str(
                (axes.get("content") or {}).get("label") or "unknown"
            ),
            # Motion scaling belongs to the selected energy axis. A confident
            # functional label must not inflate a weak energy decision.
            "confidence": float(
                energy.get("decision_confidence", energy.get("confidence"))
                or 0.0
            ),
            "boundary_probability": float(
                boundary.get(
                    "decision_confidence", boundary.get("confidence")
                ) or 0.0
            ),
            "resolution": deepcopy(self._effective_structure),
        }

    def _set_non_audio_structure(
        self,
        observation: MusicalObservation,
        *,
        source: str,
    ) -> None:
        energy = self._structure_axis(
            observation.section,
            observation.section_confidence,
            source,
            "generated non-audio observation",
        )
        unknown = self._structure_axis(
            "unknown", 0.0, source, "not supplied by generated observation"
        )
        boundary = self._structure_axis(
            "none", 0.0, source, "no generated structural boundary"
        )
        self._last_raw_observation = observation
        self._effective_structure = {
            "schema": "lumen_structure_resolution_v2",
            "source": source,
            "section": observation.section,
            "confidence": observation.section_confidence,
            "axes": {
                "functional": unknown,
                "energy": energy,
                "content": unknown,
                "boundary": boundary,
            },
            "beat_timing_authority": "generated_clock",
        }

    def _refresh_recalled_choreography(
        self,
        runtime: PerformanceRuntime,
        observation: MusicalObservation,
    ) -> None:
        """Atomically adopt candidates prepared outside the audio loop."""

        ids = self._prepared_recalled_ids
        if (
            runtime is not self._recalled_choreography_runtime
            or ids != self._recalled_choreography_ids
        ):
            runtime.set_recalled_choreography(
                self._prepared_recalled_choreography
            )
            self._recalled_choreography_ids = ids
            self._recalled_choreography_runtime = runtime

    def _schedule_memory_context_poll(self) -> None:
        """Keep SQLite/JSON timeline recall off the audio/DMX thread."""

        thread = self._memory_context_thread
        if thread is not None and thread.is_alive():
            return
        if time.monotonic() - self._memory_context_last_poll < 2.0:
            return
        self._memory_context_thread = threading.Thread(
            target=self._poll_memory_context,
            name="lumen-memory-context",
            daemon=True,
        )
        self._memory_context_thread.start()

    def _poll_memory_context(self) -> None:
        try:
            self._poll_memory_context_once()
        except Exception as error:
            with self._lock:
                self._memory_context_error = str(error)
        finally:
            self._memory_context_last_poll = time.monotonic()

    def _poll_memory_context_once(self) -> None:
        started = time.monotonic()
        with self._lock:
            song_id = self.song_id
            media = self.media
            observation = self.observation
        position_ms = self._media_position_ms()
        cached_structure = None
        if media is not None and position_ms is not None:
            cached_structure = self.memory.cached_structure_at(
                provider=media.provider,
                provider_item_id=media.provider_item_id,
                duration_ms=media.duration_ms,
                playback_position_ms=position_ms,
            )
        placements = self.memory.list_choreography_placements(
            song_id=song_id
        ) if song_id is not None else []
        selected: list[dict[str, Any]] = []
        for placement in placements:
            section_label = str(
                placement.get("section_label") or ""
            ).strip()
            start_ms = placement.get("start_ms")
            end_ms = placement.get("end_ms")
            time_matches = False
            if position_ms is not None and start_ms is not None:
                effective_end = end_ms
                if effective_end is None:
                    duration_beats = float(
                        placement.get("duration_beats") or 8.0
                    )
                    bpm = observation.bpm or 120.0
                    # Recall is polled asynchronously and choreography changes
                    # are admitted only on phrase boundaries. Keep an authored
                    # placement discoverable for one additional two-bar phrase
                    # so a call made just after a boundary cannot expire before
                    # the next non-interrupting handoff.
                    effective_end = int(
                        int(start_ms)
                        + (duration_beats + 8.0) * 60_000.0 / bpm
                    )
                time_matches = int(start_ms) <= position_ms < int(
                    effective_end
                )
            section_matches = bool(
                start_ms is None
                and section_label
                and section_label == observation.section
            )
            if time_matches or section_matches:
                selected.append(placement)
        sequences: list[ChoreographySequence] = []
        sequence_versions: list[str] = []
        for placement in selected:
            stored = self.memory.choreography_sequence(
                str(placement["sequence_id"])
            )
            if stored is None:
                continue
            steps: list[ChoreographyStep] = []
            for item in stored.get("steps") or []:
                strobe = item.get("strobe")
                if isinstance(strobe, dict):
                    strobe_value = (
                        float(strobe.get("rate") or 0.0)
                        if strobe.get("enabled") is not False
                        else 0.0
                    )
                else:
                    strobe_value = float(strobe or 0.0)
                parameters = item.get("parameters") or {}
                steps.append(ChoreographyStep(
                    start_beat=float(item["start_beat"]),
                    duration_beats=float(item["duration_beats"]),
                    fixture_scope=str(
                        item.get("fixture_scope")
                        or stored.get("fixture_scope")
                        or "overall"
                    ),
                    routine=str(item["routine"]),
                    intensity=clamp(
                        float(item.get("intensity", 1.0)), 0.0, 1.0
                    ),
                    palette=item.get("palette"),
                    strobe=clamp(strobe_value, 0.0, 1.0),
                    beat_sync=clamp(
                        float(parameters.get("beat_sync", 1.0)), 0.0, 1.0
                    ),
                    motion_speed=clamp(
                        float(parameters.get("motion_speed", 0.5)), 0.0, 1.0
                    ),
                    travel_size=clamp(
                        float(parameters.get("travel_size", 1.0)), 0.0, 1.0
                    ),
                    activity_density=clamp(
                        float(parameters.get("activity_density", 1.0)),
                        0.0,
                        1.0,
                    ),
                    brightness=(
                        None
                        if parameters.get("brightness") is None
                        else clamp(
                            float(parameters["brightness"]), 0.0, 1.0
                        )
                    ),
                    strobe_enabled=(
                        bool(strobe.get("enabled"))
                        if isinstance(strobe, dict)
                        and strobe.get("enabled") is not None
                        else None
                    ),
                    strobe_rate=(
                        clamp(
                            float(strobe.get("rate") or 0.0), 0.0, 1.0
                        )
                        if isinstance(strobe, dict) else None
                    ),
                    cue_timing=clamp(
                        float(parameters.get("cue_timing", 1.0)), 0.0, 1.0
                    ),
                    entry_behavior=str(
                        item.get("entry_behavior") or "phrase_boundary"
                    ),
                    exit_behavior=str(
                        item.get("exit_behavior") or "resolve"
                    ),
                ))
            if steps:
                sequences.append(ChoreographySequence(
                    # A reusable sequence can be placed more than once in one
                    # song. Give each placement revision a stable live identity
                    # so consuming one time-window call cannot suppress another.
                    sequence_id=(
                        f"{stored['id']}#placement:{placement.get('id', '')}:"
                        f"p{placement.get('revision', 0)}:"
                        f"s{stored.get('revision', 0)}"
                    ),
                    steps=tuple(steps),
                    source="operator_song_timeline",
                    base_priority=1.0,
                ))
                sequence_versions.append(
                    f"{stored['id']}:{stored.get('revision', 0)}:"
                    f"{placement.get('id', '')}:"
                    f"{placement.get('revision', 0)}"
                )
        ids = tuple(sequence_versions)
        with self._lock:
            # Drop results if the track changed while SQLite was queried.
            if song_id == self.song_id and media == self.media:
                self._cached_structure_prediction = cached_structure
                self._cached_structure_key = (
                    None
                    if media is None
                    else f"{media.provider}:{media.provider_item_id}"
                )
                self._cached_structure_checked_at = time.monotonic()
                self._prepared_recalled_choreography = tuple(sequences)
                self._prepared_recalled_ids = ids
                self._recalled_choreography_checked_at = time.monotonic()
                self._memory_context_last_poll = time.monotonic()
                self._memory_context_last_duration_ms = round(
                    (time.monotonic() - started) * 1000.0, 2
                )
                self._memory_context_error = None

    def _load_choreography_model(self) -> SequencePreferenceModel:
        path = self._choreography_model_path
        if not path.is_file():
            return SequencePreferenceModel()
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            model = SequencePreferenceModel.from_state_dict(state)
            if int(state.get("version", 0)) != model.STATE_VERSION:
                temporary = path.with_suffix(path.suffix + ".partial")
                temporary.write_text(
                    json.dumps(model.state_dict(), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                temporary.replace(path)
            return model
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Keep the invalid file for diagnosis; never replace it merely
            # because application startup could not parse it.
            return SequencePreferenceModel()

    def _save_choreography_model(self) -> None:
        """Queue a durable snapshot while keeping feedback acknowledgement fast."""

        with self._model_save_condition:
            self._model_save_requested += 1
            self._model_save_condition.notify_all()

    def _write_choreography_model(self) -> None:
        path = self._choreography_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(
            json.dumps(
                self._choreography_model.state_dict(),
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _choreography_model_save_worker(self) -> None:
        while True:
            with self._model_save_condition:
                self._model_save_condition.wait_for(lambda: (
                    self._model_save_requested
                    > self._model_save_completed
                    or self._model_save_stop
                ))
                if (
                    self._model_save_stop
                    and self._model_save_requested
                    <= self._model_save_completed
                ):
                    return
                target = self._model_save_requested
                # A listening-room burst normally arrives within a fraction
                # of a second. Extend the target while requests are arriving,
                # then persist only the final equivalent model state.
                while not self._model_save_stop:
                    self._model_save_condition.wait(timeout=0.15)
                    if self._model_save_requested == target:
                        break
                    target = self._model_save_requested
            try:
                self._write_choreography_model()
            except Exception as error:
                with self._model_save_condition:
                    self._model_save_error = str(error)
            finally:
                with self._model_save_condition:
                    self._model_save_completed = max(
                        self._model_save_completed, target
                    )
                    self._model_save_condition.notify_all()

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
        sequence_records: list[tuple[str | None, str, Any]] = []
        lanes = learning.get("lanes")
        if isinstance(lanes, dict):
            for lane, lane_learning in lanes.items():
                if not isinstance(lane_learning, dict):
                    continue
                # The reversible preference-model event already stores the
                # complete performed sequence and exact feedback context.
                # Rewriting that same performed sequence into the timeline
                # database on every phone tap added two HDD transactions per
                # listener without adding information. Only an explicitly
                # preferred sequence belongs in song choreography memory.
                sequence_records.append((
                    str(lane),
                    "preferred_sequence",
                    lane_learning.get("preferred_sequence"),
                ))
        else:
            sequence_records.append((
                None,
                "preferred_sequence",
                learning.get("preferred_sequence"),
            ))
        for lane, role, sequence in sequence_records:
            if not isinstance(sequence, dict):
                continue
            steps = sequence.get("steps")
            if not isinstance(steps, list) or not steps:
                continue
            sequence_name = str(sequence.get("sequence_id") or role)
            self.memory.save_choreography_sequence(
                sequence_id=(
                    f"online:{target_song_id}:{lane or 'legacy'}:"
                    f"{role}:{sequence_name}"
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
                    "lane": lane,
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
                        "strobe": {
                            "enabled": float(step.get("strobe", 0.0)) > 0,
                            "rate": step.get("strobe", 0.0),
                        },
                        "entry_behavior": step.get(
                            "entry_behavior", "phrase_boundary"
                        ),
                        "exit_behavior": step.get(
                            "exit_behavior", "resolve"
                        ),
                        "parameters": {
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
    def _feedback_effect(label: str) -> dict[str, float]:
        """Translate one explicit label without collapsing its meaning.

        These keys are the persistent/runtime contract.  In particular, speed,
        distance travelled, and how continuously a routine moves are different
        observations, as are permission to strobe and the strobe's rate.
        """

        return {
            "increase_movement": {"travel_size": 0.28},
            "more_movement": {"travel_size": 0.28},
            "decrease_movement": {"travel_size": -0.28},
            "less_movement": {"travel_size": -0.28},
            "too_busy": {"activity_density": -0.28},
            "not_busy_enough": {"activity_density": 0.28},
            "calm_down": {"activity_density": -0.35},
            "pick_it_up": {"activity_density": 0.35},
            "faster": {"motion_speed": 0.28},
            "slower": {"motion_speed": -0.28},
            "faster_side_arms": {"motion_speed": 0.18},
            "slower_side_arms": {"motion_speed": -0.18},
            "too_bright": {"brightness": -0.25},
            "dimmer": {"brightness": -0.25},
            "too_dim": {"brightness": 0.25},
            "brighter": {"brightness": 0.25},
            "more_intensity": {"brightness": 0.25},
            "no_strobe": {"strobe_enabled": -0.80},
            "no_strobes": {"strobe_enabled": -0.80},
            "less_strobe": {"strobe_enabled": -0.60},
            "less_flashing": {"strobe_enabled": -0.60},
            "strobe": {"strobe_enabled": 0.80},
            "more_strobe": {"strobe_enabled": 0.80},
            "faster_strobe": {"strobe_rate": 0.35},
            "slower_strobe": {"strobe_rate": -0.30},
            "better_beat_sync": {"beat_sync": 0.35},
            "great_timing": {"cue_timing": 0.30},
            "good_timing": {"cue_timing": 0.25},
            "timing_on_point": {"cue_timing": 0.30},
            "great_transition": {"cue_timing": 0.25},
            "bad_timing": {"cue_timing": -0.30},
            "poor_timing": {"cue_timing": -0.30},
            "cool_blue_purple": {"palette": -0.70},
            "warmer_color": {"palette": 0.70},
        }.get(label, {})

    @staticmethod
    def _feedback_note_effect(note: str | None) -> dict[str, Any]:
        """Parse a note into the same literal axes used by explicit buttons."""

        text = (note or "").casefold()
        effect: dict[str, Any] = {}

        def adjust(axis: str, amount: float) -> None:
            effect[axis] = float(effect.get(axis, 0.0)) + amount

        if any(term in text for term in ("no strobe", "stop flashing", "less flash")):
            adjust("strobe_enabled", -0.60)
        if any(term in text for term in ("faster strobe", "faster flash")):
            adjust("strobe_rate", 0.25)
        if any(term in text for term in ("slower strobe", "slower flash")):
            adjust("strobe_rate", -0.25)
        if any(term in text for term in ("strobe", "flash")) and not any(
            term in text for term in ("no strobe", "stop flashing", "less flash")
        ) and not any(term in text for term in (
            "faster strobe", "faster flash", "slower strobe", "slower flash"
        )):
            adjust("strobe_enabled", 0.35)

        motion_text = text
        for phrase in (
            "faster strobe", "faster flash", "slower strobe", "slower flash",
        ):
            motion_text = motion_text.replace(phrase, "")
        if any(term in motion_text for term in (
            "slower", "too fast", "slow movement", "move slowly",
        )):
            adjust("motion_speed", -0.18)
        if any(term in motion_text for term in ("faster", "speed up")):
            adjust("motion_speed", 0.18)
        if any(term in text for term in (
            "more movement", "larger movement", "bigger movement", "more travel",
        )):
            adjust("travel_size", 0.18)
        if any(term in text for term in (
            "less movement", "smaller movement", "less travel", "too wide",
        )):
            adjust("travel_size", -0.18)
        if any(term in text for term in (
            "pick it up", "not busy enough", "more active", "busier",
        )):
            adjust("activity_density", 0.18)
        if any(term in text for term in (
            "calm", "too busy", "less active", "give it space",
        )):
            adjust("activity_density", -0.18)
        if any(term in text for term in ("blue", "purple", "cool")):
            adjust("palette", -0.70)
        if any(term in text for term in ("warm", "red", "amber")):
            adjust("palette", 0.70)
        if any(term in text for term in ("brighter", "too dim")):
            adjust("brightness", 0.25)
        if any(term in text for term in ("dimmer", "too bright")):
            adjust("brightness", -0.25)
        if any(term in text for term in (
            "better beat sync", "on the beat", "lock to the beat",
        )):
            adjust("beat_sync", 0.25)
        if any(term in text for term in ("off beat", "out of sync")):
            adjust("beat_sync", -0.25)
        if any(term in text for term in (
            "timing good", "timing was good", "timing on point", "great timing",
        )):
            adjust("cue_timing", 0.25)
        if any(term in text for term in (
            "timing off", "bad timing", "poor timing", "missed the cue",
        )):
            adjust("cue_timing", -0.25)

        routines: dict[str, float] = {}
        for routine in REHEARSAL_ROUTINE_IDS:
            spoken = routine.replace("_", " ")
            if spoken not in text and routine not in text:
                continue
            negative = any(
                phrase in text
                for phrase in (
                    f"no {spoken}", f"less {spoken}", f"stop {spoken}",
                    f"don't use {spoken}", f"do not use {spoken}",
                )
            )
            routines[routine] = -0.50 if negative else 0.50
        if routines:
            effect["routines"] = routines
        return effect

    @staticmethod
    def _feedback_gesture_effect(label: str, gesture: str | None) -> dict[str, float]:
        if gesture and label in {"liked_this", "hold_this", "great_timing", "perfect_motion", "more_like_this", "great_transition"}:
            return {gesture: 0.42}
        if gesture and label in {"bad_timing", "too_busy", "wrong_look"}:
            return {gesture: -0.42}
        return {}

    @staticmethod
    def _feedback_routine_effect(label: str, routine: str | None) -> dict[str, float]:
        if routine and label in {"liked_this", "hold_this", "great_timing", "perfect_motion", "more_like_this", "great_transition"}:
            return {routine: 0.50}
        if routine and label in {"bad_timing", "too_busy", "wrong_look"}:
            return {routine: -0.50}
        return {}

    def _rebuild_feedback_biases(self) -> None:
        """Reconstruct preferences with recency decay and agreement confidence."""
        literal_axes = (
            "motion_speed",
            "travel_size",
            "activity_density",
            "brightness",
            "palette",
            "strobe_enabled",
            "strobe_rate",
            "beat_sync",
            "cue_timing",
        )
        rows = self.memory.all_feedback()
        now = time.time() * 1000.0
        buckets: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        listeners: dict[str, set[str]] = {}
        song_identity: dict[int, bool] = {}
        for row in rows:
            scope = row.get("scope", "overall")
            song_id = row.get("song_id")
            identified_song = False
            if song_id is not None:
                numeric_song_id = int(song_id)
                if numeric_song_id not in song_identity:
                    song = self.memory.get_song(numeric_song_id) or {}
                    provider_item_id = str(
                        song.get("provider_item_id") or ""
                    ).casefold()
                    song_identity[numeric_song_id] = bool(
                        provider_item_id
                        and not provider_item_id.startswith("unidentified:")
                    )
                identified_song = song_identity[numeric_song_id]
            song_key = (
                f"song:{song_id}" if identified_song else None
            )
            artists = [
                str(artist).casefold().strip()
                for artist in row.get("song_artists", ())
                if str(artist).strip()
            ]
            section = str(row.get("section") or "").strip().casefold()
            if section in {"", "unknown"}:
                section = None
            lane_context = row.get("lane_context")
            lifetime = (
                str(lane_context.get("lifetime") or "cue").casefold()
                if isinstance(lane_context, dict)
                else "cue"
            )
            if lifetime not in {"cue", "song", "artist", "global"}:
                lifetime = "cue"
            if scope == "group" and row.get("fixture_id") == "movers":
                target_ids: list[str | None] = [
                    fixture.fixture_id for fixture in self.rig.fixtures
                ]
            elif scope == "group" and row.get("fixture_id") == "center":
                target_ids = [
                    fixture.fixture_id
                    for fixture in self.rig.auxiliary_fixtures
                ]
            elif scope == "overall":
                # None means every fixture lane. It is a target scope, not a
                # temporal lifetime.
                target_ids = [None]
            else:
                target = str(row.get("fixture_id") or "")
                target_ids = [target] if target else []

            context_keys = []
            for target in target_ids:
                base_key = "overall" if target is None else target
                if lifetime == "global" or not song_key:
                    context_keys.append(base_key)
                elif lifetime == "artist" and artists:
                    context_keys.extend(
                        f"artist:{artist}"
                        + ("" if target is None else f":fixture:{target}")
                        for artist in artists
                    )
                elif lifetime == "song" or not section:
                    context_keys.append(
                        song_key
                        + ("" if target is None else f":fixture:{target}")
                    )
                else:
                    context_keys.append(
                        f"{song_key}:section:{section}"
                        + ("" if target is None else f":fixture:{target}")
                    )
            if not context_keys:
                continue
            age_days = max(0.0, (now - float(row.get("created_unix_ms") or now)) / 86_400_000.0)
            decay = math.exp(-age_days / 21.0)
            label_effect = self._feedback_effect(str(row.get("label", "")))
            note_effect = self._feedback_note_effect(row.get("note"))
            effect = {
                axis: float(label_effect.get(axis, 0.0))
                + float(note_effect.get(axis, 0.0))
                for axis in literal_axes
            }
            raw_value = row.get("value")
            note_has_semantics = bool(
                row.get("note")
                and (
                    any(abs(effect[axis]) > 0 for axis in literal_axes)
                    or bool(note_effect.get("routines"))
                )
            )
            weight = decay * min(
                1.0,
                1.0
                if note_has_semantics
                else abs(float(1.0 if raw_value is None else raw_value)),
            )
            for key in context_keys:
                bucket = buckets.setdefault(key, {
                    **{axis: 0.0 for axis in literal_axes},
                    "weight": 0.0,
                    "gestures": {},
                    "routines": {},
                })
                for axis in literal_axes:
                    bucket[axis] = float(bucket.get(axis, 0.0)) + (
                        effect[axis] * weight
                    )
                bucket["weight"] += weight
                gesture = str(row.get("gesture") or "")
                gesture_deltas = self._feedback_gesture_effect(str(row.get("label", "")), gesture)
                if gesture_deltas:
                    gestures = bucket.setdefault("gestures", {})
                    for gesture_name, gesture_delta in gesture_deltas.items():
                        gestures[gesture_name] = gestures.get(gesture_name, 0.0) + gesture_delta * weight
                routine = str(row.get("routine") or "")
                routine_deltas = self._feedback_routine_effect(
                    str(row.get("label", "")), routine
                )
                routines = bucket.setdefault("routines", {})
                for routine_name, routine_delta in routine_deltas.items():
                    routines[routine_name] = routines.get(routine_name, 0.0) + routine_delta * weight
                for routine_name, routine_delta in dict(
                    note_effect.get("routines") or {}
                ).items():
                    routines[routine_name] = (
                        routines.get(routine_name, 0.0)
                        + float(routine_delta) * weight
                    )
                counts[key] = counts.get(key, 0) + 1
                listeners.setdefault(key, set()).add(
                    str(
                        row.get("participant_id")
                        or f"legacy:{row.get('id')}"
                    )
                )
        rebuilt: dict[str, dict[str, Any]] = {}
        for key, bucket in buckets.items():
            listener_count = max(1, len(listeners.get(key, set())))
            repeat_count = max(0, counts[key] - listener_count)
            confidence = clamp(
                0.45
                + 0.14 * (listener_count - 1)
                + 0.06 * repeat_count,
                0.0,
                1.0,
            )
            normalizer = max(float(bucket.get("weight", 0.0)), 1e-9)
            resolved_axes = {
                axis: clamp(
                    float(bucket.get(axis, 0.0))
                    / normalizer
                    * confidence,
                    -1.0,
                    1.0,
                )
                for axis in literal_axes
            }
            rebuilt[key] = {
                **resolved_axes,
                # Compatibility read view for older runtime/UI clients. New
                # consumers must use the literal keys above. Strobe rate is
                # deliberately excluded from `strobe` so "faster strobe"
                # cannot accidentally become "enable strobe" after restart.
                "motion": clamp(
                    resolved_axes["motion_speed"]
                    + resolved_axes["travel_size"]
                    + resolved_axes["activity_density"],
                    -1.0,
                    1.0,
                ),
                "intensity": resolved_axes["brightness"],
                "strobe": resolved_axes["strobe_enabled"],
                "gestures": {
                    name: clamp(value / normalizer * confidence, -1.0, 1.0)
                    for name, value in bucket.get("gestures", {}).items()
                },
                "routines": {
                    name: clamp(value / normalizer * confidence, -1.0, 1.0)
                    for name, value in bucket.get("routines", {}).items()
                },
                "evidence": {
                    "listeners": listener_count,
                    "events": counts[key],
                    "repeat_events": repeat_count,
                    "confidence": confidence,
                },
            }
        with self._lock:
            self._feedback_biases = rebuilt

    def _queue_feedback_bias_refresh(
        self, lanes: Iterable[str]
    ) -> None:
        with self._feedback_refresh_condition:
            self._feedback_refresh_requested += 1
            self._feedback_refresh_lanes.update(
                lane for lane in lanes if lane in CHOREOGRAPHY_LANES
            )
            self._feedback_refresh_condition.notify_all()

    def _feedback_bias_refresh_worker(self) -> None:
        while True:
            with self._feedback_refresh_condition:
                self._feedback_refresh_condition.wait_for(lambda: (
                    self._feedback_refresh_requested
                    > self._feedback_refresh_completed
                    or self._feedback_refresh_stop
                ))
                if (
                    self._feedback_refresh_stop
                    and self._feedback_refresh_requested
                    <= self._feedback_refresh_completed
                ):
                    return
                target = self._feedback_refresh_requested
                while not self._feedback_refresh_stop:
                    self._feedback_refresh_condition.wait(timeout=0.12)
                    if self._feedback_refresh_requested == target:
                        break
                    target = self._feedback_refresh_requested
                lanes = set(self._feedback_refresh_lanes)
                self._feedback_refresh_lanes.clear()
            try:
                self._rebuild_feedback_biases()
                with self._lock:
                    runtime = self._runtime
                    biases = deepcopy(self._feedback_biases)
                    self._status_sequence += 1
                if runtime is not None:
                    runtime.replace_feedback(
                        biases,
                        replan_lanes=lanes,
                    )
                with self._feedback_refresh_condition:
                    self._feedback_refresh_error = None
            except Exception as error:
                with self._feedback_refresh_condition:
                    self._feedback_refresh_error = str(error)
            finally:
                with self._feedback_refresh_condition:
                    self._feedback_refresh_completed = max(
                        self._feedback_refresh_completed, target
                    )
                    self._feedback_refresh_condition.notify_all()

    def _read_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("operator settings must be a JSON object")
        return payload

    def _save_settings(self, settings: dict[str, Any] | None = None) -> None:
        payload = dict(self._settings if settings is None else settings)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.settings_path)

    def _load_motion_tunings(self) -> GroupMotionTunings:
        try:
            payload = json.loads(self.motion_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except (OSError, ValueError, TypeError):
            payload = {}
        return merged_group_motion_tunings(
            payload if isinstance(payload, dict) else {}
        )

    def _save_motion_tunings(
        self, payload: dict[str, Any] | None = None
    ) -> None:
        serialized = payload or GroupMotionTunings(
            movers=self.motion_tunings,
            center=self.center_motion_tunings,
        ).as_dict()
        self.motion_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.motion_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(serialized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.motion_path)

    def patch_motion_routine(self, payload: dict[str, Any]) -> dict[str, Any]:
        routine = str(payload.get("routine", "")).strip().lower()
        if routine not in DEFAULT_MOTION_TUNINGS:
            raise ValueError("unknown motion routine")
        scope = canonical_motion_scope(
            payload.get("scope", self.rehearsal.scope)
        )
        if scope == "overall":
            raise ValueError(
                "choose Movers or Center before editing a routine"
            )
        with self._configuration_lock:
            with self._lock:
                target: dict[str, MotionTuning] | dict[str, CenterMotionTuning]
                defaults: dict[str, MotionTuning] | dict[str, CenterMotionTuning]
                if scope == "center":
                    target = self.center_motion_tunings
                    defaults = DEFAULT_CENTER_MOTION_TUNINGS
                else:
                    target = self.motion_tunings
                    defaults = DEFAULT_MOTION_TUNINGS
                if str(payload.get("action", "")).lower() == "reset":
                    target[routine] = defaults[routine]  # type: ignore[assignment]
                else:
                    values = payload.get("values", payload)
                    if not isinstance(values, dict):
                        raise ValueError("motion values must be an object")
                    target[routine] = target[routine].patch(values)  # type: ignore[assignment,union-attr]
                motion_payload = GroupMotionTunings(
                    movers=self.motion_tunings,
                    center=self.center_motion_tunings,
                ).as_dict()
                if self._runtime is not None:
                    self._runtime.set_motion_tunings(
                        self.motion_tunings, self.center_motion_tunings
                    )
                self._add_event(
                    "rehearsal",
                    f"Tuned {scope} {routine.replace('_', ' ')} motion",
                )
                self._status_sequence += 1
            self._save_motion_tunings(motion_payload)
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
        movers = {
            "values": tuning.as_dict(),
            "defaults": DEFAULT_MOTION_TUNINGS[routine].as_dict(),
            "modified": tuning != DEFAULT_MOTION_TUNINGS[routine],
            "paths": preview_paths(routine, tuning),
            "path": str(self.motion_path),
            "velocity_feasible": feasible,
            "velocity": diagnostics,
        }
        center_tuning = self.center_motion_tunings[routine]
        center = {
            "values": center_tuning.as_dict(),
            "defaults": DEFAULT_CENTER_MOTION_TUNINGS[routine].as_dict(),
            "modified": (
                center_tuning != DEFAULT_CENTER_MOTION_TUNINGS[routine]
            ),
            "path": str(self.motion_path),
        }
        selected = (
            "center" if self.rehearsal.scope == "center" else "movers"
        )
        return {
            **(center if selected == "center" else movers),
            "scope": selected,
            "groups": {"movers": movers, "center": center},
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
        # A CLI research worker is not represented by this application's
        # thread object. Recover dead leases, then honor any live database
        # lease so a second heavy model cannot compete with Live/DMX timing.
        self._recover_abandoned_research_jobs()
        external_research_running = any(
            job["status"] == "running"
            and not str(job.get("worker_id") or "").startswith(
                "lumen-link:"
            )
            for job in self.memory.list_analysis_jobs(limit=100_000)
        )
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
            if external_research_running:
                raise RuntimeError(
                    "offline analysis is running in another Lumen process; "
                    "wait for it to finish before starting the engine"
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
        errors: list[str] = []
        self._stop.set()
        self._research_cancel.set()
        self.lumen_link.close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Audio chunks are short; allow the owner thread to finalize its
            # recorder before closing the DMX object it is still using.
            thread.join(timeout=15.0)
        output = self._output
        if thread is not None and thread.is_alive():
            errors.append("engine owner thread did not stop")
        elif output is not None:
            try:
                output.close()
            except Exception as error:
                errors.append(f"output close failed: {error}")
        recorder = self._training_recorder
        if recorder is not None and not (thread and thread.is_alive()):
            try:
                recorder.stop()
            except Exception as error:
                errors.append(str(error))
        self._trace_stop.set()
        self._trace_thread.join(timeout=10.0)
        if self._trace_thread.is_alive():
            errors.append("performance trace writer did not stop")
        preparation = self._training_prepare_thread
        if preparation is not None and preparation.is_alive():
            preparation.join(timeout=30.0)
            if preparation.is_alive():
                errors.append("training preparation worker did not stop")
        research_worker = self._research_worker_thread
        if research_worker is not None and research_worker.is_alive():
            research_worker.join(timeout=15.0)
            if research_worker.is_alive():
                errors.append("offline research worker did not stop")
        for name, background in (
            ("memory context", self._memory_context_thread),
            ("Spotify poll", self._spotify_poll_thread),
        ):
            if background is not None and background.is_alive():
                background.join(timeout=5.0)
                if background.is_alive():
                    errors.append(f"{name} worker did not stop")
        with self._feedback_refresh_condition:
            self._feedback_refresh_stop = True
            self._feedback_refresh_condition.notify_all()
        self._feedback_refresh_thread.join(timeout=10.0)
        if self._feedback_refresh_thread.is_alive():
            errors.append("feedback refresh worker did not stop")
        elif self._feedback_refresh_error is not None:
            errors.append(
                "feedback refresh failed: "
                f"{self._feedback_refresh_error}"
            )
        with self._model_save_condition:
            self._model_save_stop = True
            self._model_save_condition.notify_all()
        self._model_save_thread.join(timeout=10.0)
        if self._model_save_thread.is_alive():
            errors.append("choreography persistence worker did not stop")
        elif self._model_save_error is not None:
            errors.append(
                "choreography persistence failed: "
                f"{self._model_save_error}"
            )
        if not self._trace_thread.is_alive() and not (
            preparation is not None and preparation.is_alive()
        ):
            try:
                self.memory.checkpoint("TRUNCATE")
            except Exception as error:
                with self._lock:
                    self._add_event(
                        "memory", f"Deferred database checkpoint failed: {error}"
                    )
        if errors:
            raise RuntimeError("; ".join(errors))

    def _trace_worker(self) -> None:
        while not self._trace_stop.is_set() or not self._trace_queue.empty():
            batch: list[dict[str, Any]] = []
            try:
                queued = self._trace_queue.get(timeout=0.20)
            except queue.Empty:
                continue
            try:
                batch.append(self._materialize_trace_item(queued))
            except Exception as error:
                self._trace_queue.task_done()
                with self._lock:
                    self._add_event(
                        "memory",
                        f"Performance trace snapshot failed: {error}",
                    )
                continue
            deadline = time.monotonic() + 2.0
            while len(batch) < 128:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    queued = self._trace_queue.get(
                        timeout=min(0.20, remaining)
                    )
                    batch.append(self._materialize_trace_item(queued))
                except queue.Empty:
                    if self._trace_stop.is_set():
                        break
                except Exception as error:
                    self._trace_queue.task_done()
                    with self._lock:
                        self._add_event(
                            "memory",
                            f"Performance trace snapshot failed: {error}",
                        )
            try:
                self.memory.log_trace_batch(batch)
            except Exception as error:
                with self._lock:
                    self._add_event("memory", f"Performance trace write failed: {error}")
            finally:
                for _item in batch:
                    self._trace_queue.task_done()

    def _materialize_trace_item(
        self, item: dict[str, Any]
    ) -> dict[str, Any]:
        """Serialize a trace seed entirely outside the live control path."""

        if item.get("_kind") != "performance_seed":
            return item
        frame: RuntimeFrame = item["frame"]
        observation: MusicalObservation = item["observation"]
        raw_observation: MusicalObservation = item["raw_observation"]
        audio_metrics: AudioInputMetrics | None = item["audio_metrics"]
        # The control thread captured this beside the exact RuntimeFrame.
        # Reading mutable runtime state here could pair frame N's DMX with a
        # later phrase/step after the asynchronous trace writer caught up.
        choreography_snapshot = item.get("choreography_snapshot")
        decision = asdict(frame.decision)
        decision["gesture"] = frame.decision.gesture.value
        return {
            "_kind": "performance",
            "session_id": item["session_id"],
            "song_id": item["song_id"],
            "position_ms": item["position_ms"],
            "payload": {
                "schema": "lumen_performance_trace_v2",
                "raw_observation": asdict(raw_observation),
                "resolved_observation": asdict(observation),
                "observation": asdict(observation),
                "audio_metrics": (
                    {
                        key: value
                        for key, value in asdict(audio_metrics).items()
                        if key != "waveform"
                    }
                    if audio_metrics is not None
                    else {}
                ),
                "decision": decision,
                "controls": asdict(item["controls"]),
                "solutions": [
                    {
                        "fixture_id": solution.fixture_id,
                        "pan_deg": solution.pan_deg,
                        "tilt_deg": solution.tilt_deg,
                        "branch": solution.branch,
                    }
                    for solution in frame.solutions
                ],
                # These are the final fixture-local bytes from the same frame
                # that was handed to the DMX output.  Keeping them beside the
                # semantic decision closes the audit path from heard music to
                # effective fixture state without reopening the USB adapter.
                "fixture_dmx": self._fixture_dmx_snapshot(frame),
                "effective_outputs": [
                    output.as_dict() for output in frame.effective_outputs
                ],
                "choreography_runtime": choreography_snapshot,
                "structure_model": deepcopy(item["structure_model"]),
                "structure_resolution": deepcopy(
                    item["structure_resolution"]
                ),
            },
        }

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
        # Monitor/Live can be stopped and restarted without recreating the
        # application. Never carry 30/60-second memories or decoder hysteresis
        # from the previous listening session into the new audio stream.
        self._reset_student_stream()
        self._prepare_dedicated_line_input()
        capture_config = AudioCaptureConfig(device=self.audio_device)
        analyzer = RealtimeAudioAnalyzer(
            capture_config.sample_rate, capture_config.channels
        )
        analysis_generation = self._analysis_generation
        expected_source_frame: int | None = None
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
        control_queue: queue.Queue[_AnalyzedControlFrame] = queue.Queue(
            maxsize=16
        )
        analysis_finished = threading.Event()
        control_errors: list[BaseException] = []
        control_thread: threading.Thread | None = None
        try:
            capture = AlsaLineIn(capture_config)
            with ContinuouslyDrainedAudio(capture) as buffered_capture:
                with self._live_state_lock:
                    self._active_audio_capture = buffered_capture
                control_thread = threading.Thread(
                    target=self._run_live_control_clock,
                    args=(
                        runtime,
                        control_queue,
                        analysis_finished,
                        control_errors,
                        capture_config,
                    ),
                    name="lumen-live-control",
                    daemon=True,
                )
                control_thread.start()
                for captured in buffered_capture.chunks(
                    stop_event=self._stop
                ):
                    if self._stop.is_set():
                        break
                    if control_errors:
                        raise RuntimeError(
                            f"live control clock failed: {control_errors[0]}"
                        ) from control_errors[0]
                    pcm = captured.pcm
                    analysis_started = time.perf_counter()
                    stage_started = analysis_started
                    stage_ms: dict[str, float] = {}
                    with self._live_state_lock:
                        diagnostics = buffered_capture.diagnostics
                        self._audio_queue_depth = buffered_capture.queue_depth
                        self._audio_queue_max_depth = max(
                            self._audio_queue_max_depth,
                            buffered_capture.maximum_queue_depth,
                        )
                        self._audio_queue_delay_ms = max(
                            0.0,
                            (
                                time.monotonic()
                                - captured.captured_monotonic_s
                            ) * 1000.0,
                        )
                        self._audio_capture_diagnostics = diagnostics
                    timeline_discontinuity = False
                    if analysis_generation != self._analysis_generation:
                        analyzer.reset()
                        self._reset_student_stream(
                            reset_physical_silence=False
                        )
                        timeline_discontinuity = True
                        analysis_generation = self._analysis_generation
                    packet_frames = captured.frame_count
                    audio_discontinuity = bool(
                        expected_source_frame is not None
                        and captured.source_start_frame
                        != expected_source_frame
                    )
                    if audio_discontinuity:
                        # Temporal analysis cannot bridge missing physical PCM.
                        # The control clock receives the matching reset command
                        # in-order with the first observation after the gap.
                        analyzer.reset()
                        self._reset_student_stream(
                            reset_physical_silence=False
                        )
                    expected_source_frame = (
                        captured.source_start_frame + packet_frames
                    )
                    timestamp = captured.timestamp_s
                    raw_observation = analyzer.analyze_pcm16(
                        pcm, timestamp_s=timestamp
                    )
                    stage_ms["analyze"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    stage_started = time.perf_counter()
                    audio_metrics = analyzer.last_metrics
                    with self._live_state_lock:
                        self._tempo_diagnostics = dict(
                            analyzer.tempo_diagnostics
                        )
                    self._apply_student_structure(
                        raw_observation, audio_metrics
                    )
                    observation = self._resolve_structure(
                        raw_observation, audio_metrics
                    )
                    runtime_context = self._resolved_runtime_context()
                    stage_ms["structure"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    stage_started = time.perf_counter()
                    with self._live_state_lock:
                        active_output_frame = self.frame
                        training_choreography = (
                            self._runtime_choreography_snapshot
                        )
                    training_context = {
                        # These objects are either frozen dataclasses or
                        # copy-on-write dictionaries. Their references remain
                        # the exact generation paired with this PCM packet
                        # until the recorder worker serializes them.
                        "controls": replace(self.controls),
                        "media": self.media,
                        "structure_model": self._student_prediction,
                        "structure_resolution": self._effective_structure,
                        "choreography_runtime": training_choreography,
                    }
                    audio_frame: int | None = None
                    if recorder is not None:
                        sample_unix_ms = round(
                            time.time() * 1000
                            - max(0.0, time.monotonic() - timestamp) * 1000
                        )
                        audio_frame = recorder.submit(
                            pcm,
                            song_id=self.song_id,
                            position_ms=self._media_position_ms(
                                at_monotonic_s=timestamp
                            ),
                            source_start_frame=captured.source_start_frame,
                            captured_unix_ms=sample_unix_ms,
                            payload=lambda frame=active_output_frame,
                            resolved=observation,
                            metrics=audio_metrics,
                            raw=raw_observation,
                            context=training_context: (
                                self._training_frame_payload(
                                    resolved,
                                    frame,
                                    metrics,
                                    raw_observation=raw,
                                    captured_context=context,
                                )
                            ),
                        )
                    stage_ms["recorder_submit"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    self._offer_control_frame(
                        control_queue,
                        _AnalyzedControlFrame(
                            observation=observation,
                            raw_observation=raw_observation,
                            audio_metrics=audio_metrics,
                            audio_bytes=len(pcm),
                            training_audio_frame=audio_frame,
                            runtime_context=runtime_context,
                            analysis_started_perf_s=analysis_started,
                            analysis_stages_ms=stage_ms,
                            timeline_discontinuity=timeline_discontinuity,
                            audio_discontinuity=audio_discontinuity,
                        ),
                    )
        finally:
            analysis_finished.set()
            if control_thread is not None:
                control_thread.join(timeout=3.0)
                if control_thread.is_alive():
                    control_errors.append(
                        RuntimeError("live control clock did not stop")
                    )
            with self._live_state_lock:
                self._active_audio_capture = None
            if recorder is not None:
                recorder.stop()
                final_status = recorder.status()
                training_history = self.memory.training_summary()
                with self._lock:
                    self._training_history = training_history
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
                    self._start_training_preparation(self._session_id)
        if control_errors:
            raise RuntimeError(
                f"live control clock failed: {control_errors[0]}"
            ) from control_errors[0]

    def _offer_control_frame(
        self,
        control_queue: queue.Queue[_AnalyzedControlFrame],
        item: _AnalyzedControlFrame,
    ) -> None:
        """Publish analysis without ever waiting for the show clock."""

        while True:
            try:
                control_queue.put_nowait(item)
            except queue.Full:
                try:
                    control_queue.get_nowait()
                except queue.Empty:
                    continue
                with self._live_state_lock:
                    self._control_queue_drops += 1
                continue
            with self._live_state_lock:
                depth = control_queue.qsize()
                self._control_queue_depth = depth
                self._control_queue_max_depth = max(
                    self._control_queue_max_depth, depth
                )
            return

    def _run_live_control_clock(
        self,
        runtime: PerformanceRuntime,
        control_queue: queue.Queue[_AnalyzedControlFrame],
        analysis_finished: threading.Event,
        errors: list[BaseException],
        capture_config: AudioCaptureConfig,
    ) -> None:
        """React immediately to analysis and bridge genuine gaps at 30 Hz."""

        fallback_period_s = 1.0 / 30.0
        packet_period_s = (
            capture_config.chunk_frames / capture_config.sample_rate
        )
        stale_threshold_s = max(
            fallback_period_s, packet_period_s * 1.25
        )
        fallback_at = time.monotonic() + stale_threshold_s
        active: _AnalyzedControlFrame | None = None
        try:
            while not self._stop.is_set():
                if analysis_finished.is_set() and control_queue.empty():
                    return
                newest: _AnalyzedControlFrame | None = None
                # The 30 Hz clock is faster than the 23.4 Hz PCM source. Keep
                # ordinary arecord/CPU bursts in FIFO order so no beat or
                # structural evidence is discarded. Collapse only a genuinely
                # stale backlog that already exceeds a quarter second.
                if control_queue.qsize() > 8:
                    while control_queue.qsize() > 2:
                        try:
                            newest = control_queue.get_nowait()
                        except queue.Empty:
                            break
                        with self._live_state_lock:
                            self._control_queue_drops += 1
                timeout = max(0.0, fallback_at - time.monotonic())
                try:
                    newest = control_queue.get(timeout=timeout)
                except queue.Empty:
                    newest = None
                with self._live_state_lock:
                    self._control_queue_depth = control_queue.qsize()
                fresh_analysis = newest is not None
                if newest is not None:
                    active = newest
                    late_ms = 0.0
                    fallback_at = time.monotonic() + stale_threshold_s
                else:
                    now = time.monotonic()
                    late_ms = max(0.0, (now - fallback_at) * 1000.0)
                    fallback_at = now + fallback_period_s
                if active is None:
                    continue

                if fresh_analysis and active.timeline_discontinuity:
                    runtime.notify_timeline_discontinuity()
                if fresh_analysis and active.audio_discontinuity:
                    runtime.notify_audio_discontinuity()
                # Even a fresh FIFO result may have waited through a short
                # burst. Resolve its phase on the authoritative source clock
                # so output does not deliberately reproduce queue latency.
                observation = self._extrapolate_control_observation(
                    active.observation, capture_config
                )
                raw_observation = replace(
                    active.raw_observation,
                    timestamp_s=observation.timestamp_s,
                    beat_phase=observation.beat_phase,
                    bar_phase=observation.bar_phase,
                    beat_pulse=observation.beat_pulse,
                    onset_strength=observation.onset_strength,
                    novelty=observation.novelty,
                )
                stage_ms = dict(active.analysis_stages_ms)
                stage_started = time.perf_counter()
                runtime.set_structure_context(**active.runtime_context)
                self._refresh_recalled_choreography(runtime, observation)
                frame = runtime.step(observation)
                stage_ms["runtime"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                stage_started = time.perf_counter()
                self._accept_runtime_frame(
                    observation,
                    frame,
                    audio_metrics=(
                        active.audio_metrics if fresh_analysis else None
                    ),
                    audio_bytes=(active.audio_bytes if fresh_analysis else 0),
                    training_audio_frame=(
                        active.training_audio_frame
                        if fresh_analysis
                        else None
                    ),
                    raw_observation=raw_observation,
                )
                stage_ms["publish"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                if fresh_analysis:
                    self._record_live_pipeline_timing(
                        stage_ms,
                        total_ms=(
                            time.perf_counter()
                            - active.analysis_started_perf_s
                        ) * 1000.0,
                        budget_ms=(
                            active.audio_metrics.frame_count
                            * 1000.0
                            / capture_config.sample_rate
                        ),
                    )
                with self._live_state_lock:
                    self._control_ticks += 1
                    if not fresh_analysis:
                        self._control_interpolated_ticks += 1
                    self._control_maximum_late_ms = max(
                        self._control_maximum_late_ms, late_ms
                    )
        except BaseException as error:
            errors.append(error)
            self._stop.set()

    def _extrapolate_control_observation(
        self,
        observation: MusicalObservation,
        capture_config: AudioCaptureConfig,
    ) -> MusicalObservation:
        """Advance a short lighting tick on the physical sample clock."""

        target_timestamp = observation.timestamp_s
        capture = self._active_audio_capture
        diagnostics = capture.diagnostics if capture is not None else {}
        origin = diagnostics.get("sample_clock_origin_s")
        source_frames = diagnostics.get("source_frames")
        source_age_ms = diagnostics.get("last_packet_age_ms")
        if (
            origin is not None
            and source_frames is not None
            and source_age_ms is not None
            and float(source_age_ms) <= 750.0
        ):
            target_timestamp = max(
                target_timestamp,
                float(origin)
                + float(source_frames) / capture_config.sample_rate,
            )
        elapsed = max(0.0, target_timestamp - observation.timestamp_s)
        if elapsed <= 0.0:
            return observation
        bpm = observation.bpm
        beat_phase = observation.beat_phase
        bar_phase = observation.bar_phase
        beat_pulse = observation.beat_pulse * math.exp(-elapsed / 0.14)
        if bpm is not None and observation.beat_confidence >= 0.10:
            advanced_beats = elapsed * bpm / 60.0
            beat_phase = (beat_phase + advanced_beats) % 1.0
            bar_phase = (bar_phase + advanced_beats / 4.0) % 1.0
            seconds_from_beat = beat_phase * 60.0 / bpm
            beat_pulse = max(
                beat_pulse,
                math.exp(-seconds_from_beat / 0.14),
            )
        return replace(
            observation,
            timestamp_s=target_timestamp,
            beat_phase=beat_phase,
            bar_phase=bar_phase,
            beat_pulse=clamp(beat_pulse, 0.0, 1.0),
            onset_strength=observation.onset_strength
            * math.exp(-elapsed / 0.12),
            novelty=observation.novelty * math.exp(-elapsed / 0.30),
        )

    def _start_training_preparation(self, session_id: str | None = None) -> None:
        """Build verified recording identities and queue teachers off-thread."""
        requested_session = str(session_id or self._session_id)
        with self._lock:
            running = self._training_prepare_thread
            if running is not None and running.is_alive():
                self._training_prepare_pending = True
                self._training_prepare_pending_session = requested_session
                self._add_event(
                    "memory",
                    "Offline preparation is already processing a prior capture",
                )
                return
            thread = threading.Thread(
                target=self._prepare_training_capture,
                args=(requested_session,),
                name="lumen-offline-preparation",
                daemon=True,
            )
            self._training_prepare_thread = thread
            thread.start()

    def _prepare_training_capture(self, session_id: str) -> None:
        try:
            # The live writers no longer trigger SQLite auto-checkpoints. Fold
            # committed pages from this run before scanning its compact track
            # identities, entirely on the preparation worker.
            self.memory.checkpoint("PASSIVE")
            result = export_research_session_index(
                self.memory, self.training_root, session_id
            )
            research = ResearchJobCoordinator(
                self.memory,
                training_root=self.training_root,
                research_root=self.training_root / "research",
            ).prepare_export(result["path"], queue_songformer=True)
            consensus = self.memory.refresh_operator_structure_consensus()
            self.memory.mark_research_session_prepared(
                session_id, result["path"]
            )
            training_history = self.memory.training_summary()
            with self._lock:
                self._last_training_export = result["path"]
                self._training_history = training_history
                self._add_event(
                    "memory",
                    (
                        "Prepared captured songs and queued "
                        f"{research['jobs_queued']} offline teacher job(s); "
                        f"rebuilt {consensus['songs']} corrected song timeline(s)"
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
                pending_session = self._training_prepare_pending_session
                self._training_prepare_pending = False
                self._training_prepare_pending_session = None
                self._training_prepare_thread = None
            self._schedule_research_readiness_refresh()
            if rerun and not self._stop.is_set():
                self._start_training_preparation(pending_session)

    def _training_frame_payload(
        self,
        observation: MusicalObservation,
        frame: RuntimeFrame | None,
        audio_metrics: AudioInputMetrics | None = None,
        *,
        raw_observation: MusicalObservation | None = None,
        captured_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision: dict[str, Any] | None = None
        if frame is not None:
            decision = asdict(frame.decision)
            decision["gesture"] = frame.decision.gesture.value
        context = captured_context or {}
        captured_media = context.get("media", self.media)
        media: dict[str, Any] | None = None
        if captured_media is not None:
            media = {
                "provider": captured_media.provider,
                "provider_item_id": captured_media.provider_item_id,
                "title": captured_media.title,
                "artists": list(captured_media.artists),
                "album": captured_media.album,
                "duration_ms": captured_media.duration_ms,
                "is_playing": captured_media.is_playing,
                "device_name": captured_media.device_name,
            }
        source_observation = raw_observation or observation
        semantic_payload = (
            self._semantic_audio_payload(source_observation, audio_metrics)
            if audio_metrics is not None
            else {"observation": asdict(observation)}
        )
        return {
            "schema": "lumen_semantic_frame_v1",
            "raw_observation": asdict(source_observation),
            "resolved_observation": asdict(observation),
            "observation": semantic_payload["observation"],
            "audio_metrics": semantic_payload.get("audio_metrics", {}),
            "decision": decision,
            "controls": asdict(context.get("controls", self.controls)),
            "media": media,
            "structure_model": deepcopy(
                context.get("structure_model", self._student_prediction)
            ),
            "structure_resolution": deepcopy(
                context.get(
                    "structure_resolution", self._effective_structure
                )
            ),
            "solutions": [
                {
                    "fixture_id": solution.fixture_id,
                    "pan_deg": solution.pan_deg,
                    "tilt_deg": solution.tilt_deg,
                    "branch": solution.branch,
                }
                for solution in (() if frame is None else frame.solutions)
            ],
            "fixture_dmx": (
                [] if frame is None else self._fixture_dmx_snapshot(frame)
            ),
            "effective_outputs": (
                []
                if frame is None
                else [output.as_dict() for output in frame.effective_outputs]
            ),
            "choreography_runtime": deepcopy(
                context.get("choreography_runtime")
                if captured_context is not None
                else (
                    self._runtime.choreography_snapshot()
                    if self._runtime is not None
                    else None
                )
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
            self._set_non_audio_structure(
                observation, source="simulated_demo"
            )
            runtime.set_structure_context(
                **self._resolved_runtime_context()
            )
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
            self._set_non_audio_structure(
                observation, source="generated_rehearsal"
            )
            runtime.set_structure_context(
                **self._resolved_runtime_context()
            )
            self._accept_runtime_frame(observation, runtime.step(observation))

    def _accept_runtime_frame(
        self,
        observation: MusicalObservation,
        frame: RuntimeFrame,
        *,
        audio_metrics: AudioInputMetrics | None = None,
        audio_bytes: int = 0,
        training_audio_frame: int | None = None,
        raw_observation: MusicalObservation | None = None,
    ) -> None:
        # This is a small semantic snapshot (two planner lanes and the three
        # effective fixture outputs), not persistence work. Capture it before
        # publishing the RuntimeFrame so dashboard/feedback readers can never
        # observe a new decision beside missing or previous choreography.
        active_runtime = self._runtime
        current_choreography_snapshot = (
            active_runtime.choreography_snapshot()
            if active_runtime is not None
            else None
        )
        emit_trace = False
        emit_decision = False
        trace_session_id = ""
        trace_song_id: int | None = None
        trace_controls = replace(self.controls)
        trace_structure_model: dict[str, Any] | None = None
        trace_structure_resolution: dict[str, Any] = {}
        runtime: PerformanceRuntime | None = None
        trace_choreography_snapshot: dict[str, Any] | None = None
        with self._live_state_lock:
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
            if audio_metrics is not None or training_audio_frame is not None:
                self._training_audio_frame = training_audio_frame
            self.observation = observation
            self.frame = frame
            self._runtime_choreography_snapshot = (
                current_choreography_snapshot
            )
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
                emit_trace = True
                trace_session_id = self._session_id
                trace_song_id = self.song_id
                trace_controls = replace(self.controls)
                # Both dictionaries are replaced, not mutated, by the
                # analysis path. Capturing their current references is enough;
                # the trace worker performs the expensive deep copy.
                trace_structure_model = self._student_prediction
                trace_structure_resolution = self._effective_structure
                runtime = active_runtime
            gesture = frame.decision.gesture.value
            if gesture != self._last_gesture:
                self._last_gesture = gesture
                self._add_event(
                    "gesture",
                    f"{gesture.title()}: {frame.decision.reason.split('.')[0]}.",
                )
                if self.song_id is not None:
                    emit_decision = True
                    trace_song_id = self.song_id
            self._status_sequence += 1
        trace_position = self._media_position_ms(
            at_monotonic_s=(
                observation.timestamp_s
                if audio_metrics is not None
                else None
            )
        )
        if emit_trace:
            # Reuse the exact snapshot published with this RuntimeFrame. JSON
            # and deep-copy work remains deferred to the trace writer.
            trace_choreography_snapshot = current_choreography_snapshot
            try:
                self._trace_queue.put_nowait(
                    {
                        "_kind": "performance_seed",
                        "session_id": trace_session_id,
                        "song_id": trace_song_id,
                        "position_ms": trace_position,
                        "frame": frame,
                        "observation": observation,
                        "raw_observation": raw_observation or observation,
                        "audio_metrics": audio_metrics,
                        "controls": trace_controls,
                        "structure_model": trace_structure_model,
                        "structure_resolution": trace_structure_resolution,
                        "choreography_snapshot": (
                            trace_choreography_snapshot
                        ),
                    }
                )
            except queue.Full:
                with self._live_state_lock:
                    self._trace_queue_drops += 1
        if emit_decision and trace_song_id is not None:
            try:
                self._trace_queue.put_nowait(
                    {
                        "_kind": "decision",
                        "decision": frame.decision,
                        "song_id": trace_song_id,
                        "position_ms": trace_position,
                        "observation": observation,
                    }
                )
            except queue.Full:
                with self._live_state_lock:
                    self._trace_queue_drops += 1
        self._schedule_spotify_poll()
        self._schedule_memory_context_poll()

    def _record_live_pipeline_timing(
        self,
        stages_ms: dict[str, float],
        *,
        total_ms: float,
        budget_ms: float,
    ) -> None:
        """Publish inexpensive evidence of audio-to-DMX deadline health."""

        rounded_stages = {
            name: round(value, 3) for name, value in stages_ms.items()
        }
        with self._live_state_lock:
            # Copy-on-write lets the dashboard serialize the previously
            # published timing record without contending with the live path.
            timing = {
                **self._live_pipeline_timing,
                "maximum_stages_ms": dict(
                    self._live_pipeline_timing.get(
                        "maximum_stages_ms", {}
                    )
                ),
            }
            timing["packets"] = int(timing.get("packets", 0)) + 1
            timing["last_total_ms"] = round(total_ms, 3)
            timing["maximum_total_ms"] = round(
                max(float(timing.get("maximum_total_ms", 0.0)), total_ms),
                3,
            )
            timing["budget_ms"] = round(budget_ms, 3)
            if total_ms > budget_ms:
                timing["deadline_misses"] = int(
                    timing.get("deadline_misses", 0)
                ) + 1
            timing["stages_ms"] = rounded_stages
            maximums = timing.setdefault("maximum_stages_ms", {})
            for name, value in stages_ms.items():
                maximums[name] = round(
                    max(float(maximums.get(name, 0.0)), value), 3
                )
            self._live_pipeline_timing = timing

    def _schedule_spotify_poll(self) -> None:
        """Keep network metadata work off the real-time audio/DMX loop."""
        thread = self._spotify_poll_thread
        if thread is not None and thread.is_alive():
            return
        if (
            time.monotonic() - self._last_media_poll
            < SPOTIFY_MEDIA_POLL_INTERVAL_S
        ):
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
        self._audio_queue_depth = 0
        self._audio_queue_max_depth = 0
        self._audio_queue_delay_ms = 0.0
        self._audio_capture_diagnostics = {}
        self._active_audio_capture = None
        self._control_queue_depth = 0
        self._control_queue_max_depth = 0
        self._control_queue_drops = 0
        self._control_ticks = 0
        self._control_interpolated_ticks = 0
        self._control_maximum_late_ms = 0.0
        self._tempo_diagnostics = {}
        self._live_pipeline_timing = {
            "packets": 0,
            "last_total_ms": 0.0,
            "maximum_total_ms": 0.0,
            "deadline_misses": 0,
            "stages_ms": {},
            "maximum_stages_ms": {},
        }
        self._analysis_history.clear()
        self._last_analysis_history_at = None

    def _audio_snapshot_unlocked(self, running: bool) -> dict[str, Any]:
        now = time.monotonic()
        processed_age_ms = (
            None
            if self._audio_last_packet_at is None
            else max(0.0, (now - self._audio_last_packet_at) * 1000.0)
        )
        active_capture = self._active_audio_capture
        capture_diagnostics = (
            active_capture.diagnostics
            if active_capture is not None
            else deepcopy(self._audio_capture_diagnostics)
        )
        source_age_ms = capture_diagnostics.get("last_packet_age_ms")
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
        elif self._audio_packets == 0 and not capture_diagnostics.get(
            "packets_read"
        ):
            state = "waiting"
            label = "WAITING FOR PCM"
            detail = f"ALSA is opening {self.audio_device}; no packet has arrived yet."
        elif (
            source_age_ms is not None
            and float(source_age_ms) > 750.0
        ):
            state = "stale"
            label = "PCM SOURCE STALLED"
            detail = (
                "The capture reader has not received physical PCM for "
                f"{float(source_age_ms) / 1000.0:.1f}s."
            )
        elif (
            processed_age_ms is not None
            and processed_age_ms > 750.0
        ) or (
            self._audio_packets == 0
            and int(capture_diagnostics.get("packets_read") or 0) > 0
        ):
            state = "pipeline_stale"
            label = "ANALYSIS PIPELINE STALLED"
            detail = (
                "Fresh physical PCM is still being captured, but the live "
                "analysis/control pipeline has not completed its next frame."
            )
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
            "capture_queue_depth": self._audio_queue_depth,
            "capture_queue_max_depth": self._audio_queue_max_depth,
            "capture_queue_delay_ms": self._audio_queue_delay_ms,
            "capture_queue": capture_diagnostics,
            "control_clock": {
                "mode": "analysis_event_driven_with_sample_clock_bridge",
                "rate_hz": 30.0,
                "fallback_rate_hz": 30.0,
                "queue_depth": self._control_queue_depth,
                "maximum_queue_depth": self._control_queue_max_depth,
                "coalesced_frames": self._control_queue_drops,
                "ticks": self._control_ticks,
                "interpolated_ticks": self._control_interpolated_ticks,
                "maximum_late_ms": round(
                    self._control_maximum_late_ms, 3
                ),
                "timing_authority": "audio_sample_clock",
            },
            "tempo_clock": deepcopy(self._tempo_diagnostics),
            "live_pipeline": deepcopy(self._live_pipeline_timing),
            "last_packet_age_ms": processed_age_ms,
            "last_processed_frame_age_ms": processed_age_ms,
            "last_source_packet_age_ms": source_age_ms,
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
            center_motion_tunings=self.center_motion_tunings,
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
        with self._configuration_lock:
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
                settings = dict(self._settings)
                self._status_sequence += 1
            self._save_settings(settings)
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
            training_history = self.memory.training_summary()
            with self._lock:
                self._last_training_export = result["path"]
                self._training_history = training_history
                self._add_event(
                    "memory", "Built neural-training dataset manifest"
                )
                self._status_sequence += 1
            return result
        finally:
            self._training_export_lock.release()

    def _load_research_readiness_cache(self) -> dict[str, Any] | None:
        path = self._research_readiness_cache_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        training = payload.get("training")
        if (
            payload.get("schema") != "lumen_operator_readiness_cache_v1"
            or not isinstance(training, dict)
            or payload.get("database_path")
            != str(self.memory_path.resolve(strict=False))
        ):
            return None
        return payload

    def _store_research_readiness_cache(
        self, training: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            "schema": "lumen_operator_readiness_cache_v1",
            "created_unix_ms": int(time.time() * 1000),
            "database_path": str(self.memory_path.resolve(strict=False)),
            "training": deepcopy(training),
        }
        path = self._research_readiness_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        with self._research_readiness_lock:
            self._research_readiness_cache = payload
            self._research_readiness_error = None
        return payload

    def _refresh_research_readiness_cache(self) -> None:
        try:
            database_bytes = self.memory_path.stat().st_size
            if database_bytes < 256 * 1024 * 1024:
                # Tests and new installations complete this in milliseconds.
                training = training_readiness(
                    self.memory,
                    research_root=self.training_root / "research",
                )
            else:
                # A mature library parses hundreds of megabytes of provenance
                # and example JSON. Run that audit in a low-priority process so
                # it cannot contend for the live console's interpreter lock or
                # leave its temporary allocations in the long-lived UI heap.
                command = [
                    sys.executable,
                    "-m",
                    "lumen_engine",
                    "research-status",
                    "--root",
                    str(self.training_root / "research"),
                    "--memory",
                    str(self.memory_path),
                ]
                if shutil.which("nice"):
                    command = ["nice", "-n", "10", *command]
                if shutil.which("ionice"):
                    command = ["ionice", "-c", "2", "-n", "7", *command]
                environment = dict(os.environ)
                source_root = str(PROJECT_DIR / "src")
                existing_python_path = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = (
                    source_root
                    if not existing_python_path
                    else source_root + os.pathsep + existing_python_path
                )
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=3_600,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(
                        "readiness subprocess failed"
                        + (f": {detail}" if detail else "")
                    )
                result = json.loads(completed.stdout)
                training = result.get("training")
                if not isinstance(training, dict):
                    raise RuntimeError(
                        "readiness subprocess returned no training state"
                    )
            self._store_research_readiness_cache(training)
        except Exception as error:
            with self._research_readiness_lock:
                self._research_readiness_error = (
                    f"{type(error).__name__}: {error}"
                )
        finally:
            with self._research_readiness_lock:
                self._research_readiness_thread = None

    def _schedule_research_readiness_refresh(self) -> bool:
        with self._research_readiness_lock:
            thread = self._research_readiness_thread
            if thread is not None and thread.is_alive():
                return True
            thread = threading.Thread(
                target=self._refresh_research_readiness_cache,
                name="lumen-readiness-audit",
                daemon=True,
            )
            self._research_readiness_thread = thread
            thread.start()
            return True

    @staticmethod
    def _pending_research_readiness() -> dict[str, Any]:
        return {
            "recordings_planned": 0,
            "recordings_processed": 0,
            "recordings_captured": 0,
            "recordings_eligible": 0,
            "eligible_teacher_jobs": 0,
            "eligible_teacher_jobs_complete": 0,
            "progress": 0.0,
            "usable_examples": 0,
            "split_counts": {},
            "teacher_errors": [],
            "provenance_errors": [],
            "label_balance": {},
            "train_ready": False,
            "activation_ready": False,
            "blockers": [
                "Lumen is refreshing the offline training summary"
            ],
            "activation_blockers": [],
            "model": {},
        }

    def research_status(
        self, *, wait_for_readiness: bool = True
    ) -> dict[str, Any]:
        # Status polling is also the recovery path for a console that remains
        # open when an external CLI worker dies. Healthy leases are retained;
        # dead or stale leases are atomically requeued and immediately stop
        # holding the Analyze/Train/Live controls disabled.
        self._recover_abandoned_research_jobs()
        result = self.research.status(include_training=False)
        # Exact readiness is intentionally requested only by callers performing
        # an offline operation. Browser bootstrap and polling use the durable
        # cache below and never verify the teacher corpus inline.
        training_payload = (
            training_readiness(
                self.memory,
                research_root=self.training_root / "research",
            )
            if wait_for_readiness
            else None
        )
        if isinstance(training_payload, dict):
            training = deepcopy(training_payload)
            cached = self._store_research_readiness_cache(training)
            readiness_refreshing = False
            readiness_error = None
        else:
            with self._research_readiness_lock:
                cached = deepcopy(self._research_readiness_cache)
                readiness_thread = self._research_readiness_thread
                readiness_error = self._research_readiness_error
            if cached is None:
                readiness_refreshing = (
                    self._schedule_research_readiness_refresh()
                )
                training = self._pending_research_readiness()
            else:
                readiness_refreshing = bool(
                    readiness_thread is not None
                    and readiness_thread.is_alive()
                )
                training = deepcopy(cached["training"])
            result["readiness_cache"] = {
                "refreshing": readiness_refreshing,
                "created_unix_ms": (
                    cached.get("created_unix_ms")
                    if cached is not None
                    else None
                ),
                "error": readiness_error,
            }
        # SQLite can wait on an offline writer. Never perform that wait while
        # holding the lock used to publish audio/DMX state.
        running_jobs = [
            job
            for job in self.memory.list_analysis_jobs(limit=100_000)
            if job["status"] == "running"
            and not str(job.get("worker_id") or "").startswith(
                "lumen-link:"
            )
        ]
        with self._lock:
            worker = self._research_worker_thread
            local_worker_running = bool(
                worker is not None and worker.is_alive()
            )
            preparation = self._training_prepare_thread
            preparation_running = bool(
                preparation is not None and preparation.is_alive()
            )
            external_worker_running = bool(
                running_jobs and not local_worker_running
            )
            worker_running = local_worker_running or external_worker_running
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
                local_worker_running and self._research_cancel.is_set()
            )
            result["worker"] = {
                "running": worker_running,
                "externally_managed": external_worker_running,
                "cancel_supported": local_worker_running,
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
            may_reload_model = bool(
                not worker_running
                and self.engine_phase in {"ready", "fault"}
            )
        # Model file loading may decompress/allocate arrays and is not part of
        # the audio-state critical section.
        if may_reload_model and self._student_artifact_changed():
            self._load_student_model()
        with self._lock:
            runtime_state = (
                "error"
                if self._student_model_error
                else self._student_model_state
            )
            model = training.setdefault("model", {})
            model["artifact_present"] = bool(
                model.get("active_artifact_exists")
                or model.get("active")
            )
            model["active"] = self._student_model is not None
            model["runtime_state"] = runtime_state
            model["runtime_error"] = self._student_model_error
            model["runtime_notice"] = self._student_model_notice
            model["runtime_gate_reasons"] = list(
                self._student_model_gate_reasons
            )
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
        """Prepare captures, queue both axis teachers, and resume them."""
        self._recover_abandoned_research_jobs()
        external_research_running = any(
            job["status"] == "running"
            and not str(job.get("worker_id") or "").startswith(
                "lumen-link:"
            )
            for job in self.memory.list_analysis_jobs(limit=100_000)
        )
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
            if external_research_running:
                raise RuntimeError(
                    "offline research is already running in another "
                    "Lumen process"
                )
            preparation = self._training_prepare_thread
            if preparation is not None and preparation.is_alive():
                raise RuntimeError(
                    "the last audio capture is still being prepared; wait for "
                    "Audio Laboratory to report that preparation is complete"
                )
        # Recovery must precede export preparation and the queued-job count;
        # otherwise a crash-stranded `running` row can make Analyze appear to
        # have no resumable work.
        export = self._prepare_unindexed_research_captures()
        routed_to_link = self.lumen_link.route_queued_jobs()
        analysis_jobs = self.memory.list_analysis_jobs(limit=100_000)
        queued = sum(
            job["status"] == "queued"
            and job["job_type"] in {EDMFORMER_JOB, SONGFORMER_JOB}
            for job in analysis_jobs
        )
        if not queued:
            ineligible = int(export.get("recordings_ineligible") or 0)
            partial = int(export.get("recordings_partial") or 0)
            unknown = int(export.get("recordings_unknown") or 0)
            if ineligible:
                message = (
                    f"Found {ineligible} captured recording(s), but none were "
                    "complete enough for whole-song structure analysis"
                    f" ({partial} partial, {unknown} unidentified). The audio "
                    "is retained; record complete songs and analyze again."
                )
            else:
                message = (
                    "No new or retryable full-song recordings are queued for "
                    "structure analysis."
                )
            return {
                "export": export,
                "research": self.research_status(
                    wait_for_readiness=False
                ),
                "started": False,
                "message": message,
            }
        local_job_types = [
            job_type
            for job_type in (EDMFORMER_JOB, SONGFORMER_JOB)
            if not (
                job_type == EDMFORMER_JOB
                and self.lumen_link.ready_for_offload()
            )
            if any(
                job["status"] == "queued"
                and job["job_type"] == job_type
                and str(
                    (job.get("payload") or {}).get("execution_target")
                    or "automatic"
                )
                != "threadripper"
                for job in analysis_jobs
            )
        ]
        research = (
            self.start_research_worker({"job_types": local_job_types})
            if local_job_types
            else self.research_status(wait_for_readiness=False)
        )
        return {
            "export": export,
            "research": research,
            "started": True,
            "routed_to_threadripper": routed_to_link,
        }

    def _prepare_unindexed_research_captures(self) -> dict[str, Any]:
        """Incrementally index captures that have never reached research."""

        if not self._training_export_lock.acquire(blocking=False):
            raise RuntimeError("a training dataset build is already in progress")
        try:
            prepared_ids = self.memory.research_prepared_session_ids()
            sessions = [
                session
                for session in self.memory.training_sessions()
                if str(session.get("status")) in {"complete", "quota"}
                and int(session.get("frames_written") or 0) > 0
                and str(session["id"]) not in prepared_ids
            ]
            coordinator = ResearchJobCoordinator(
                self.memory,
                training_root=self.training_root,
                research_root=self.training_root / "research",
            )
            exports: list[str] = []
            recordings = jobs_queued = skipped = 0
            ineligible = partial = unknown = 0
            for session in sessions:
                session_id = str(session["id"])
                result = export_research_session_index(
                    self.memory, self.training_root, session_id
                )
                research = coordinator.prepare_export(
                    result["path"], queue_songformer=True
                )
                self.memory.refresh_operator_structure_consensus()
                self.memory.mark_research_session_prepared(
                    session_id, result["path"]
                )
                exports.append(str(result["path"]))
                recordings += int(research["recordings"])
                jobs_queued += int(research["jobs_queued"])
                skipped += len(research["teachers_skipped"])
                ineligible += int(research.get("recordings_ineligible") or 0)
                partial += int(research.get("recordings_partial") or 0)
                unknown += int(research.get("recordings_unknown") or 0)
            replacements = coordinator.requeue_obsolete_edmformer_jobs()
            songformer_replacements = (
                coordinator.requeue_obsolete_songformer_jobs()
            )
            jobs_queued += int(replacements["jobs_queued"])
            jobs_queued += int(songformer_replacements["jobs_queued"])
            with self._lock:
                if exports:
                    self._last_training_export = exports[-1]
                self._training_history = self.memory.training_summary()
                self._status_sequence += 1
            return {
                "mode": "incremental_research_preparation",
                "sessions_prepared": len(sessions),
                "recordings": recordings,
                "jobs_queued": jobs_queued,
                "teachers_skipped": skipped,
                "obsolete_teacher_jobs_requeued": int(
                    replacements["jobs_queued"]
                ),
                "obsolete_teacher_audio_unavailable": list(
                    replacements["unavailable"]
                ) + list(songformer_replacements["unavailable"]),
                "recordings_ineligible": ineligible,
                "recordings_partial": partial,
                "recordings_unknown": unknown,
                "paths": exports,
            }
        finally:
            self._training_export_lock.release()

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
        self._recover_abandoned_research_jobs()
        analysis_jobs = self.memory.list_analysis_jobs(limit=100_000)
        external_research_running = any(
            job["status"] == "running"
            and not str(job.get("worker_id") or "").startswith(
                "lumen-link:"
            )
            for job in analysis_jobs
        )
        available = sum(
            job["status"] == "queued" and job["job_type"] in job_types
            for job in analysis_jobs
        )
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
            if external_research_running:
                raise RuntimeError(
                    "offline research is already running in another "
                    "Lumen process"
                )
            preparation = self._training_prepare_thread
            if preparation is not None and preparation.is_alive():
                raise RuntimeError(
                    "the last audio capture is still being prepared; wait for "
                    "Audio Laboratory to report that preparation is complete"
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
        return self.research_status(wait_for_readiness=False)

    def cancel_research_worker(self) -> dict[str, Any]:
        worker = self._research_worker_thread
        if worker is None or not worker.is_alive():
            raise RuntimeError("no offline research batch is running")
        self._research_cancel.set()
        with self._lock:
            self._add_event("memory", "Pausing offline analysis after cancellation")
            self._status_sequence += 1
        return self.research_status(wait_for_readiness=False)

    def train_structure_student(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        epochs = int(payload.get("epochs", 30))
        if not 1 <= epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        self._recover_abandoned_research_jobs()
        external_research_running = any(
            job["status"] == "running"
            and not str(job.get("worker_id") or "").startswith(
                "lumen-link:"
            )
            for job in self.memory.list_analysis_jobs(limit=100_000)
        )
        with self._lock:
            if self.engine_phase not in {"ready", "fault"}:
                raise RuntimeError(
                    "stop Monitor or Live before training the CPU student"
                )
            worker = self._research_worker_thread
            if worker is not None and worker.is_alive():
                raise RuntimeError("an offline research job is already running")
            if external_research_running:
                raise RuntimeError(
                    "offline research is already running in another "
                    "Lumen process"
                )
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
        # Recompute the expensive readiness evidence after, rather than inside,
        # the worker completion path.  The UI keeps receiving its last verified
        # summary while this daemon refreshes the cache.
        self._schedule_research_readiness_refresh()

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
        history = dict(self._training_history)
        historical_bytes = int(history.get("bytes", 0))
        current_bytes = int(current.get("bytes_written", 0)) if current else 0
        return {
            "enabled": self.training_capture_enabled,
            "path": str(self.training_root),
            "max_bytes": round(self.training_max_gb * 1024**3),
            "disk_free_bytes": self._training_disk_free_bytes,
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

    def _read_training_disk_free(self) -> int | None:
        try:
            return int(shutil.disk_usage(self.training_root.parent).free)
        except OSError:
            return None

    def connect_spotify(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_id = str(
            payload.get("client_id") or self.spotify_client_id
        ).strip()
        if not client_id:
            raise ValueError("paste the Spotify developer client ID first")
        with self._configuration_lock:
            with self._lock:
                if (
                    self._spotify_login_thread is not None
                    and self._spotify_login_thread.is_alive()
                ):
                    raise RuntimeError(
                        "Spotify connection is already in progress"
                    )
                self.spotify_client_id = client_id[:256]
                self._settings["spotify_client_id"] = self.spotify_client_id
                settings = dict(self._settings)
                self._spotify_login_phase = "connecting"
                self._spotify_error = None
            self._save_settings(settings)
            with self._lock:
                self._spotify_login_thread = threading.Thread(
                    target=self._complete_spotify_login,
                    name="lumen-spotify-login",
                    daemon=True,
                )
                self._spotify_login_thread.start()
                self._add_event(
                    "media",
                    "Opening Spotify authorization in the desktop browser",
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
        # Several phones may open the remote together.  Serialize Spotify's
        # network calls and share the resulting console payload rather than
        # issuing the same profile/device/library requests per browser.
        with self._spotify_console_lock:
            return self._spotify_console_locked(query, playlist_id)

    def _spotify_console_locked(
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
        cached_at = self._spotify_console_cached_at.get(cache_key, 0.0)
        cache_age = max(0.0, time.time() - cached_at)
        if cache_key in self._spotify_console_cache and cache_age < 1.0:
            return deepcopy(self._spotify_console_cache[cache_key])
        if time.time() < self._spotify_rate_limited_until and cache_key in self._spotify_console_cache:
            cached = deepcopy(self._spotify_console_cache[cache_key])
            cached["stale"] = True
            cached["message"] = "Spotify is rate limited; showing the last known player state."
            return cached
        try:
            oauth = SpotifyOAuthPKCE(
                client_id=self.spotify_client_id,
                cache=SpotifyTokenCache(DEFAULT_SPOTIFY_TOKEN),
            )
            client = SpotifyWebAPI(oauth.valid_token)
            if cache_key in self._spotify_console_cache and cache_age < 60.0:
                # Playback is the only console data that normally changes from
                # second to second.  Reuse profile, device and library data so
                # a responsive player does not repeatedly enumerate all of
                # Spotify on this dedicated machine.
                playback_payload = client.playback()
                client.last_playback_payload = playback_payload
                console = deepcopy(self._spotify_console_cache[cache_key])
                console["playback"] = spotify_playback_summary(
                    playback_payload
                )
                console["observed_at_unix_ms"] = round(time.time() * 1000)
                console.pop("stale", None)
            else:
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
            self._spotify_console_cache[cache_key] = deepcopy(console)
            self._spotify_console_cached_at[cache_key] = time.time()
            return console
        except Exception as error:
            if "rate limited" in str(error).lower() and cache_key in self._spotify_console_cache:
                self._spotify_rate_limited_until = time.time() + 15.0
                cached = deepcopy(self._spotify_console_cache[cache_key])
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
        with self._spotify_console_lock:
            self._spotify_console_cache.clear()
            self._spotify_console_cached_at.clear()
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
                previous_media = self.media
                expected_position_ms = self._media_position_ms()
                count_play = key != self._last_media_key
                same_recording = bool(
                    previous_media is not None
                    and media.provider_item_id is not None
                    and previous_media.provider_item_id
                    == media.provider_item_id
                    and previous_media.provider == media.provider
                )
                position_delta_ms = (
                    media.observed_position_ms - expected_position_ms
                    if same_recording
                    and media.observed_position_ms is not None
                    and expected_position_ms is not None
                    else 0
                )
                playback_seek = bool(
                    same_recording
                    and abs(position_delta_ms) >= 2_500
                )
            song_id = self.memory.remember_media(
                media, count_play=count_play
            )
            with self._lock:
                self.media = media
                self.song_id = song_id
                runtime = self._runtime
                section = self.observation.section
                if count_play or playback_seek:
                    self._analysis_generation += 1
                    self._cached_structure_prediction = None
                    self._cached_structure_key = None
                    self._cached_structure_checked_at = 0.0
                    self._recalled_choreography_checked_at = 0.0
                    self._recalled_choreography_ids = ()
                    self._prepared_recalled_choreography = ()
                    self._prepared_recalled_ids = ()
                    self._memory_context_last_poll = 0.0
                    self._teaching_snapshot_cache = None
                    self._teaching_snapshot_checked_at = 0.0
                if count_play:
                    self._last_media_key = key
                    self._add_event(
                        "media", f"Now playing {media.display_name}"
                    )
                elif playback_seek:
                    self._add_event(
                        "media",
                        (
                            "Playback position changed; reset causal audio "
                            "and choreography context"
                        ),
                    )
            if runtime is not None:
                if count_play or playback_seek:
                    runtime.set_recalled_choreography(())
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
        lifetime = str(payload.get("lifetime", "cue")).strip().casefold()
        if lifetime not in {"cue", "song", "artist", "global"}:
            raise ValueError(
                "feedback lifetime must be cue, song, artist, or global"
            )
        raw_fixture_id = payload.get("fixture_id")
        fixture_id = str(raw_fixture_id).strip() if raw_fixture_id is not None else None
        group_id = str(payload.get("group_id", "")).strip() or None
        participant_id = str(payload.get("participant_id", "")).strip() or None
        participant_name = (
            str(payload.get("participant_name", "")).strip() or None
        )
        client_event_id = (
            str(payload.get("client_event_id", "")).strip() or None
        )
        if participant_id is not None and len(participant_id) > 96:
            raise ValueError("participant_id is too long")
        if participant_name is not None and len(participant_name) > 32:
            raise ValueError("participant_name is too long")
        if client_event_id is not None and len(client_event_id) > 128:
            raise ValueError("client_event_id is too long")
        if scope == "fixture" and not fixture_id:
            raise ValueError("fixture feedback requires a fixture_id")
        if scope == "group" and group_id not in {"movers", "center"}:
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
                listening_session_id = self._session_id
            feedback_fixture = group_id if scope == "group" else fixture_id
            target_lanes = self._feedback_target_lanes(
                scope, feedback_fixture
            )
            choreography_snapshot = (
                runtime.choreography_snapshot()
                if runtime is not None
                else {}
            )
            snapshot_lanes = choreography_snapshot.get("lanes", {})
            lane_context = {
                "version": 1,
                "lifetime": lifetime,
                "lanes": {
                    lane: {
                        "section": str(
                            context_observation.section or "unknown"
                        ),
                        "active_sequence_id": str(
                            (snapshot_lanes.get(lane) or {}).get(
                                "active_sequence_id"
                            )
                            or "unknown"
                        ),
                        "boundary_id": str(
                            (snapshot_lanes.get(lane) or {}).get(
                                "active_boundary_id"
                            )
                            or "unknown"
                        ),
                        "routine": str(
                            (
                                (snapshot_lanes.get(lane) or {}).get(
                                    "active_step"
                                )
                                or {}
                            ).get("routine")
                            or "unknown"
                        ),
                    }
                    for lane in target_lanes
                },
            }
            # A listener burst is a sliding five-second consensus window, not
            # a wall-clock bucket. Otherwise eight phones tapping together at
            # xx:x4.999 could be split merely because midnight's modulo clock
            # advanced while the serialized database writes were completing.
            consensus_now_ms = round(time.time() * 1000)
            consensus_start_ms = consensus_now_ms
            prior_feedback = self.memory.list_feedback(song_id)
            prior_starts = [
                self._feedback_consensus_start_ms(row)
                for row in prior_feedback
                if row.get("label") == label
                and row.get("scope") == scope
                and row.get("fixture_id") == feedback_fixture
                and str(row.get("listening_session_id") or "")
                == str(listening_session_id)
                and isinstance(row.get("lane_context"), dict)
                and str(
                    row["lane_context"].get("lifetime") or "cue"
                ) == lifetime
                and row["lane_context"].get("lanes")
                == lane_context["lanes"]
                and self._feedback_consensus_start_ms(row)
                <= consensus_now_ms
                < self._feedback_consensus_start_ms(row) + 5_000
            ]
            if prior_starts:
                consensus_start_ms = max(prior_starts)
            lane_context["consensus_started_unix_ms"] = consensus_start_ms
            performed_routines = {
                item["routine"]
                for item in lane_context["lanes"].values()
            }
            persisted_routine = (
                next(iter(performed_routines))
                if len(performed_routines) == 1
                else "parallel"
            )
            feedback_event = self.memory.add_feedback_event(
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
                    routine=persisted_routine,
                    capture_session_id=capture_session_id,
                    audio_frame_index=capture_audio_frame,
                ),
                participant_id=participant_id,
                participant_name=participant_name,
                client_event_id=client_event_id,
                listening_session_id=listening_session_id,
                lane_context=lane_context,
            )
            feedback_id = int(feedback_event["id"])
            feedback_created = bool(feedback_event["created"])
            # Keep a compact semantic routine alongside the raw feedback. It
            # is deliberately made from moments and decisions, not DMX bytes,
            # so it can be resolved against a changed rig later.
            song_feedback = self.memory.list_feedback(song_id)
            current_feedback = next(
                (
                    row for row in song_feedback
                    if int(row.get("id") or 0) == feedback_id
                ),
                None,
            )
            created_ms = int(
                (current_feedback or {}).get("created_unix_ms")
                or time.time() * 1000
            )
            persisted_feedback = current_feedback or {}
            batch_label = str(persisted_feedback.get("label") or label)
            batch_scope = str(persisted_feedback.get("scope") or scope)
            batch_fixture = persisted_feedback.get(
                "fixture_id", feedback_fixture
            )
            batch_session = str(
                persisted_feedback.get("listening_session_id")
                or listening_session_id
            )
            target_lanes = self._feedback_target_lanes(
                batch_scope, batch_fixture
            )
            batch_start_ms = self._feedback_consensus_start_ms(
                persisted_feedback
            )
            model_event_id = self._feedback_batch_group_event_id(
                song_id=song_id,
                listening_session_id=batch_session,
                created_unix_ms=batch_start_ms,
                label=batch_label,
                scope=batch_scope,
                fixture_id=batch_fixture,
                lifetime=lifetime,
            )
            model_event_ids: dict[str, str] = {}
            occurrences_by_lane: dict[str, int] = {}
            urgency_by_lane: dict[str, float] = {}
            participants_by_lane: dict[str, int] = {}
            for lane in target_lanes:
                performed_context = self._feedback_lane_context(
                    persisted_feedback, lane
                )
                if performed_context is None:
                    continue
                matching_recent = [
                    row for row in song_feedback
                    if row.get("label") == batch_label
                    and row.get("scope") == batch_scope
                    and row.get("fixture_id") == batch_fixture
                    and str(row.get("listening_session_id") or "")
                    == batch_session
                    and str(
                        (row.get("lane_context") or {}).get("lifetime")
                        or "cue"
                    ) == lifetime
                    and self._feedback_lane_context(row, lane)
                    == performed_context
                    and self._feedback_consensus_start_ms(row)
                    == batch_start_ms
                ]
                occurrences_by_lane[lane] = max(1, len(matching_recent))
                participants = {
                    str(
                        row.get("participant_id")
                        or f"legacy:{row.get('id')}"
                    )
                    for row in matching_recent
                }
                distinct = max(1, len(participants))
                participants_by_lane[lane] = distinct
                repeated_taps = max(
                    0, occurrences_by_lane[lane] - distinct
                )
                urgency_by_lane[lane] = clamp(
                    0.45
                    + 0.14 * (distinct - 1)
                    + 0.06 * repeated_taps,
                    0.0,
                    1.0,
                )
                model_event_ids[lane] = self._feedback_batch_event_id(
                    song_id=song_id,
                    listening_session_id=batch_session,
                    created_unix_ms=batch_start_ms,
                    label=batch_label,
                    scope=batch_scope,
                    fixture_id=batch_fixture,
                    lane=lane,
                    section=performed_context["section"],
                    routine=performed_context["routine"],
                    active_sequence_id=performed_context[
                        "active_sequence_id"
                    ],
                    boundary_id=performed_context["boundary_id"],
                    lifetime=lifetime,
                )
            occurrences = max(occurrences_by_lane.values(), default=1)
            distinct_participants = max(
                participants_by_lane.values(), default=1
            )
            urgency = max(urgency_by_lane.values(), default=0.45)
            learned_sequence: dict[str, Any] | None = None
            if (
                runtime is not None
                and abs(value) > 0.0
                and label != "operator_note"
                and feedback_created
            ):
                learned_sequence = runtime.learn_choreography_feedback(
                    label=label,
                    value=clamp(value, -1.0, 1.0),
                    urgency=urgency,
                    # One replaceable event represents the whole five-second
                    # consensus window. Repeated presses raise urgency and
                    # mass without creating 1+2+3 duplicate examples.
                    occurrences=max(1, occurrences),
                    scope=scope,
                    fixture_id=feedback_fixture,
                    preferred_routine=None,
                    created_unix_ms=created_ms,
                    event_ids_by_lane=model_event_ids,
                    occurrences_by_lane=occurrences_by_lane,
                    urgency_by_lane=urgency_by_lane,
                    lifetime=lifetime,
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
            refresh_lanes = self._feedback_target_lanes(
                scope, feedback_fixture
            )
            refresh_live = self.engine_mode == "live"
            if refresh_live:
                self._queue_feedback_bias_refresh(refresh_lanes)
            else:
                self._rebuild_feedback_biases()
            with self._lock:
                current_runtime = self._runtime
                biases = deepcopy(self._feedback_biases)
                if capture_audio_frame is not None and feedback_created:
                    self._training_linked_feedback += 1
                if feedback_created:
                    self._add_event(
                        "feedback",
                        f"Recorded feedback: {label.replace('_', ' ')}",
                    )
                self._status_sequence += 1
            if current_runtime is not None and not refresh_live:
                current_runtime.replace_feedback(
                    biases,
                    replan_lanes=refresh_lanes,
                )
        return {
            "feedback_id": feedback_id,
            "created": feedback_created,
            "song_id": song_id,
            "feedback_occurrences": max(1, occurrences),
            "participant_agreement": distinct_participants,
            "urgency": urgency,
            "participant_name": participant_name,
            "model_event_id": model_event_id,
            "model_event_ids": model_event_ids,
            "sequence_learning": learned_sequence,
        }

    def add_training_annotation(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip().lower()
        allowed = {
            "musical_context": {
                *CANONICAL_TECHNO_SECTIONS,
                *(role.value for role in ContentRole if role.value not in {
                    "unknown", "silence",
                }),
                *(event.value for event in TransitionEvent),
            },
            "preferred_action": {
                "keep_current", "breathe", "fan_sweep", "figure_eight",
                "opposing_chase", "beat_nod", "counter_rotate",
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
        # Musical structure describes the song, never a fixture. Older mobile
        # clients reused the lighting target selector and accidentally stored
        # group-scoped structure calls. Normalize at the authority boundary so
        # those taps still teach the intended song-wide axis.
        if kind == "musical_context":
            scope = "overall"
            fixture_id = None
            group_id = None
        participant_id = str(payload.get("participant_id", "")).strip() or None
        participant_name = (
            str(payload.get("participant_name", "")).strip() or None
        )
        client_event_id = (
            str(payload.get("client_event_id", "")).strip() or None
        )
        if scope == "group" and group_id not in {"movers", "center"}:
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
            annotation_event = self.memory.add_training_annotation_event(
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
                participant_id=participant_id,
                participant_name=participant_name,
                client_event_id=client_event_id,
                listening_session_id=self._session_id,
                context={
                    "observation": asdict(observation_snapshot),
                    "decision": decision,
                    "controls": controls_snapshot,
                    "participant": {
                        "id": participant_id,
                        "name": participant_name,
                        "client_event_id": client_event_id,
                        "listening_session_id": self._session_id,
                    },
                },
            )
            annotation_id = int(annotation_event["id"])
            annotation_created = bool(annotation_event["created"])
            sequence_learning: dict[str, Any] | None = None
            if (
                kind == "preferred_action"
                and runtime is not None
                and annotation_created
            ):
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
                        lifetime="cue",
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
                if capture_audio_frame is not None and annotation_created:
                    self._training_annotations += 1
                if annotation_created:
                    self._add_event(
                        "feedback",
                        f"Training label: {label.replace('_', ' ')}",
                    )
                self._status_sequence += 1
        return {
            "annotation_id": annotation_id,
            "created": annotation_created,
            "song_id": song_id,
            "kind": kind,
            "label": label,
            "linked_to_audio": capture_audio_frame is not None,
            "sequence_learning": sequence_learning,
            "structure_consensus": (
                "pending_offline_rebuild"
                if kind == "musical_context" and annotation_created
                else None
            ),
        }

    def save_choreography_proposal(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Save an editable semantic routine and place it on this song."""

        scope = canonical_motion_scope(payload.get("scope", "movers"))
        name = str(payload.get("name", "")).strip()[:80] or (
            f"{scope.title()} taught routine"
        )
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("at least one choreography step is required")
        steps: list[dict[str, Any]] = []
        next_start = 0.0
        for raw in raw_steps[:24]:
            if not isinstance(raw, dict):
                raise ValueError("each choreography step must be an object")
            routine = str(raw.get("routine", "")).strip().lower()
            if routine not in REHEARSAL_ROUTINE_IDS:
                raise ValueError("unknown choreography routine")
            start_beat = float(raw.get("start_beat", next_start))
            duration_beats = clamp(
                float(raw.get("duration_beats", 8.0)), 0.25, 128.0
            )
            if start_beat < 0:
                raise ValueError("step start beat must be non-negative")
            step_scope = canonical_motion_scope(
                raw.get("fixture_scope", scope)
            )
            if scope != "overall" and step_scope != scope:
                raise ValueError("a group sequence cannot control another group")
            palette = str(raw.get("palette", "")).strip() or None
            if palette is not None and palette not in PALETTE_FAMILIES:
                raise ValueError("unknown choreography palette")
            strobe_rate = clamp(
                float(raw.get("strobe_rate", raw.get("strobe", 0.0))),
                0.0,
                1.0,
            )
            strobe_enabled = bool(
                raw.get("strobe_enabled", strobe_rate > 0.0)
            )
            entry = str(raw.get("entry_behavior", "phrase_boundary")).strip()
            exit_behavior = str(raw.get("exit_behavior", "resolve")).strip()
            if entry not in {"phrase_boundary", "soft", "accent"}:
                raise ValueError("unknown entry behavior")
            if exit_behavior not in {"resolve", "hold", "blackout", "crossfade"}:
                raise ValueError("unknown exit behavior")
            steps.append({
                "start_beat": start_beat,
                "duration_beats": duration_beats,
                "fixture_scope": step_scope,
                "routine": routine,
                "intensity": clamp(
                    float(raw.get("intensity", 1.0)), 0.0, 1.0
                ),
                "palette": palette,
                "strobe": {
                    "enabled": strobe_enabled,
                    "rate": strobe_rate,
                },
                "entry_behavior": entry,
                "exit_behavior": exit_behavior,
                "parameters": {
                    "beat_sync": clamp(
                        float(raw.get("beat_sync", 1.0)), 0.0, 1.0
                    ),
                    "motion_speed": clamp(
                        float(raw.get("motion_speed", 0.5)), 0.0, 1.0
                    ),
                    "travel_size": clamp(
                        float(raw.get("travel_size", 1.0)), 0.0, 1.0
                    ),
                    "activity_density": clamp(
                        float(raw.get("activity_density", 1.0)), 0.0, 1.0
                    ),
                    "brightness": (
                        None
                        if raw.get("brightness") is None
                        else clamp(float(raw["brightness"]), 0.0, 1.0)
                    ),
                    "cue_timing": clamp(
                        float(raw.get("cue_timing", 1.0)), 0.0, 1.0
                    ),
                },
            })
            next_start = start_beat + duration_beats
        steps.sort(key=lambda item: float(item["start_beat"]))
        participant_id = str(payload.get("participant_id", "")).strip() or None
        participant_name = str(payload.get("participant_name", "")).strip() or None
        client_event_id = str(payload.get("client_event_id", "")).strip() or None
        if participant_id is not None and len(participant_id) > 96:
            raise ValueError("participant_id is too long")
        if participant_name is not None and len(participant_name) > 32:
            raise ValueError("participant_name is too long")
        if client_event_id is not None and len(client_event_id) > 128:
            raise ValueError("client_event_id is too long")
        requested_sequence_id = (
            str(payload.get("sequence_id", "")).strip() or None
        )
        if requested_sequence_id is not None:
            existing_sequence = self.memory.choreography_sequence(
                requested_sequence_id
            )
            if (
                existing_sequence is not None
                and existing_sequence.get("participant_id")
                and existing_sequence.get("participant_id") != participant_id
            ):
                raise ValueError("sequence belongs to another listener")
        with self._lock:
            song_id = self.song_id
            media = self.media
            section = self.observation.section
            position_ms = self._media_position_ms()
            cached = deepcopy(self._cached_structure_prediction)
        if song_id is None or media is None:
            raise ValueError(
                "identify a Spotify track before placing a song sequence"
            )
        if position_ms is None:
            raise ValueError(
                "the current track position is unavailable; refresh Spotify and try again"
            )
        requested_section = (
            str(payload.get("section_label", "")).strip() or None
        )
        if (
            requested_section is not None
            and requested_section not in CANONICAL_TECHNO_SECTIONS
        ):
            raise ValueError(
                "song sequence section must use a canonical techno state"
            )
        recording = (
            self.memory.resolve_recording_version(
                provider=media.provider,
                provider_item_id=media.provider_item_id,
                duration_ms=media.duration_ms,
            )
            if media is not None else None
        )
        axes = (cached or {}).get("axes") or {}
        timeline_id = next(
            (
                value.get("timeline_id")
                for value in axes.values()
                if isinstance(value, dict) and value.get("timeline_id")
            ),
            None,
        )
        sequence_id = self.memory.save_choreography_sequence(
            sequence_id=requested_sequence_id,
            song_id=song_id,
            recording_id=(recording or {}).get("id"),
            timeline_id=timeline_id,
            name=name,
            fixture_scope=scope,
            participant_id=participant_id,
            participant_name=participant_name,
            client_event_id=client_event_id,
            source="operator_sequence_editor",
            confidence=1.0,
            context={
                "section": section,
                "position_ms": position_ms,
                "cached_structure": cached,
            },
            steps=steps,
        )
        placement_id = None
        if payload.get("place", True):
            placement_event_id = (
                None if client_event_id is None
                else f"{client_event_id}:placement"
            )
            requested_placement_id = (
                str(payload.get("placement_id", "")).strip() or None
            )
            if requested_placement_id is not None:
                existing_placement = self.memory.choreography_placement(
                    requested_placement_id
                )
                if (
                    existing_placement is not None
                    and existing_placement.get("participant_id")
                    and existing_placement.get("participant_id")
                    != participant_id
                ):
                    raise ValueError("placement belongs to another listener")
            placement_id = self.memory.save_choreography_placement(
                placement_id=requested_placement_id,
                sequence_id=sequence_id,
                song_id=song_id,
                recording_id=(recording or {}).get("id"),
                timeline_id=timeline_id,
                fixture_scope=scope,
                start_ms=position_ms,
                duration_beats=max(
                    float(step["start_beat"])
                    + float(step["duration_beats"])
                    for step in steps
                ),
                section_label=(
                    requested_section or section
                ),
                participant_id=participant_id,
                participant_name=participant_name,
                client_event_id=placement_event_id,
                source="operator_sequence_editor",
                context={"name": name, "steps": len(steps)},
            )
        self._recalled_choreography_checked_at = 0.0
        self._teaching_snapshot_checked_at = 0.0
        with self._lock:
            self._add_event(
                "memory", f"Taught {scope} sequence: {name}"
            )
            self._status_sequence += 1
        return {
            "sequence_id": sequence_id,
            "placement_id": placement_id,
            "song_id": song_id,
            "recording_id": (recording or {}).get("id"),
            "steps": len(steps),
            "scope": scope,
        }

    def edit_choreography_history(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip().lower()
        action = str(payload.get("action", "")).strip().lower()
        item_id = str(payload.get("id", "")).strip()
        if not item_id:
            raise ValueError("choreography id is required")
        participant_id = str(payload.get("participant_id", "")).strip() or None
        existing = (
            self.memory.choreography_sequence(item_id, include_deleted=True)
            if kind == "sequence"
            else self.memory.choreography_placement(
                item_id, include_deleted=True
            )
            if kind == "placement"
            else None
        )
        if (
            participant_id is not None
            and existing is not None
            and existing.get("participant_id")
            and existing.get("participant_id") != participant_id
        ):
            raise ValueError("choreography item belongs to another listener")
        operations = {
            ("sequence", "delete"): self.memory.delete_choreography_sequence,
            ("sequence", "undo"): self.memory.undo_choreography_sequence,
            ("placement", "delete"): self.memory.delete_choreography_placement,
            ("placement", "undo"): self.memory.undo_choreography_placement,
        }
        operation = operations.get((kind, action))
        if operation is None:
            raise ValueError("unknown choreography history operation")
        changed = operation(item_id)
        if not changed:
            raise ValueError("choreography item or revision was not found")
        self._recalled_choreography_checked_at = 0.0
        self._teaching_snapshot_checked_at = 0.0
        with self._lock:
            self._status_sequence += 1
            self._add_event(
                "memory", f"{action.title()} {kind} {item_id[-8:]}"
            )
        return {"changed": True, "kind": kind, "action": action, "id": item_id}

    def delete_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback_id = int(payload.get("feedback_id", 0))
        if feedback_id < 1:
            raise ValueError("feedback_id is required")
        participant_id = str(payload.get("participant_id", "")).strip() or None
        with self._feedback_lock:
            existing = next(
                (
                    row
                    for row in self.memory.all_feedback()
                    if int(row.get("id") or 0) == feedback_id
                ),
                None,
            )
            if (
                participant_id is not None
                and existing is not None
                and existing.get("participant_id") != participant_id
            ):
                raise ValueError("feedback entry belongs to another listener")
            deleted = self.memory.delete_feedback(feedback_id)
            if not deleted:
                raise ValueError("feedback entry was not found")
            remaining_feedback = self.memory.all_feedback()
            sequence_update_removed = False
            if existing is not None:
                batch_start = self._feedback_consensus_start_ms(existing)
                target_lanes = self._feedback_target_lanes(
                    str(existing.get("scope") or "overall"),
                    existing.get("fixture_id"),
                )
                has_lane_context = isinstance(
                    existing.get("lane_context"), dict
                )
                if has_lane_context:
                    lifetime = str(
                        (existing.get("lane_context") or {}).get(
                            "lifetime"
                        )
                        or "cue"
                    )
                    for lane in target_lanes:
                        performed_context = self._feedback_lane_context(
                            existing, lane
                        )
                        if performed_context is None:
                            continue
                        model_event_id = self._feedback_batch_event_id(
                            song_id=int(existing.get("song_id") or 0),
                            listening_session_id=str(
                                existing.get("listening_session_id") or ""
                            ),
                            created_unix_ms=batch_start,
                            label=str(existing.get("label") or ""),
                            scope=str(
                                existing.get("scope") or "overall"
                            ),
                            fixture_id=existing.get("fixture_id"),
                            lane=lane,
                            section=performed_context["section"],
                            routine=performed_context["routine"],
                            active_sequence_id=performed_context[
                                "active_sequence_id"
                            ],
                            boundary_id=performed_context["boundary_id"],
                            lifetime=lifetime,
                        )
                        batch_rows = [
                            row for row in remaining_feedback
                            if int(row.get("song_id") or 0)
                            == int(existing.get("song_id") or 0)
                            and row.get("listening_session_id")
                            == existing.get("listening_session_id")
                            and row.get("label") == existing.get("label")
                            and row.get("scope") == existing.get("scope")
                            and row.get("fixture_id")
                            == existing.get("fixture_id")
                            and str(
                                (row.get("lane_context") or {}).get(
                                    "lifetime"
                                )
                                or "cue"
                            ) == lifetime
                            and self._feedback_lane_context(row, lane)
                            == performed_context
                            and self._feedback_consensus_start_ms(row)
                            == batch_start
                        ]
                        sequence_update_removed = (
                            self._revise_or_forget_feedback_batch(
                                model_event_id, batch_rows
                            )
                            or sequence_update_removed
                        )
                else:
                    # Version-1 consensus rows were keyed only by the public
                    # compatibility routine. Continue to undo those rows.
                    model_event_id = self._feedback_batch_event_id(
                        song_id=int(existing.get("song_id") or 0),
                        listening_session_id=str(
                            existing.get("listening_session_id") or ""
                        ),
                        created_unix_ms=batch_start,
                        label=str(existing.get("label") or ""),
                        scope=str(existing.get("scope") or "overall"),
                        fixture_id=existing.get("fixture_id"),
                        section=str(
                            existing.get("section") or "unknown"
                        ),
                        routine=str(
                            existing.get("routine") or "unknown"
                        ),
                    )
                    batch_rows = [
                        row for row in remaining_feedback
                        if int(row.get("song_id") or 0)
                        == int(existing.get("song_id") or 0)
                        and row.get("listening_session_id")
                        == existing.get("listening_session_id")
                        and row.get("label") == existing.get("label")
                        and row.get("scope") == existing.get("scope")
                        and row.get("fixture_id")
                        == existing.get("fixture_id")
                        and row.get("section") == existing.get("section")
                        and row.get("routine") == existing.get("routine")
                        and not isinstance(row.get("lane_context"), dict)
                        and self._feedback_consensus_start_ms(row)
                        == batch_start
                    ]
                    sequence_update_removed = (
                        self._revise_or_forget_feedback_batch(
                            model_event_id, batch_rows
                        )
                        or sequence_update_removed
                    )
            # The original released model used one event per database row.
            # Always try this exact fallback; it is a no-op for current rows.
            sequence_update_removed = (
                self._choreography_model.forget(
                    f"feedback:{feedback_id}"
                )
                or sequence_update_removed
            )
            if sequence_update_removed:
                self._save_choreography_model()
            if existing is not None and existing.get("song_id") is not None:
                song_id = int(existing["song_id"])
                self._save_feedback_routine(
                    song_id, self.memory.list_feedback(song_id)
                )
            refresh_lanes = self._feedback_target_lanes(
                str((existing or {}).get("scope") or "overall"),
                (existing or {}).get("fixture_id"),
            )
            refresh_live = self.engine_mode == "live"
            if refresh_live:
                self._queue_feedback_bias_refresh(refresh_lanes)
            else:
                self._rebuild_feedback_biases()
            with self._lock:
                runtime = self._runtime
                biases = deepcopy(self._feedback_biases)
                self._add_event(
                    "feedback", f"Removed feedback #{feedback_id}"
                )
                self._status_sequence += 1
            if runtime is not None and not refresh_live:
                runtime.replace_feedback(
                    biases,
                    replan_lanes=refresh_lanes,
                )
        return {
            "deleted": True,
            "feedback_id": feedback_id,
            "sequence_update_removed": sequence_update_removed,
        }

    def _revise_or_forget_feedback_batch(
        self,
        model_event_id: str,
        batch_rows: list[dict[str, Any]],
    ) -> bool:
        if not batch_rows:
            return self._choreography_model.forget(model_event_id)
        participants = {
            str(row.get("participant_id") or f"legacy:{row.get('id')}")
            for row in batch_rows
        }
        repeat_taps = max(0, len(batch_rows) - len(participants))
        urgency = clamp(
            0.45
            + 0.14 * (max(1, len(participants)) - 1)
            + 0.06 * repeat_taps,
            0.0,
            1.0,
        )
        return self._choreography_model.revise_feedback_event(
            model_event_id,
            occurrences=len(batch_rows),
            urgency=urgency,
        )

    @staticmethod
    def _feedback_consensus_start_ms(
        feedback_row: dict[str, Any],
    ) -> int:
        lane_context = feedback_row.get("lane_context")
        if isinstance(lane_context, dict):
            value = lane_context.get("consensus_started_unix_ms")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        # Preserve the identifier and undo behavior of feedback written before
        # sliding consensus anchors were stored.
        created_ms = int(feedback_row.get("created_unix_ms") or 0)
        return created_ms // 5_000 * 5_000

    @staticmethod
    def _feedback_batch_event_id(
        *,
        song_id: int,
        listening_session_id: str,
        created_unix_ms: int,
        label: str,
        scope: str,
        fixture_id: Any,
        section: str,
        routine: str,
        lane: str | None = None,
        active_sequence_id: str | None = None,
        boundary_id: str | None = None,
        lifetime: str | None = None,
    ) -> str:
        # Calls without a lane preserve the version-1 identifier used by
        # older tests and tools. New persisted feedback always supplies a
        # lane and therefore includes the planner lease in its identity.
        batch_start_ms = int(created_unix_ms) // 5_000 * 5_000
        material = "\x1f".join(
            (
                str(song_id),
                str(listening_session_id),
                str(batch_start_ms),
                str(label),
                str(scope),
                str(fixture_id or ""),
                str(section),
                str(routine),
                *( () if lifetime is None else (str(lifetime),) ),
            )
        )
        legacy = "feedback-batch:" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()[:24]
        if lane is None:
            return legacy
        group = LumenApplication._feedback_batch_group_event_id(
            song_id=song_id,
            listening_session_id=listening_session_id,
            created_unix_ms=created_unix_ms,
            label=label,
            scope=scope,
            fixture_id=fixture_id,
            lifetime=lifetime,
        )
        context_material = "\x1f".join((
            str(lane),
            str(section),
            str(routine),
            str(active_sequence_id or "unknown"),
            str(boundary_id or "unknown"),
        ))
        context_hash = hashlib.sha256(
            context_material.encode("utf-8")
        ).hexdigest()[:16]
        return f"{group}:{context_hash}:{lane}"

    @staticmethod
    def _feedback_batch_group_event_id(
        *,
        song_id: int,
        listening_session_id: str,
        created_unix_ms: int,
        label: str,
        scope: str,
        fixture_id: Any,
        lifetime: str | None = None,
    ) -> str:
        batch_start_ms = int(created_unix_ms) // 5_000 * 5_000
        material = "\x1f".join((
            str(song_id),
            str(listening_session_id),
            str(batch_start_ms),
            str(label),
            str(scope),
            str(fixture_id or ""),
            *( () if lifetime is None else (str(lifetime),) ),
        ))
        return "feedback-batch:" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _feedback_lane_context(
        feedback_row: dict[str, Any], lane: str
    ) -> dict[str, str] | None:
        encoded = feedback_row.get("lane_context")
        if not isinstance(encoded, dict):
            return None
        lanes = encoded.get("lanes")
        if not isinstance(lanes, dict):
            return None
        context = lanes.get(lane)
        if not isinstance(context, dict):
            return None
        return {
            "section": str(context.get("section") or "unknown"),
            "routine": str(context.get("routine") or "unknown"),
            "active_sequence_id": str(
                context.get("active_sequence_id") or "unknown"
            ),
            "boundary_id": str(
                context.get("boundary_id") or "unknown"
            ),
        }

    def _feedback_target_lanes(
        self, scope: str, fixture_or_group_id: Any
    ) -> tuple[str, ...]:
        """Map persisted feedback targets to independent live lanes."""

        if str(scope).casefold().strip() == "overall":
            return CHOREOGRAPHY_LANES
        target = str(fixture_or_group_id or "").casefold().strip()
        semantic = choreography_lanes_for_scope(target)
        if semantic:
            return semantic
        if any(
            fixture.fixture_id.casefold() == target
            for fixture in self.rig.fixtures
        ):
            return ("movers",)
        if any(
            fixture.fixture_id.casefold() == target
            for fixture in self.rig.auxiliary_fixtures
        ):
            return ("center",)
        return CHOREOGRAPHY_LANES

    def _media_position_ms(
        self, *, at_monotonic_s: float | None = None
    ) -> int | None:
        media = self.media
        sample_age_ms = (
            0
            if at_monotonic_s is None
            else max(
                0,
                round((time.monotonic() - at_monotonic_s) * 1000),
            )
        )
        if media is None:
            if self.started_at is None:
                return None
            position = round((time.monotonic() - self.started_at) * 1000)
            return max(0, position - sample_age_ms)
        if media.observed_position_ms is None:
            return None
        if not media.is_playing or media.observed_at_unix_ms is None:
            return media.observed_position_ms
        elapsed = (
            int(time.time() * 1000)
            - sample_age_ms
            - media.observed_at_unix_ms
        )
        position = max(0, media.observed_position_ms + elapsed)
        if media.duration_ms is not None:
            position = min(position, media.duration_ms)
        return position

    def _poll_spotify_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_media_poll < SPOTIFY_MEDIA_POLL_INTERVAL_S:
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
        # Bootstrap is an interface composition operation. System probes and
        # SQLite summaries can take seconds, so capture only the small mutable
        # pieces under the live-state lock and perform every slow operation
        # after releasing it.
        disk_free_bytes = self._read_training_disk_free()
        with self._lock:
            self._training_disk_free_bytes = disk_free_bytes
            rig = deepcopy(self._rig_payload)
            status = self._snapshot_unlocked()
            settings = deepcopy(self.operator_settings())
        return {
            "project": {
                "name": "Lumen Engine",
                "version": "0.7.2",
                "role": "Spatial music-lighting control",
            },
            "rig": rig,
            "profiles": [
                profile_summary(profile)
                for profile in PARTY_PARROT_PROFILES.values()
            ],
            "status": status,
            "memory": self.memory.summary(limit=30),
            "research": self.research_status(wait_for_readiness=False),
            "system": self.scan_system(),
            "settings": settings,
        }

    def snapshot(
        self, *, include_analysis_history: bool = True
    ) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot_unlocked()
            if not include_analysis_history:
                snapshot["analysis_history"] = []
            return snapshot

    def song_teaching_snapshot(self, *, force: bool = False) -> dict[str, Any]:
        """Read the teaching timeline without holding the audio-state lock."""

        with self._lock:
            song_id = self.song_id
            media = self.media
            cached_structure = deepcopy(self._cached_structure_prediction)
            active_ids = list(self._recalled_choreography_ids)
        position_ms = self._media_position_ms()
        recording = (
            self.memory.resolve_recording_version(
                provider=media.provider,
                provider_item_id=media.provider_item_id,
                duration_ms=media.duration_ms,
            )
            if media is not None else None
        )
        recording_id = (recording or {}).get("id")
        if song_id is None or recording_id is None:
            return {
                "available": False,
                "recording": recording,
                "cached_structure": cached_structure,
                "structure_timelines": [],
                "sequences": [],
                "placements": [],
            }
        with self._teaching_lock:
            now = time.monotonic()
            if (
                not force
                and self._teaching_snapshot_cache is not None
                and self._teaching_snapshot_cache.get("song_id") == song_id
                and self._teaching_snapshot_cache.get("recording_id")
                == recording_id
                and now - self._teaching_snapshot_checked_at < 2.0
            ):
                result = deepcopy(self._teaching_snapshot_cache)
            else:
                sequences = self.memory.list_choreography_sequences(
                    song_id=song_id
                )
                placements = self.memory.list_choreography_placements(
                    song_id=song_id
                )
                timelines = self.memory.structure_timelines_for_recording(
                    str(recording_id)
                )
                result = {
                    "available": True,
                    "song_id": song_id,
                    "recording_id": recording_id,
                    "recording": recording,
                    "structure_timelines": timelines,
                    "sequences": sequences[-40:],
                    "placements": placements[-80:],
                }
                self._teaching_snapshot_cache = deepcopy(result)
                self._teaching_snapshot_checked_at = now
        result["position_ms"] = position_ms
        result["cached_structure"] = cached_structure
        result["active_sequence_ids"] = active_ids
        return result

    def structure_training_library(
        self, recording_id: str | None = None
    ) -> dict[str, Any]:
        """Return the operator's complete offline timeline review library."""

        catalog = self.memory.structure_timeline_catalog()
        active_run_ids: set[str] = set()
        evaluation_path = self._student_model_path.with_name(
            self._student_model_path.stem + ".evaluation.json"
        )
        if evaluation_path.is_file():
            try:
                evaluation = json.loads(
                    evaluation_path.read_text(encoding="utf-8")
                )
                active_run_ids = {
                    str(run_id)
                    for run_id in evaluation.get("teacher_run_ids", [])
                }
            except (OSError, json.JSONDecodeError):
                active_run_ids = set()
        for item in catalog:
            included = bool(
                active_run_ids.intersection(item.get("teacher_run_ids", []))
            )
            item["included_in_active_student"] = included
            if included:
                item["training_status"] = "active_student_source"
            elif item.get("training_eligible"):
                item["training_status"] = "ready_for_next_training"
            elif item.get("capture_status") == "partial":
                item["training_status"] = "excluded_partial_capture"
            else:
                item["training_status"] = "diagnostic_only"
        selected = None
        requested = str(recording_id or "").strip()
        if requested:
            selected = next(
                (
                    item for item in catalog
                    if item["recording_id"] == requested
                ),
                None,
            )
            if selected is None:
                raise ValueError("the selected timeline recording was not found")
        elif catalog:
            selected = next(
                (
                    item for item in catalog
                    if item["review_status"] == "needs_review"
                ),
                catalog[0],
            )
        selected_id = (
            str(selected["recording_id"]) if selected is not None else None
        )
        return {
            "catalog": catalog,
            "recordings": len(catalog),
            "needs_review": sum(
                item["review_status"] == "needs_review" for item in catalog
            ),
            "reviewed": sum(bool(item["reviewed"]) for item in catalog),
            "selected_recording_id": selected_id,
            "selected_recording": selected,
            "structure_timelines": (
                self.memory.structure_timelines_for_recording(selected_id)
                if selected_id is not None else []
            ),
        }

    def correct_structure_timeline(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach an operator correction to the exact playing recording."""

        base_timeline_id = str(
            payload.get("base_timeline_id", "")
        ).strip()
        if not base_timeline_id:
            raise ValueError("base_timeline_id is required")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        if len(segments) > 2_000:
            raise ValueError("correction contains too many segments")
        participant_id = (
            str(payload.get("participant_id", "")).strip() or None
        )
        participant_name = (
            str(payload.get("participant_name", "")).strip() or None
        )
        note = str(payload.get("note", "")).strip() or None
        if participant_id is not None and len(participant_id) > 96:
            raise ValueError("participant_id is too long")
        if participant_name is not None and len(participant_name) > 32:
            raise ValueError("participant_name is too long")
        if note is not None and len(note) > 1_000:
            raise ValueError("correction note is too long")

        base = self.memory.structure_timeline(base_timeline_id)
        if base is None:
            raise ValueError("the selected recording timeline is unavailable")
        requested_recording_id = str(
            payload.get("recording_id", "")
        ).strip()
        with self._lock:
            media = self.media
            song_id = self.song_id
        playing_recording = (
            self.memory.resolve_recording_version(
                provider=media.provider,
                provider_item_id=media.provider_item_id,
                duration_ms=media.duration_ms,
            )
            if media is not None else None
        )
        recording_id = requested_recording_id or str(
            (playing_recording or {}).get("id") or ""
        )
        if not recording_id:
            raise ValueError("select a song from the timeline library")
        if str(base.get("recording_id")) != recording_id:
            raise ValueError(
                "the selected timeline does not belong to the selected recording"
            )

        timeline_id = self.memory.save_structure_correction(
            base_timeline_id=base_timeline_id,
            segments=segments,
            participant_id=participant_id,
            participant_name=participant_name,
            note=note,
        )
        position_ms = self._media_position_ms()
        is_playing_recording = str(
            (playing_recording or {}).get("id") or ""
        ) == recording_id
        cached = (
            self.memory.cached_structure_at(
                recording_id=recording_id,
                playback_position_ms=position_ms,
            )
            if is_playing_recording and position_ms is not None else None
        )
        with self._lock:
            if (
                is_playing_recording
                and media == self.media
                and song_id == self.song_id
            ):
                self._cached_structure_prediction = cached
                self._cached_structure_checked_at = time.monotonic()
            self._memory_context_last_poll = 0.0
            self._teaching_snapshot_checked_at = 0.0
            self._teaching_snapshot_cache = None
            self._add_event(
                "memory",
                f"Saved structure correction for {base_timeline_id}",
            )
            self._status_sequence += 1
        return {
            "timeline_id": timeline_id,
            "base_timeline_id": base_timeline_id,
            "recording_id": recording_id,
            "cached_structure": cached,
        }

    def review_structure_timeline(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Approve or reject an unmodified teacher timeline for recall."""

        timeline_id = str(payload.get("timeline_id", "")).strip()
        status = str(payload.get("status", "")).strip().casefold()
        if not timeline_id:
            raise ValueError("timeline_id is required")
        participant_id = str(
            payload.get("participant_id", "")
        ).strip() or None
        participant_name = str(
            payload.get("participant_name", "")
        ).strip() or None
        note = str(payload.get("note", "")).strip() or None
        if participant_id is not None and len(participant_id) > 96:
            raise ValueError("participant_id is too long")
        if participant_name is not None and len(participant_name) > 32:
            raise ValueError("participant_name is too long")
        if note is not None and len(note) > 1_000:
            raise ValueError("review note is too long")
        timeline = self.memory.structure_timeline(timeline_id)
        if timeline is None:
            raise ValueError("the selected recording timeline is unavailable")
        requested_recording_id = str(
            payload.get("recording_id", "")
        ).strip()
        with self._lock:
            media = self.media
            song_id = self.song_id
        playing_recording = (
            self.memory.resolve_recording_version(
                provider=media.provider,
                provider_item_id=media.provider_item_id,
                duration_ms=media.duration_ms,
            )
            if media is not None else None
        )
        recording_id = requested_recording_id or str(
            (playing_recording or {}).get("id") or ""
        )
        if not recording_id:
            raise ValueError("select a song from the timeline library")
        if str(timeline.get("recording_id")) != recording_id:
            raise ValueError(
                "the selected timeline does not belong to the selected recording"
            )
        review_candidate = next(
            (
                item
                for item in self.memory.structure_timelines_for_recording(
                    recording_id
                )
                if str(item.get("id")) == timeline_id
            ),
            None,
        )
        if (
            status == "approved"
            and (
                review_candidate is None
                or not review_candidate.get("review_eligible", False)
            )
        ):
            raise ValueError(
                "this diagnostic timeline cannot be approved for Live recall; "
                "save a canonical operator correction instead"
            )
        review = self.memory.review_structure_timeline(
            timeline_id=timeline_id,
            status=status,
            participant_id=participant_id,
            participant_name=participant_name,
            note=note,
        )
        position_ms = self._media_position_ms()
        is_playing_recording = str(
            (playing_recording or {}).get("id") or ""
        ) == recording_id
        cached = (
            self.memory.cached_structure_at(
                recording_id=recording_id,
                playback_position_ms=position_ms,
            )
            if is_playing_recording and position_ms is not None else None
        )
        with self._lock:
            if (
                is_playing_recording
                and media == self.media
                and song_id == self.song_id
            ):
                self._cached_structure_prediction = cached
                self._cached_structure_checked_at = time.monotonic()
            self._memory_context_last_poll = 0.0
            self._teaching_snapshot_checked_at = 0.0
            self._teaching_snapshot_cache = None
            self._add_event(
                "memory", f"Structure timeline {timeline_id} {status}"
            )
            self._status_sequence += 1
        return {
            "timeline_id": timeline_id,
            "recording_id": recording_id,
            "review": review,
            "cached_structure": cached,
        }

    def _cached_song_teaching_snapshot_unlocked(self) -> dict[str, Any]:
        cached = self._teaching_snapshot_cache
        if cached is None or cached.get("song_id") != self.song_id:
            return {
                "available": self.song_id is not None,
                "song_id": self.song_id,
                "position_ms": self._media_position_ms(),
                "cached_structure": self._cached_structure_prediction,
                "structure_timelines": [],
                "sequences": [],
                "placements": [],
                "loading": self.song_id is not None,
            }
        result = deepcopy(cached)
        result["position_ms"] = self._media_position_ms()
        result["cached_structure"] = self._cached_structure_prediction
        result["active_sequence_ids"] = list(self._recalled_choreography_ids)
        return result

    def _feedback_evidence_snapshot_unlocked(self) -> dict[str, Any]:
        """Return the totals displayed by Live without its large context map."""

        evidence = [
            value["evidence"]
            for value in self._feedback_biases.values()
            if value.get("evidence")
        ]
        if not evidence:
            return {}
        # The interface has always displayed the greatest listener count and
        # the sum of contextual event counts.  Preserve those values exactly,
        # but send one compact record instead of the complete learning model
        # ten times per second.
        return {
            "summary": {
                "listeners": max(int(item.get("listeners", 0)) for item in evidence),
                "events": sum(int(item.get("events", 0)) for item in evidence),
                "repeat_events": sum(
                    int(item.get("repeat_events", 0)) for item in evidence
                ),
                "confidence": max(
                    float(item.get("confidence", 0.0)) for item in evidence
                ),
            }
        }

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
                    "error"
                    if self._student_model_error
                    else self._student_model_state
                ),
                "model_path": str(self._student_model_path),
                "error": self._student_model_error,
                "notice": self._student_model_notice,
                "gate_reasons": list(self._student_model_gate_reasons),
                "prediction": self._student_prediction,
                "cached_timeline": self._cached_structure_prediction,
                "effective_source": (
                    self._effective_structure.get(
                        "source", "live_analyzer"
                    )
                ),
                "effective_resolution": deepcopy(
                    self._effective_structure
                ),
                "memory_context": {
                    "lookup_duration_ms": self._memory_context_last_duration_ms,
                    "error": self._memory_context_error,
                    "runs_on_audio_thread": False,
                    "timing_role": "structural_context_only",
                    "beat_sync_authority": "audio_sample_clock",
                },
            },
            "song_teaching": self._cached_song_teaching_snapshot_unlocked(),
            "choreography": deepcopy(self._runtime_choreography_snapshot),
            "learning": {
                "model_revision": self._choreography_model.revision,
                "learned_sequence_candidates": len(
                    self._choreography_model.learned_candidates()
                ),
                "applied_feedback_evidence": (
                    self._feedback_evidence_snapshot_unlocked()
                ),
                "application_rule": (
                    "Feedback accumulates immediately, but scalar and "
                    "sequence changes wait for the next phrase boundary."
                ),
                "trace_queue_drops": self._trace_queue_drops,
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
            query_values = parse_qs(parsed.query)
            include_history = query_values.get("history", ["0"])[0] in {
                "1",
                "true",
                "yes",
            }
            self._json_body(
                HTTPStatus.OK,
                self.server.status_body(
                    include_analysis_history=include_history
                ),
            )
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
        if path == "/api/choreography":
            force = str(
                parse_qs(parsed.query).get("force", [""])[0]
            ).lower() in {"1", "true", "yes"}
            self._json(
                HTTPStatus.OK,
                self.server.application.song_teaching_snapshot(force=force),
            )
            return
        if path == "/api/structure/library":
            recording_id = parse_qs(parsed.query).get(
                "recording_id", [None]
            )[0]
            try:
                self._json(
                    HTTPStatus.OK,
                    self.server.application.structure_training_library(
                        recording_id
                    ),
                )
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if path == "/api/research":
            self._json(
                HTTPStatus.OK,
                self.server.application.research_status(
                    wait_for_readiness=False
                ),
            )
            return
        if path == "/api/link/status":
            query_values = parse_qs(parsed.query)
            summary_only = query_values.get("summary", ["0"])[0] in {
                "1",
                "true",
                "yes",
            }
            status = self.server.application.lumen_link.status()
            if summary_only:
                # The phone remote needs only enough information to show
                # connection and active-work state. Keep the detailed event
                # history and full job list off its periodic response.
                status = {
                    **status,
                    "jobs": list(status.get("jobs") or [])[:1],
                    "events": [],
                    "summary": True,
                }
            self._json(
                HTTPStatus.OK,
                status,
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
            elif path == "/api/choreography/save":
                result = app.save_choreography_proposal(payload)
            elif path == "/api/choreography/history":
                result = app.edit_choreography_history(payload)
            elif path == "/api/structure/correct":
                result = app.correct_structure_timeline(payload)
            elif path == "/api/structure/review":
                result = app.review_structure_timeline(payload)
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
            elif path.startswith("/api/link/"):
                action = path.rsplit("/", 1)[-1]
                if action not in {
                    "test",
                    "enable",
                    "pause",
                    "resume",
                    "disable",
                }:
                    raise ValueError("unknown Lumen Link control")
                result = app.lumen_link.control(action)
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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._json_body(status, body)

    def _json_body(self, status: HTTPStatus, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        return

    def _shutdown_server(self) -> None:
        time.sleep(0.15)
        self.server.shutdown()


class LumenHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # The stdlib default is five pending sockets. A listening session can
    # legitimately wake several phones and the desktop dashboard at once, so
    # absorb that connection burst instead of resetting valid feedback before
    # a request thread is assigned.
    request_queue_size = 128

    def __init__(
        self,
        address: tuple[str, int],
        application: LumenApplication,
    ) -> None:
        self.application = application
        # Rendering a status document walks the 3D solution, DMX heatmap,
        # analysis history, training state, and choreography state. Multiple
        # browsers need the same display generation; recomputing it once per
        # socket can starve audio analysis on this six-core machine. This
        # cache is deliberately owned by the HTTP layer, never the live clock,
        # and remains fresh enough for a smooth 30 Hz operator display.
        self._status_body_lock = threading.Lock()
        self._status_body_cached_at: dict[bool, float] = {}
        self._status_body_cache: dict[bool, bytes] = {}
        super().__init__(address, LumenRequestHandler)

    def status_body(
        self, *, include_analysis_history: bool = True
    ) -> bytes:
        now = time.monotonic()
        with self._status_body_lock:
            if (
                include_analysis_history in self._status_body_cache
                and now
                - self._status_body_cached_at.get(
                    include_analysis_history, 0.0
                )
                < 1.0 / 30.0
            ):
                return self._status_body_cache[include_analysis_history]
            body = json.dumps(
                self.application.snapshot(
                    include_analysis_history=include_analysis_history
                ),
                separators=(",", ":"),
            ).encode("utf-8")
            self._status_body_cache[include_analysis_history] = body
            self._status_body_cached_at[
                include_analysis_history
            ] = time.monotonic()
            return body

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
