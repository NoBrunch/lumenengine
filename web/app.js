"use strict";

const app = {
  remote: window.location.pathname === "/remote",
  bootstrap: null,
  status: null,
  system: null,
  memory: null,
  spotify: null,
  page: "performance",
  selectedFixtureId: null,
  roomView: "plan",
  rigView: "plan",
  lastAudioPacketCount: 0,
  lastBeatPhase: null,
  lastBeatPulse: 0,
  beatDisplayTurns: 0,
  pointer: { dragging: false, moved: false, fixtureId: null },
  roomCamera: { yaw: -0.55, pitch: 0.62, panX: 0, panY: 0, zoom: 1 },
  polling: null,
  pollInFlight: false,
  systemRefreshing: false,
  teachingRefreshing: false,
  researchRefreshing: false,
  operatorTask: null,
  operatorTaskSerial: 0,
  researchTaskStartedAt: 0,
  lastStatusReceivedAt: 0,
  statusLatencyMs: 0,
  controlTimer: null,
  disconnected: false,
  pollCount: 0,
  spotifyRefreshing: false,
  spotifyFetchedAt: 0,
  spotifyPlaylistId: "",
  spotifyTransferDeviceId: "",
  spotifyHistory: [],
  spotifyHistoryIndex: -1,
  calibrationActive: false,
  calibrationCaptures: {},
  rehearsalTimer: null,
  motionTimer: null,
  motionEditorScope: null,
  participantId: null,
  participantName: "",
  feedbackReceipts: [],
  teaching: null,
  structureLibrary: null,
  selectedStructureRecordingId: null,
  sequenceDraft: [{ routine: "breathe", duration_beats: 8, intensity: 0.72, brightness: 0.72, motion_speed: 0.5, travel_size: 1, activity_density: 1, beat_sync: 1, palette: "", strobe: 0 }],
  editingSequenceId: null,
  editingPlacementId: null,
  editingStructureTimelineId: null,
  choreographyUndo: null,
  touchBlockedUntil: Date.now() + 1200,
  floatingPanelZ: 200,
};

function blockWakeTouches(milliseconds = 1200) {
  app.touchBlockedUntil = Math.max(
    app.touchBlockedUntil,
    Date.now() + milliseconds,
  );
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") blockWakeTouches();
});
window.addEventListener("pageshow", () => blockWakeTouches());
window.addEventListener("focus", () => blockWakeTouches());

function initializeParticipantIdentity() {
  const storageKey = "lumen.feedback.participant.v1";
  const nameKey = "lumen.feedback.participantName.v1";
  let participantId = window.localStorage.getItem(storageKey);
  if (!participantId) {
    participantId = window.crypto?.randomUUID?.()
      || `listener-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    window.localStorage.setItem(storageKey, participantId);
  }
  app.participantId = participantId;
  app.participantName = window.localStorage.getItem(nameKey) || "";
}

initializeParticipantIdentity();

function renderParticipantIdentity() {
  for (const id of ["feedback-listener-name", "remote-listener-name"]) {
    const input = $(id);
    if (input && document.activeElement !== input) {
      input.value = app.participantName;
    }
  }
}

function updateParticipantName(value) {
  app.participantName = String(value || "").trim().slice(0, 32);
  window.localStorage.setItem(
    "lumen.feedback.participantName.v1",
    app.participantName
  );
  renderParticipantIdentity();
}

const $ = (id) => document.getElementById(id);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

async function api(path, options = {}) {
  const init = {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
  };
  if (options.body !== undefined) init.body = JSON.stringify(options.body);
  const response = await fetch(path, init);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { error: `${response.status} ${response.statusText}` };
  }
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function clamp(value, low = 0, high = 1) {
  return Math.max(low, Math.min(high, Number(value) || 0));
}

function percent(value) {
  return `${Math.round(clamp(value) * 100)}%`;
}

function formatTime(milliseconds) {
  if (milliseconds === null || milliseconds === undefined || !Number.isFinite(Number(milliseconds))) return "--:--";
  const seconds = Math.max(0, Math.floor(Number(milliseconds) / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function formatElapsed(unixMs) {
  if (!unixMs) return "—";
  const delta = Math.max(0, Date.now() - Number(unixMs));
  if (delta < 60_000) return "just now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h ago`;
  return `${Math.floor(delta / 86_400_000)}d ago`;
}

function label(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function toast(title, message = "", kind = "") {
  const element = document.createElement("div");
  element.className = `toast ${kind}`;
  element.innerHTML = `<b>${escapeHtml(title)}</b>${message ? `<span>${escapeHtml(message)}</span>` : ""}`;
  $("toast-stack").append(element);
  window.setTimeout(() => element.remove(), 4200);
}

function beginOperatorTask(title, detail = "", button = null) {
  const task = {
    id: ++app.operatorTaskSerial,
    title,
    detail,
    startedAt: Date.now(),
    button,
    buttonWasDisabled: Boolean(button?.disabled),
  };
  app.operatorTask = task;
  if (button) {
    button.classList.add("task-pending");
    button.setAttribute("aria-busy", "true");
    button.disabled = true;
  }
  renderOperatorTask();
  return task;
}

function finishOperatorTask(task) {
  if (task?.button) {
    task.button.classList.remove("task-pending");
    task.button.removeAttribute("aria-busy");
    task.button.disabled = task.buttonWasDisabled;
  }
  if (app.operatorTask?.id === task?.id) app.operatorTask = null;
  renderOperatorTask();
}

function researchServerTask(research = app.bootstrap?.research || {}) {
  const preparation = Boolean(research.preparation?.running);
  const worker = research.worker || {};
  if (!preparation && !worker.running) {
    app.researchTaskStartedAt = 0;
    return null;
  }
  if (!app.researchTaskStartedAt) app.researchTaskStartedAt = Date.now();
  const progress = worker.progress || {};
  const currentType = String(progress.current_job_type || "");
  const done = Number(progress.processed || 0);
  const planned = Number(progress.planned || 0);
  const stage = String(progress.resources?.stage || "");
  const stageLabels = {
    student_feature_preparation: "Preparing causal audio features",
    student_training: "Training the temporal model",
  };
  if (preparation) {
    return {
      title: "Preparing captured audio",
      detail: "Verifying continuity, recording identity, checksums, and full-song eligibility.",
      startedAt: app.researchTaskStartedAt,
    };
  }
  if (worker.cancel_requested) {
    return {
      title: "Pausing offline work",
      detail: "Waiting for the current model checkpoint; completed jobs are preserved.",
      startedAt: app.researchTaskStartedAt,
    };
  }
  if (currentType === "student.train") {
    return {
      title: "Training and validating the structure model",
      detail: stageLabels[stage] || "Using trusted training songs, then evaluating the held-out songs.",
      startedAt: app.researchTaskStartedAt,
    };
  }
  return {
    title: "Analyzing recordings with EDMFormer",
    detail: planned > 0
      ? `${done} of ${planned} queued jobs finished. The current full-song job is still active.`
      : "A full-song teacher job is active. Progress will update after each recording completes.",
    startedAt: app.researchTaskStartedAt,
  };
}

function renderOperatorTask(research = app.bootstrap?.research || {}) {
  const element = $("operator-task");
  if (!element) return;
  const task = app.operatorTask || researchServerTask(research);
  element.classList.toggle("hidden", !task);
  if (!task) return;
  setText("operator-task-title", task.title || "Working");
  setText("operator-task-detail", task.detail || "Lumen is completing the requested operation.");
  const elapsed = Math.max(0, (Date.now() - Number(task.startedAt || Date.now())) / 1000);
  setText("operator-task-elapsed", formatUptime(elapsed));
}

function confirmButtonPress(button) {
  if (!button) return;
  button.classList.remove("press-confirmed");
  void button.offsetWidth;
  button.classList.add("press-confirmed");
  window.setTimeout(() => button.classList.remove("press-confirmed"), 380);
}

function setText(id, value) {
  const element = $(id);
  const next = String(value ?? "");
  if (element && element.textContent !== next) element.textContent = next;
}

function setWidth(id, value) {
  const element = $(id);
  const next = `${clamp(value) * 100}%`;
  if (element && element.style.width !== next) element.style.width = next;
}

function setStatusClass(element, state) {
  if (!element) return;
  element.classList.remove("ok", "active", "warn", "error");
  if (state) element.classList.add(state);
}

function fixtures() {
  if (!app.bootstrap) return [];
  const rig = app.bootstrap.rig;
  return [
    ...(rig.fixtures || []).map((fixture) => ({ ...fixture, kind: "moving" })),
    ...(rig.auxiliary_fixtures || []).map((fixture) => ({ ...fixture, kind: "auxiliary" })),
  ];
}

function profileFor(key) {
  return (app.bootstrap?.profiles || []).find((profile) => profile.key === key) || null;
}

function selectedFixture() {
  return fixtures().find((fixture) => fixture.id === app.selectedFixtureId) || null;
}

function setPage(name) {
  if (!["performance", "rehearsal", "rig", "audio", "memory", "music", "system"].includes(name)) return;
  app.page = name;
  $$(".workspace-page").forEach((page) => page.classList.toggle("active", page.dataset.page === name));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.nav === name));
  if (name === "rig") window.setTimeout(drawRig, 30);
  if (name === "performance") window.setTimeout(drawPerformanceRoom, 30);
  if (name === "rehearsal") {
    renderRehearsal(app.status?.rehearsal || {});
    window.setTimeout(drawMotionPath, 30);
  }
  if (name === "audio") {
    renderTrainingDataset(app.status?.training || {}, app.status?.engine || {});
    window.setTimeout(drawScope, 30);
  }
  if (name === "system") renderDmx(app.status || {});
  if (name === "music") refreshSpotifyConsole(false);
}

function panelWorkspaceKey(panel, index) {
  const page = panel.closest(".workspace-page")?.dataset.page || "workspace";
  const heading = panel.querySelector(":scope > .panel-titlebar h3")?.textContent
    || panel.className;
  const slug = heading.trim().toLocaleLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${page}.${slug || index}`;
}

function saveFloatingPanel(panel) {
  const key = panel.dataset.panelWorkspaceKey;
  if (!key) return;
  const state = panel.classList.contains("panel-floating")
    ? {
        floating: true,
        left: parseFloat(panel.style.left) || 0,
        top: parseFloat(panel.style.top) || 0,
        width: parseFloat(panel.style.width) || panel.offsetWidth,
        height: parseFloat(panel.style.height) || panel.offsetHeight,
      }
    : { floating: false };
  window.localStorage.setItem(`lumen.panel.${key}.v2`, JSON.stringify(state));
}

function constrainFloatingPanel(panel) {
  const minimumWidth = Math.min(340, window.innerWidth - 16);
  const minimumHeight = Math.min(210, window.innerHeight - 16);
  const width = Math.max(
    minimumWidth,
    Math.min(
      parseFloat(panel.style.width) || panel.offsetWidth,
      window.innerWidth - 8,
    ),
  );
  const height = Math.max(
    minimumHeight,
    Math.min(
      parseFloat(panel.style.height) || panel.offsetHeight,
      window.innerHeight - 8,
    ),
  );
  const left = Math.max(
    4,
    Math.min(
      parseFloat(panel.style.left) || 4,
      window.innerWidth - width - 4,
    ),
  );
  const top = Math.max(
    4,
    Math.min(
      parseFloat(panel.style.top) || 4,
      window.innerHeight - height - 4,
    ),
  );
  Object.assign(panel.style, {
    width: `${width}px`,
    height: `${height}px`,
    left: `${left}px`,
    top: `${top}px`,
  });
}

function setPanelFloating(panel, floating, saved = null) {
  const button = panel.querySelector(
    ":scope > .panel-titlebar .panel-float-toggle",
  );
  if (!floating) {
    panel.classList.remove("panel-floating");
    for (const property of ["left", "top", "width", "height", "zIndex", "cursor"]) {
      panel.style[property] = "";
    }
    if (button) {
      button.textContent = "↗";
      button.title = "Float, move, and resize this panel";
      button.setAttribute("aria-label", button.title);
    }
    saveFloatingPanel(panel);
    return;
  }
  const rect = panel.getBoundingClientRect();
  const preferredWidth = panel.classList.contains("sequence-editor-panel")
    ? Math.max(rect.width, window.innerWidth * 0.82)
    : Math.max(rect.width, Math.min(760, window.innerWidth * 0.7));
  const preferredHeight = panel.classList.contains("sequence-editor-panel")
    ? Math.max(rect.height, window.innerHeight * 0.82)
    : Math.max(rect.height, Math.min(620, window.innerHeight * 0.72));
  panel.classList.add("panel-floating");
  Object.assign(panel.style, {
    left: `${saved?.left ?? Math.max(12, (window.innerWidth - preferredWidth) / 2)}px`,
    top: `${saved?.top ?? Math.max(12, (window.innerHeight - preferredHeight) / 2)}px`,
    width: `${saved?.width ?? preferredWidth}px`,
    height: `${saved?.height ?? preferredHeight}px`,
    zIndex: String(++app.floatingPanelZ),
  });
  constrainFloatingPanel(panel);
  if (button) {
    button.textContent = "↙";
    button.title = "Dock this panel back into the dashboard";
    button.setAttribute("aria-label", button.title);
  }
  saveFloatingPanel(panel);
}

function beginPanelMove(event, panel) {
  if (!panel.classList.contains("panel-floating") || event.button !== 0) return;
  if (event.target.closest("button, input, select, a, summary")) return;
  event.preventDefault();
  panel.style.zIndex = String(++app.floatingPanelZ);
  const rect = panel.getBoundingClientRect();
  const startX = event.clientX;
  const startY = event.clientY;
  const move = (next) => {
    panel.style.left = `${rect.left + next.clientX - startX}px`;
    panel.style.top = `${rect.top + next.clientY - startY}px`;
    constrainFloatingPanel(panel);
  };
  const finish = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    saveFloatingPanel(panel);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish, { once: true });
}

function beginPanelResize(event, panel, edge) {
  if (!panel.classList.contains("panel-floating") || event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  panel.style.zIndex = String(++app.floatingPanelZ);
  const rect = panel.getBoundingClientRect();
  const startX = event.clientX;
  const startY = event.clientY;
  const minimumWidth = Math.min(340, window.innerWidth - 16);
  const minimumHeight = Math.min(210, window.innerHeight - 16);
  const move = (next) => {
    const dx = next.clientX - startX;
    const dy = next.clientY - startY;
    let left = rect.left;
    let top = rect.top;
    let width = rect.width;
    let height = rect.height;
    if (edge.includes("e")) width = Math.max(minimumWidth, rect.width + dx);
    if (edge.includes("s")) height = Math.max(minimumHeight, rect.height + dy);
    if (edge.includes("w")) {
      width = Math.max(minimumWidth, rect.width - dx);
      left = rect.right - width;
    }
    if (edge.includes("n")) {
      height = Math.max(minimumHeight, rect.height - dy);
      top = rect.bottom - height;
    }
    Object.assign(panel.style, {
      left: `${left}px`,
      top: `${top}px`,
      width: `${width}px`,
      height: `${height}px`,
    });
    constrainFloatingPanel(panel);
  };
  const finish = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    saveFloatingPanel(panel);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish, { once: true });
}

function panelResizeEdge(event, panel) {
  if (!panel.classList.contains("panel-floating")) return "";
  const rect = panel.getBoundingClientRect();
  const margin = 12;
  const vertical = event.clientY - rect.top <= margin
    ? "n"
    : rect.bottom - event.clientY <= margin ? "s" : "";
  const horizontal = event.clientX - rect.left <= margin
    ? "w"
    : rect.right - event.clientX <= margin ? "e" : "";
  return `${vertical}${horizontal}`;
}

function installPanelWorkspace() {
  if (app.remote) return;
  $$(".desktop-workspace .panel").forEach((panel, index) => {
    const titlebar = panel.querySelector(":scope > .panel-titlebar");
    if (!titlebar || titlebar.querySelector(".panel-float-toggle")) return;
    panel.dataset.panelWorkspaceKey = panelWorkspaceKey(panel, index);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mini-tool panel-float-toggle";
    button.textContent = "↗";
    button.title = "Float, move, and resize this panel";
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => {
      setPanelFloating(panel, !panel.classList.contains("panel-floating"));
    });
    titlebar.append(button);
    titlebar.addEventListener(
      "pointerdown",
      (event) => beginPanelMove(event, panel),
    );
    panel.addEventListener("pointerdown", (event) => {
      const edge = panelResizeEdge(event, panel);
      if (edge) beginPanelResize(event, panel, edge);
    }, { capture: true });
    panel.addEventListener("pointermove", (event) => {
      if (!panel.classList.contains("panel-floating")) return;
      const cursors = {
        n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
        ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize",
      };
      panel.style.cursor = cursors[panelResizeEdge(event, panel)] || "";
    });
    panel.addEventListener("pointerleave", () => {
      if (panel.classList.contains("panel-floating")) panel.style.cursor = "";
    });
    const storageKey = `lumen.panel.${panel.dataset.panelWorkspaceKey}.v2`;
    try {
      const saved = JSON.parse(
        window.localStorage.getItem(storageKey) || "null",
      );
      if (saved?.floating) setPanelFloating(panel, true, saved);
    } catch {
      window.localStorage.removeItem(storageKey);
    }
  });
}

async function initialize() {
  if (app.remote) document.body.classList.add("remote-mode");
  installHandlers();
  installPanelWorkspace();
  renderParticipantIdentity();
  renderSequenceDraft();
  updateClock();
  window.setInterval(updateClock, 1000);
  try {
    app.bootstrap = await api("/api/bootstrap");
    app.status = app.bootstrap.status;
    app.system = app.bootstrap.system;
    app.memory = app.bootstrap.memory;
    renderBootstrap();
    renderStatus();
    await refreshSongTeaching();
    await refreshStructureLibrary();
    refreshSpotifyConsole(false);
    app.disconnected = false;
    $("loading-screen").classList.add("loaded");
  } catch (error) {
    $("loading-screen").classList.add("loaded");
    toast("Could not open Lumen", error.message, "error");
    renderConnection(false);
  }
  app.polling = window.setTimeout(pollStatus, 100);
  window.addEventListener("resize", () => {
    $$(".panel-floating").forEach(constrainFloatingPanel);
    drawPerformanceRoom();
    drawRig();
    drawScope();
    drawMotionPath();
  });
}

async function pollStatus() {
  if (app.pollInFlight) return;
  app.pollInFlight = true;
  const requestedAt = performance.now();
  try {
    const status = await api("/api/status");
    app.statusLatencyMs = performance.now() - requestedAt;
    app.lastStatusReceivedAt = Date.now();
    app.status = status;
    if (app.disconnected) toast("Lumen reconnected", "Live state is available again.", "success");
    app.disconnected = false;
    renderStatus();
    renderConnection(true);
    app.pollCount += 1;
    if (app.pollCount % 600 === 0) void refreshSystemStatus();
    if (app.pollCount % 50 === 0) void refreshSongTeaching();
    const researchRunning = Boolean(app.bootstrap?.research?.worker?.running);
    if (app.page === "audio" && app.pollCount % (researchRunning ? 50 : 300) === 0) void refreshResearch();
    if (app.pollCount % 100 === 0 && app.system?.spotify?.token_present) {
      refreshSpotifyConsole(false);
    }
  } catch {
    if (!app.disconnected) toast("Connection interrupted", "Trying to reconnect to the local Lumen service.", "error");
    app.disconnected = true;
    renderConnection(false);
  } finally {
    app.pollInFlight = false;
    if (app.polling !== null) {
      app.polling = window.setTimeout(pollStatus, 100);
    }
  }
}

async function refreshSystemStatus() {
  if (app.systemRefreshing) return;
  app.systemRefreshing = true;
  try {
    app.system = await api("/api/system");
    renderSystem(app.system);
    updateComponentStatuses();
  } catch (_error) {
    // Live status remains authoritative; a hardware inventory refresh may be
    // retried later without marking the operator console disconnected.
  } finally {
    app.systemRefreshing = false;
  }
}

function renderBootstrap() {
  const rig = app.bootstrap.rig;
  setText("rig-name", rig.name);
  setText("room-fixture-count", `${fixtures().length} fixtures · ${Number(rig.room.width_m).toFixed(2)} × ${Number(rig.room.depth_m).toFixed(2)} m`);
  setText("patch-count", `${fixtures().length} PATCHED`);
  setText("patch-universe-count", new Set(fixtures().map((fixture) => fixture.universe || 0)).size);
  const footprint = fixtures().reduce((total, fixture) => total + (profileFor(fixture.profile_key)?.dmx_footprint || 1), 0);
  setText("patch-channel-count", footprint);
  renderFixtureList();
  renderFeedbackTargets();
  renderRehearsal(app.status?.rehearsal || {});
  if (!app.selectedFixtureId && fixtures().length) selectFixture(fixtures()[0].id);
  renderSystem(app.system);
  renderOperatorSettings(app.bootstrap.settings || {});
  renderResearch(app.bootstrap.research || {});
  renderMemory(app.memory);
  buildDmxHeatmap();
  renderServiceDetails();
  drawPerformanceRoom();
  drawRig();
  drawScope();
}

function renderFeedbackTargets() {
  const options = ['<option value="overall">Overall performance</option>'];
  if (fixtures().filter((fixture) => fixture.kind === "moving").length > 1) {
    options.push('<option value="group:movers">Movers</option>');
  }
  if (fixtures().some((fixture) => fixture.kind === "auxiliary")) {
    options.push('<option value="group:center">Center multi-effect</option>');
  }
  ["feedback-scope", "remote-feedback-scope"].forEach((id) => {
    const select = $(id);
    if (select) select.innerHTML = options.join("");
  });
}

function renderRehearsal(rehearsal = {}) {
  const routines = rehearsal.routines || [];
  const container = $("rehearsal-routines");
  if (container && container.dataset.signature !== routines.map((item) => item.id).join("|")) {
    container.dataset.signature = routines.map((item) => item.id).join("|");
    container.innerHTML = routines.map((item) => `<button class="routine-card" data-rehearsal-routine="${escapeHtml(item.id)}"><b>${escapeHtml(item.name)}</b><span>${escapeHtml(item.description)}</span></button>`).join("");
  }
  $$('[data-rehearsal-routine]').forEach((button) => {
    button.classList.toggle("active", button.dataset.rehearsalRoutine === rehearsal.routine);
  });
  const selected = routines.find((item) => item.id === rehearsal.routine) || {};
  const scopeSelect = $("rehearsal-scope");
  if (scopeSelect) {
    const scopeOptions = [
      '<option value="movers">Movers</option>',
      '<option value="center">Center multi-effect</option>',
      '<option value="overall">Whole rig</option>',
    ];
    const signature = scopeOptions.join("");
    if (scopeSelect.dataset.signature !== signature) {
      scopeSelect.dataset.signature = signature;
      scopeSelect.innerHTML = signature;
    }
  }
  setText("rehearsal-routine-name", selected.name || label(rehearsal.routine || "figure_eight"));
  setText("rehearsal-routine-description", selected.description || "Select a movement to inspect it in isolation.");
  const running = Boolean(app.status?.engine?.running && app.status?.engine?.mode === "rehearsal");
  setText("rehearsal-state", running ? "RUNNING" : "STOPPED");
  setText("rehearsal-output-badge", rehearsal.output === "live" ? "LIVE RIG" : "VIRTUAL PREVIEW");
  const tour = $("rehearsal-tour");
  if (tour) {
    tour.textContent = `Tour: ${rehearsal.tour ? "on" : "off"}`;
    tour.classList.toggle("active", Boolean(rehearsal.tour));
  }
  const values = {
    "rehearsal-output": rehearsal.output,
    "rehearsal-scope": rehearsal.scope,
    "rehearsal-bpm": rehearsal.bpm,
    "rehearsal-size": Number(rehearsal.size) * 100,
    "rehearsal-intensity": Number(rehearsal.intensity) * 100,
    "rehearsal-strobe": Number(rehearsal.strobe) * 100,
    "rehearsal-palette": rehearsal.palette,
  };
  for (const [id, value] of Object.entries(values)) {
    const element = $(id);
    if (element && document.activeElement !== element && value !== undefined) element.value = value;
  }
  if ($("rehearsal-isolate")) $("rehearsal-isolate").checked = rehearsal.isolate !== false;
  setText("rehearsal-bpm-value", `${Math.round(Number(rehearsal.bpm || 120))} BPM`);
  setText("rehearsal-size-value", percent(rehearsal.size));
  setText("rehearsal-intensity-value", percent(rehearsal.intensity));
  setText("rehearsal-strobe-value", Number(rehearsal.strobe || 0) ? percent(rehearsal.strobe) : "Off");
  if ($("rehearsal-output")) $("rehearsal-output").disabled = running;
  if ($("rehearsal-start")) $("rehearsal-start").disabled = running;
  if ($("rehearsal-stop")) $("rehearsal-stop").disabled = !running;
  renderMotionEditor(rehearsal.motion_editor || {});
}

function renderMotionEditor(editor = {}) {
  if (!app.motionEditorScope) app.motionEditorScope = editor.scope || "movers";
  const scope = app.motionEditorScope;
  const scopedEditor = editor.groups?.[scope] || editor;
  const values = scopedEditor.values || {};
  const scopeSelect = $("motion-editor-scope");
  if (scopeSelect && document.activeElement !== scopeSelect) {
    scopeSelect.value = scope;
  }
  $("motion-mover-controls")?.classList.toggle("hidden", scope !== "movers");
  $("motion-mover-preview")?.classList.toggle("hidden", scope !== "movers");
  $("motion-center-controls")?.classList.toggle("hidden", scope !== "center");
  $("motion-center-preview")?.classList.toggle("hidden", scope !== "center");
  const controls = {
    "motion-cycle": values.cycle_beats,
    "motion-pan-size": Number(values.pan_size) * 100,
    "motion-tilt-size": Number(values.tilt_size) * 100,
    "motion-pan-center": Number(values.pan_center) * 100,
    "motion-tilt-center": Number(values.tilt_center) * 100,
    "motion-relationship": values.relationship,
    "motion-direction": values.direction,
  };
  if (scope === "movers") {
    for (const [id, value] of Object.entries(controls)) {
      const element = $(id);
      if (element && document.activeElement !== element && value !== undefined) element.value = value;
    }
    setText("motion-cycle-value", `${Number(values.cycle_beats || 0).toFixed(0)} beats`);
    setText("motion-pan-size-value", percent(values.pan_size));
    setText("motion-tilt-size-value", percent(values.tilt_size));
    setText("motion-pan-center-value", percent(values.pan_center));
    setText("motion-tilt-center-value", percent(values.tilt_center));
  }
  const centerControls = {
    "center-cycle": values.cycle_beats,
    "center-relationship": values.relationship,
    "center-body-travel": Number(values.body_travel) * 100,
    "center-body-speed": Number(values.body_speed) * 8,
    "center-body-phase": Number(values.body_phase) * 100,
    "center-body-direction": values.body_direction,
    "center-arm-1-travel": Number(values.arm_1_travel) * 100,
    "center-arm-1-speed": Number(values.arm_1_speed) * 8,
    "center-arm-1-phase": Number(values.arm_1_phase) * 100,
    "center-arm-1-direction": values.arm_1_direction,
    "center-arm-2-travel": Number(values.arm_2_travel) * 100,
    "center-arm-2-speed": Number(values.arm_2_speed) * 8,
    "center-arm-2-phase": Number(values.arm_2_phase) * 100,
    "center-arm-2-direction": values.arm_2_direction,
    "center-emitter-pattern": values.emitter_pattern,
    "center-color-pattern": values.color_pattern,
    "center-laser-mode": values.laser_mode,
    "center-laser-level": Number(values.laser_level) * 100,
    "center-strip-program": values.strip_program,
    "center-strip-speed": Number(values.strip_speed) * 100,
    "center-strobe-level": Number(values.strobe_level) * 100,
    "center-intensity": Number(values.intensity) * 100,
    "center-blackout-accent": Number(values.blackout_accent) * 100,
  };
  if (scope === "center") {
    for (const [id, value] of Object.entries(centerControls)) {
      const element = $(id);
      if (element && document.activeElement !== element && value !== undefined) element.value = value;
    }
    setText("center-cycle-value", `${Number(values.cycle_beats || 0).toFixed(0)} beats`);
    setText("center-body-travel-value", percent(values.body_travel));
    setText("center-body-speed-value", `${Number(values.body_speed || 0).toFixed(2)}×`);
    setText("center-body-phase-value", percent(values.body_phase));
    setText("center-arm-1-travel-value", percent(values.arm_1_travel));
    setText("center-arm-1-speed-value", `${Number(values.arm_1_speed || 0).toFixed(2)}×`);
    setText("center-arm-1-phase-value", percent(values.arm_1_phase));
    setText("center-arm-2-travel-value", percent(values.arm_2_travel));
    setText("center-arm-2-speed-value", `${Number(values.arm_2_speed || 0).toFixed(2)}×`);
    setText("center-arm-2-phase-value", percent(values.arm_2_phase));
    setText("center-laser-level-value", percent(values.laser_level));
    setText("center-strip-program-value", Math.round(Number(values.strip_program || 0)));
    setText("center-strip-speed-value", percent(values.strip_speed));
    setText("center-strobe-level-value", Number(values.strobe_level || 0) ? percent(values.strobe_level) : "Off");
    setText("center-intensity-value", percent(values.intensity));
    setText("center-blackout-accent-value", Number(values.blackout_accent || 0) ? percent(values.blackout_accent) : "Off");
    setText("center-preview-body", `${Number(values.body_speed || 0).toFixed(2)}× · ${percent(values.body_travel)}`);
    setText("center-preview-arm-1", `${Number(values.arm_1_speed || 0).toFixed(2)}× · ${percent(values.arm_1_travel)}`);
    setText("center-preview-arm-2", `${Number(values.arm_2_speed || 0).toFixed(2)}× · ${percent(values.arm_2_travel)}`);
  }
  setText("motion-editor-status", scopedEditor.modified ? `${scope.toUpperCase()} · CUSTOM · SAVED` : `${scope.toUpperCase()} · DEFAULT · SAVED`);
  const velocity = scopedEditor.velocity || [];
  const worst = velocity.reduce((current, item) => Math.max(
    current,
    Number(item.required_pan_deg_s || 0) / Math.max(1, Number(item.maximum_pan_deg_s || 1)),
    Number(item.required_tilt_deg_s || 0) / Math.max(1, Number(item.maximum_tilt_deg_s || 1)),
  ), 0);
  setText("motion-velocity-status", scope === "center" ? "CENTER CONTROLS" : scopedEditor.velocity_feasible === false ? `TOO FAST · ${Math.round(worst * 100)}%` : `VELOCITY OK · ${Math.round(worst * 100)}%`);
  $("motion-velocity-status")?.classList.toggle("warn", scopedEditor.velocity_feasible === false);
  if (scope === "movers") drawMotionPath(scopedEditor.paths || []);
}

function drawMotionPath(paths = app.status?.rehearsal?.motion_editor?.paths || []) {
  const configured = configureCanvas($("motion-path-canvas"));
  if (!configured) return;
  const { context: ctx, width, height } = configured;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#071012";
  ctx.fillRect(0, 0, width, height);
  const pad = 28;
  ctx.strokeStyle = "rgba(93, 126, 124, .22)";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const x = pad + (width - pad * 2) * index / 4;
    const y = pad + (height - pad * 2) * index / 4;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, height - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(width - pad, y); ctx.stroke();
  }
  ctx.fillStyle = "#708783";
  ctx.font = "10px DejaVu Sans Mono";
  ctx.fillText("LEFT", pad, height - 9); ctx.fillText("RIGHT", width - pad - 32, height - 9);
  ctx.save(); ctx.translate(11, pad + 20); ctx.rotate(-Math.PI / 2); ctx.fillText("TILT", 0, 0); ctx.restore();
  const colors = ["#68d9cd", "#d7a85e"];
  paths.forEach((path, fixtureIndex) => {
    ctx.strokeStyle = colors[fixtureIndex] || "#b8c8c5";
    ctx.lineWidth = 2;
    ctx.beginPath();
    path.forEach(([pan, tilt], index) => {
      const x = pad + Number(pan) * (width - pad * 2);
      const y = height - pad - Number(tilt) * (height - pad * 2);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (path.length) {
      const cycle = Number(app.status?.rehearsal?.motion_editor?.groups?.movers?.values?.cycle_beats || 1);
      const beats = Number(app.status?.engine?.uptime_s || 0) * Number(app.status?.rehearsal?.bpm || 120) / 60;
      const marker = path[Math.floor((beats % cycle) / cycle * (path.length - 1))];
      if (marker) {
        ctx.fillStyle = colors[fixtureIndex];
        ctx.beginPath();
        ctx.arc(pad + marker[0] * (width - pad * 2), height - pad - marker[1] * (height - pad * 2), 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });
}

function motionFormValues() {
  if (app.motionEditorScope === "center") {
    return {
      cycle_beats: Number($("center-cycle")?.value || 8),
      relationship: $("center-relationship")?.value || "synchronized",
      body_travel: Number($("center-body-travel")?.value || 0) / 100,
      body_speed: Number($("center-body-speed")?.value || 8) / 8,
      body_phase: Number($("center-body-phase")?.value || 0) / 100,
      body_direction: Number($("center-body-direction")?.value || 1),
      arm_1_travel: Number($("center-arm-1-travel")?.value || 0) / 100,
      arm_1_speed: Number($("center-arm-1-speed")?.value || 8) / 8,
      arm_1_phase: Number($("center-arm-1-phase")?.value || 0) / 100,
      arm_1_direction: Number($("center-arm-1-direction")?.value || 1),
      arm_2_travel: Number($("center-arm-2-travel")?.value || 0) / 100,
      arm_2_speed: Number($("center-arm-2-speed")?.value || 8) / 8,
      arm_2_phase: Number($("center-arm-2-phase")?.value || 0) / 100,
      arm_2_direction: Number($("center-arm-2-direction")?.value || 1),
      emitter_pattern: $("center-emitter-pattern")?.value || "both",
      color_pattern: $("center-color-pattern")?.value || "palette",
      laser_mode: $("center-laser-mode")?.value || "off",
      laser_level: Number($("center-laser-level")?.value || 0) / 100,
      strip_program: Number($("center-strip-program")?.value || 0),
      strip_speed: Number($("center-strip-speed")?.value || 0) / 100,
      strobe_level: Number($("center-strobe-level")?.value || 0) / 100,
      intensity: Number($("center-intensity")?.value || 0) / 100,
      blackout_accent: Number($("center-blackout-accent")?.value || 0) / 100,
    };
  }
  return {
    cycle_beats: Number($("motion-cycle")?.value || 8),
    pan_size: Number($("motion-pan-size")?.value || 0) / 100,
    tilt_size: Number($("motion-tilt-size")?.value || 0) / 100,
    pan_center: Number($("motion-pan-center")?.value || 50) / 100,
    tilt_center: Number($("motion-tilt-center")?.value || 50) / 100,
    relationship: $("motion-relationship")?.value || "synchronized",
    direction: Number($("motion-direction")?.value || 1),
  };
}

async function patchMotionRoutine(values, action = null) {
  try {
    const body = { routine: app.status?.rehearsal?.routine, scope: app.motionEditorScope || "movers", values };
    if (action) body.action = action;
    app.status = await api("/api/motion-routine", { method: "POST", body });
    renderStatus();
  } catch (error) {
    toast("Motion edit failed", error.message, "error");
  }
}

function queueMotionRoutine() {
  window.clearTimeout(app.motionTimer);
  setText("motion-editor-status", "SAVING…");
  app.motionTimer = window.setTimeout(() => patchMotionRoutine(motionFormValues()), 110);
}

function rehearsalFormValues() {
  return {
    output: $("rehearsal-output")?.value || "virtual",
    scope: $("rehearsal-scope")?.value || "movers",
    bpm: Number($("rehearsal-bpm")?.value || 120),
    size: Number($("rehearsal-size")?.value || 100) / 100,
    intensity: Number($("rehearsal-intensity")?.value || 68) / 100,
    strobe: Number($("rehearsal-strobe")?.value || 0) / 100,
    palette: $("rehearsal-palette")?.value || "party_vivid",
    isolate: Boolean($("rehearsal-isolate")?.checked),
  };
}

async function patchRehearsal(values, quiet = false) {
  try {
    app.status = await api("/api/rehearsal", { method: "POST", body: values });
    renderStatus();
  } catch (error) {
    if (!quiet) toast("Rehearsal change failed", error.message, "error");
  }
}

function queueRehearsal(values) {
  window.clearTimeout(app.rehearsalTimer);
  app.rehearsalTimer = window.setTimeout(() => patchRehearsal(values, true), 90);
}

async function startRehearsal() {
  try {
    app.status = await api("/api/rehearsal", { method: "POST", body: rehearsalFormValues() });
    app.status = await api("/api/engine/start", { method: "POST", body: { mode: "rehearsal" } });
    renderStatus();
    toast("Rehearsal started", app.status.rehearsal.output === "live" ? "The selected routine is on the physical rig." : "Virtual DMX preview is active.", "success");
  } catch (error) {
    toast("Rehearsal could not start", error.message, "error");
  }
}

function stepRehearsal(direction) {
  const rehearsal = app.status?.rehearsal || {};
  const routines = rehearsal.routines || [];
  const index = Math.max(0, routines.findIndex((item) => item.id === rehearsal.routine));
  const next = routines[(index + direction + routines.length) % routines.length];
  if (next) patchRehearsal({ routine: next.id, tour: false });
}

function songTime(milliseconds) {
  if (milliseconds === null || milliseconds === undefined) return "No position";
  const totalSeconds = Math.max(0, Math.floor(Number(milliseconds) / 1000));
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

async function refreshSongTeaching(force = false) {
  if (app.teachingRefreshing) return;
  app.teachingRefreshing = true;
  try {
    app.teaching = await api(`/api/choreography${force ? "?force=1" : ""}`);
    renderSongTeaching(app.teaching);
  } catch (error) {
    setText("teacher-timeline-summary", `Song memory unavailable · ${error.message}`);
  } finally {
    app.teachingRefreshing = false;
  }
}

function renderSongTeaching(teaching = app.teaching || app.status?.song_teaching || {}) {
  setText("sequence-song-position", teaching.available ? `SONG ${teaching.song_id} · ${songTime(teaching.position_ms)}` : "No identified song");
  setText("remote-sequence-position", teaching.available ? songTime(teaching.position_ms) : "No song identity");
  const cached = teaching.cached_structure;
  if (cached?.axes) {
    const axes = ["functional", "energy", "content"].map((axis) => {
      const value = cached.axes[axis];
      return `${label(axis)}: ${label(value?.label || "unknown")} ${value ? percent(value.confidence) : ""}`;
    });
    const next = cached.boundary?.next;
    const authorities = [...new Set(Object.values(cached.axes)
      .filter(Boolean)
      .map((value) => value.recall_authority || value.teacher?.name || "cached teacher"))];
    setText("teacher-timeline-summary", `${axes.join(" · ")}${next ? ` · next boundary in ${songTime(next.in_ms)}` : ""} · ${authorities.map(label).join(" + ")} · line-in audio still drives beats`);
  } else {
    setText("teacher-timeline-summary", "No cached teacher timeline for this recording yet. Live analysis remains active.");
  }
  if (!app.structureLibrary) renderStructureTimelines(teaching);
  renderSequenceHistory(teaching);
}

async function refreshStructureLibrary(recordingId = app.selectedStructureRecordingId) {
  try {
    const query = recordingId
      ? `?recording_id=${encodeURIComponent(recordingId)}`
      : "";
    app.structureLibrary = await api(`/api/structure/library${query}`);
    app.selectedStructureRecordingId = app.structureLibrary.selected_recording_id || null;
    renderStructureLibrary();
  } catch (error) {
    setText("structure-library-status", `Song training library unavailable · ${error.message}`);
  }
}

function structureCatalogLabel(item) {
  const state = {
    needs_review: "NEEDS REVIEW",
    approved: "APPROVED",
    corrected: "CORRECTED",
    rejected: "REJECTED",
    diagnostic_only: "DIAGNOSTIC",
  }[item.review_status] || label(item.review_status);
  const artist = (item.artists || []).join(", ");
  return `${state} · ${item.title}${artist ? ` — ${artist}` : ""}`;
}

function filteredStructureCatalog() {
  const catalog = app.structureLibrary?.catalog || [];
  const needle = ($("structure-song-search")?.value || "").trim().toLocaleLowerCase();
  const statusFilter = $("structure-song-filter")?.value || "all";
  return catalog.filter((item) => {
    const haystack = `${item.title} ${(item.artists || []).join(" ")} ${item.album || ""}`.toLocaleLowerCase();
    const statusMatches = statusFilter === "all"
      || (statusFilter === "reviewed" && item.reviewed)
      || item.review_status === statusFilter
      || item.recording_id === app.selectedStructureRecordingId;
    return statusMatches && (!needle || haystack.includes(needle));
  });
}

function renderStructureSongOptions() {
  const select = $("structure-song-select");
  if (!select) return;
  const catalog = app.structureLibrary?.catalog || [];
  const matches = filteredStructureCatalog();
  select.innerHTML = matches.length
    ? matches.map((item) => `<option value="${escapeHtml(item.recording_id)}">${escapeHtml(structureCatalogLabel(item))}</option>`).join("")
    : `<option value="">${catalog.length ? "No songs match this search" : "No analyzed songs available"}</option>`;
  if (matches.some((item) => item.recording_id === app.selectedStructureRecordingId)) {
    select.value = app.selectedStructureRecordingId;
  } else {
    select.value = "";
  }
}

function renderStructureCatalogTable() {
  const body = $("structure-catalog-body");
  if (!body) return;
  const matches = filteredStructureCatalog();
  const trainingLabel = {
    active_student_source: "Used by active model",
    ready_for_next_training: "Ready for next training",
    excluded_partial_capture: "Excluded: partial capture",
    diagnostic_only: "Not a training source",
  };
  body.innerHTML = matches.length ? matches.map((item) => `
    <tr data-structure-recording="${escapeHtml(item.recording_id)}" class="${item.recording_id === app.selectedStructureRecordingId ? "selected" : ""}">
      <td><button data-structure-recording="${escapeHtml(item.recording_id)}"><b>${escapeHtml(item.title)}</b><small>${escapeHtml(item.album || "")}</small></button></td>
      <td>${escapeHtml((item.artists || []).join(", ") || "—")}</td>
      <td>${songTime(item.duration_ms)}</td>
      <td>${escapeHtml((item.teacher_sources || []).join(" + ") || "Local")}</td>
      <td><span class="catalog-state state-${escapeHtml(item.review_status)}">${escapeHtml(label(item.review_status))}</span><small>${item.timeline_count} ${item.timeline_count === 1 ? "timeline" : "timelines"}</small></td>
      <td><span class="catalog-state training-${escapeHtml(item.training_status)}">${escapeHtml(trainingLabel[item.training_status] || label(item.training_status))}</span><small>${escapeHtml(label(item.capture_status || "unknown capture"))}</small></td>
      <td>${escapeHtml(label(item.split || "—"))}</td>
      <td>${formatElapsed(item.latest_timeline_unix_ms)}</td>
    </tr>`).join("") : '<tr><td colspan="8">No songs match the current search and status filter.</td></tr>';
}

function renderStructureLibrary() {
  const library = app.structureLibrary || {};
  renderStructureSongOptions();
  renderStructureCatalogTable();
  const selected = library.selected_recording;
  setText(
    "structure-timeline-count",
    `${Number(library.needs_review || 0)} TO REVIEW · ${Number(library.recordings || 0)} SONGS`,
  );
  setText(
    "structure-library-status",
    selected
      ? `${label(selected.review_status)} · ${selected.title}${(selected.artists || []).length ? ` — ${selected.artists.join(", ")}` : ""} · ${songTime(selected.duration_ms)} · ${selected.timeline_count} ${selected.timeline_count === 1 ? "timeline" : "timelines"}`
      : "No analyzed song timelines are available yet.",
  );
  setText("structure-selected-title", selected?.title || "No song selected");
  setText("structure-selected-artist", (selected?.artists || []).join(", ") || selected?.album || "Choose a song above");
  setText("structure-selected-total", songTime(selected?.duration_ms || 0));
  renderStructureOverview(library);
  renderStructureTimelines(library);
}

function structureOverviewTimeline(library = {}) {
  const timelines = library.structure_timelines || [];
  return timelines.find((item) => item.provenance === "operator_correction")
    || timelines.find((item) => item.review?.status === "approved")
    || timelines.find((item) => item.review_eligible !== false)
    || timelines[0]
    || null;
}

function renderStructureOverview(library = {}) {
  const container = $("structure-overview");
  if (!container) return;
  const selected = library.selected_recording;
  const timeline = structureOverviewTimeline(library);
  if (!selected || !timeline?.segments?.length) {
    container.innerHTML = '<span class="empty-state">Select an analyzed song to display its section map.</span>';
    return;
  }
  const duration = Math.max(1, Number(selected.duration_ms || timeline.segments.at(-1)?.end_ms || 1));
  const segments = timeline.segments.map((segment, index) => {
    const start = Math.max(0, Number(segment.start_ms || 0));
    const end = Math.max(start, Number(segment.end_ms ?? duration));
    const width = Math.max(0.4, (end - start) / duration * 100);
    const energy = segment.energy_label || "unknown";
    const primary = segment.functional_label || energy || segment.raw_label || "unknown";
    return `<button class="structure-overview-segment energy-${escapeHtml(energy)}" data-overview-segment="${index}" style="width:${width}%" title="${escapeHtml(`${songTime(start)}–${songTime(end)} · ${label(primary)} · ${label(energy)}`)}"><b>${escapeHtml(label(primary))}</b><small>${songTime(start)}</small></button>`;
  }).join("");
  container.innerHTML = `<div class="structure-overview-track">${segments}<i id="structure-overview-playhead"></i></div><div class="structure-overview-legend"><span>Breakdown</span><span>Build</span><span>Drop</span><span>Groove</span><em>${escapeHtml(timeline.teacher?.name || timeline.provenance || "Timeline")}</em></div>`;
  updateStructureOverviewPlayhead(app.status || {});
}

function updateStructureOverviewPlayhead(status = {}) {
  const selectedId = app.structureLibrary?.selected_recording_id;
  const playingId = app.teaching?.recording_id || status.song_teaching?.recording_id;
  const selected = app.structureLibrary?.selected_recording;
  const isPlaying = Boolean(selectedId && selectedId === playingId);
  const position = isPlaying ? Number(status.media?.live_position_ms || app.teaching?.position_ms || 0) : 0;
  const duration = Number(selected?.duration_ms || 0);
  const progress = duration > 0 ? clamp(position / duration) : 0;
  const playhead = $("structure-overview-playhead");
  if (playhead) {
    playhead.style.left = `${progress * 100}%`;
    playhead.classList.toggle("inactive", !isPlaying);
  }
  setText("structure-selected-position", isPlaying ? songTime(position) : "Not playing");
}

const STRUCTURE_LABELS = {
  functional: ["", "intro", "verse", "pre_chorus", "chorus", "post_chorus", "bridge", "outro"],
  energy: ["", "silence", "intro", "breakdown", "build", "drop", "groove", "outro"],
  content: ["", "vocal", "instrumental", "solo", "transition"],
};
const STRUCTURE_EVENTS = ["", "section_start", "energy_rise", "energy_fall", "build_start", "drop_onset", "breakdown_onset", "groove_return", "outro_start", "track_end"];

function structureSelect(axis, value, index) {
  const values = [...(STRUCTURE_LABELS[axis] || [""])];
  const current = value || "";
  if (current && !values.includes(current)) values.push(current);
  return `<select data-structure-axis="${axis}" data-segment-index="${index}">${values.map((item) => `<option value="${escapeHtml(item)}"${item === current ? " selected" : ""}>${escapeHtml(item ? label(item) : "Not supplied")}</option>`).join("")}</select>`;
}

function structureEventSelect(value, index) {
  const current = value || "";
  return `<select data-structure-event="${index}">${STRUCTURE_EVENTS.map((item) => `<option value="${item}"${item === current ? " selected" : ""}>${escapeHtml(item ? label(item) : "No event")}</option>`).join("")}</select>`;
}

function structureTransitionEvent(segment = {}) {
  // Current normalized timelines use transition_event. Keep the old event
  // alias readable so existing local operator corrections remain editable.
  return segment.provenance?.transition_event
    || segment.provenance?.event
    || segment.transition_event
    || "";
}

function renderStructureTimelines(teaching = {}) {
  const container = $("structure-timeline-list");
  const count = $("structure-timeline-count");
  if (!container) return;
  const timelines = teaching.structure_timelines || [];
  if (count && !Array.isArray(teaching.catalog)) {
    count.textContent = `${timelines.length} ${timelines.length === 1 ? "TIMELINE" : "TIMELINES"}`;
  }
  if (!timelines.length) {
    container.innerHTML = '<p class="empty-state">No offline teacher timeline has been generated for this exact recording yet.</p>';
    return;
  }
  const pendingTimelines = timelines.filter((timeline) => (
    timeline.review_eligible !== false
    && !["approved", "rejected"].includes(
      timeline.review?.status || "unreviewed",
    )
  ));
  const displayedTimelines = pendingTimelines.length
    ? pendingTimelines
    : timelines;
  container.innerHTML = displayedTimelines.map((timeline) => {
    const teacher = timeline.teacher?.name || timeline.provenance || "Local timeline";
    const scored = Number(timeline.confidence || 0) > 0;
    const review = timeline.review?.status || "unreviewed";
    const isEditing = app.editingStructureTimelineId === timeline.id;
    const correctedFrom = timeline.metadata?.corrects_timeline_id;
    const rows = (timeline.segments || []).map((segment, index) => {
      const cells = isEditing
        ? `<td>${structureSelect("functional", segment.functional_label, index)}</td><td>${structureSelect("energy", segment.energy_label, index)}</td><td>${structureSelect("content", segment.content_label, index)}</td><td>${structureEventSelect(structureTransitionEvent(segment), index)}</td>`
        : `<td>${escapeHtml(label(segment.functional_label || "—"))}</td><td>${escapeHtml(label(segment.energy_label || "—"))}</td><td>${escapeHtml(label(segment.content_label || "—"))}</td><td>${escapeHtml(label(structureTransitionEvent(segment) || "—"))}</td>`;
      const start = Number(segment.start_ms || 0) / 1000;
      const end = segment.end_ms === null || segment.end_ms === undefined ? "" : Number(segment.end_ms) / 1000;
      const timing = isEditing
        ? `<td class="structure-time-edit"><input data-structure-start="${index}" type="number" min="0" step="0.1" value="${start}"><span>to</span><input data-structure-end="${index}" type="number" min="0" step="0.1" value="${end}"></td>`
        : `<td class="structure-time-readout"><b>${songTime(segment.start_ms)}</b><span>to</span><b>${segment.end_ms === null ? "end" : songTime(segment.end_ms)}</b></td>`;
      return `<tr data-structure-segment="${index}">${timing}${cells}<td title="Original teacher label">${escapeHtml(segment.raw_label || "—")}</td><td>${segment.label_confidence > 0 ? percent(segment.label_confidence) : "unscored"}</td></tr>`;
    }).join("");
    const reviewDisabled = timeline.review_eligible === false ? " disabled title=\"Diagnostic evidence cannot be approved for Live recall\"" : "";
    const reviewActions = review === "approved" || review === "rejected"
      ? `<span class="structure-review-complete state-${escapeHtml(review)}">${escapeHtml(label(review))}</span><button data-timeline-review="unreviewed" data-timeline-id="${escapeHtml(timeline.id)}">Reopen review</button>`
      : `<button data-timeline-review="approved" data-timeline-id="${escapeHtml(timeline.id)}"${reviewDisabled}>Approve</button><button data-timeline-review="rejected" data-timeline-id="${escapeHtml(timeline.id)}">Reject</button>`;
    return `<article class="structure-timeline-card${review === "approved" ? " approved" : review === "rejected" ? " rejected" : ""}">
      <header><div><b>${escapeHtml(teacher)}</b><span>${scored ? `${percent(timeline.confidence)} model confidence` : "unscored model output"} · ${escapeHtml(label(timeline.recall_authority || review))}</span><small>${escapeHtml(timeline.timeline_version || "unknown version")}${correctedFrom ? ` · correction of ${escapeHtml(correctedFrom)}` : ""}</small></div><div class="structure-review-actions">${reviewActions}<button data-timeline-correct="${escapeHtml(timeline.id)}">${isEditing ? "Cancel correction" : "Correct labels"}</button></div></header>
      <div class="structure-table-wrap"><table><thead><tr><th>Time</th><th>Function</th><th>Energy</th><th>Content</th><th>Transition event</th><th>Raw teacher label</th><th>Model score</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${isEditing ? `<div class="structure-correction-actions"><label><span>Correction note</span><input id="structure-correction-note" maxlength="1000" placeholder="What did the teacher get wrong?"></label><button class="primary" data-timeline-save="${escapeHtml(timeline.id)}">Save immutable correction</button></div>` : ""}
    </article>`;
  }).join("") + (pendingTimelines.length && timelines.length > pendingTimelines.length
    ? `<p class="structure-reviewed-hidden">${timelines.length - pendingTimelines.length} reviewed or historical timeline${timelines.length - pendingTimelines.length === 1 ? " is" : "s are"} hidden while this song still has active review work. Select Reviewed after completing the queue to inspect the full history.</p>`
    : "");
}

async function reviewStructureTimeline(timelineId, status) {
  const reviewedRecordingId = app.selectedStructureRecordingId;
  try {
    await api("/api/structure/review", { method: "POST", body: {
      timeline_id: timelineId,
      recording_id: app.selectedStructureRecordingId,
      status,
      participant_id: app.participantId,
      participant_name: app.participantName || null,
    }});
    await refreshStructureLibrary(reviewedRecordingId);
    await refreshSongTeaching(true);
    const currentStillPending = (app.structureLibrary?.catalog || []).some(
      (item) => item.recording_id === reviewedRecordingId
        && item.review_status === "needs_review",
    );
    const nextReview = !currentStillPending && (app.structureLibrary?.catalog || []).find(
      (item) => item.review_status === "needs_review"
        && item.recording_id !== reviewedRecordingId,
    );
    if (nextReview && status !== "unreviewed") {
      await refreshStructureLibrary(nextReview.recording_id);
    }
    const confirmation = {
      approved: "Approved for exact-song recall; your trust decision is stored with this timeline.",
      rejected: "Excluded from exact-song recall and future student targets; it remains available in review history.",
      unreviewed: "Returned to the review queue.",
    }[status] || "Review state updated.";
    toast(
      status === "unreviewed" ? "Timeline review reopened" : `Timeline ${status}`,
      nextReview && status !== "unreviewed"
        ? `${confirmation} Opened the next song awaiting review.`
        : currentStillPending && status !== "unreviewed"
          ? `${confirmation} Opened the next timeline for this song.`
        : confirmation,
      "success",
    );
  } catch (error) {
    toast("Timeline review failed", error.message, "error");
  }
}

async function saveStructureCorrection(timelineId) {
  const timeline = (app.structureLibrary?.structure_timelines || app.teaching?.structure_timelines || []).find((item) => item.id === timelineId);
  if (!timeline) return;
  const container = $("structure-timeline-list");
  const segments = (timeline.segments || []).map((segment, index) => {
    const readAxis = (axis) => container.querySelector(`[data-structure-axis="${axis}"][data-segment-index="${index}"]`)?.value || null;
    const start = Number(container.querySelector(`[data-structure-start="${index}"]`)?.value || 0);
    const endInput = container.querySelector(`[data-structure-end="${index}"]`)?.value;
    return {
      segment_index: index,
      start_ms: Math.round(start * 1000),
      end_ms: endInput === "" ? null : Math.round(Number(endInput) * 1000),
      functional_label: readAxis("functional"),
      energy_label: readAxis("energy"),
      content_label: readAxis("content"),
      event: container.querySelector(`[data-structure-event="${index}"]`)?.value || null,
    };
  });
  try {
    await api("/api/structure/correct", { method: "POST", body: {
      base_timeline_id: timelineId,
      recording_id: app.selectedStructureRecordingId,
      segments,
      note: $("structure-correction-note")?.value || null,
      participant_id: app.participantId,
      participant_name: app.participantName || null,
    }});
    app.editingStructureTimelineId = null;
    await refreshStructureLibrary(app.selectedStructureRecordingId);
    await refreshSongTeaching(true);
    toast("Correction saved", "The teacher original remains intact; this recording now recalls your revision.", "success");
  } catch (error) {
    toast("Correction could not be saved", error.message, "error");
  }
}

function renderSequenceDraft() {
  const container = $("sequence-steps");
  if (!container) return;
  let startBeat = 0;
  container.innerHTML = app.sequenceDraft.map((step, index) => {
    const rowStart = startBeat;
    startBeat += Number(step.duration_beats || 0);
    const paletteOptions = [
      ["", "Automatic"], ["party_vivid", "Party vivid"], ["midnight_teal", "Midnight teal"],
      ["cool", "Cool"], ["warm", "Warm"], ["magenta_blue", "Magenta / blue"],
      ["cyan_violet", "Cyan / violet"], ["red_amber", "Red / amber"],
    ].map(([value, name]) => `<option value="${value}"${step.palette === value ? " selected" : ""}>${name}</option>`).join("");
    const routineOptions = ["breathe", "fan_sweep", "figure_eight", "opposing_chase", "beat_nod", "counter_rotate"]
      .map((routine) => `<option value="${routine}"${step.routine === routine ? " selected" : ""}>${label(routine)}</option>`).join("");
    return `<div class="sequence-step" data-sequence-step="${index}">
      <div class="sequence-step-number"><b>${index + 1}</b><span>beat ${rowStart.toFixed(0)}</span></div>
      <label><span>Movement</span><select data-step-field="routine">${routineOptions}</select></label>
      <label><span>Length</span><input data-step-field="duration_beats" type="number" min="1" max="128" step="1" value="${Number(step.duration_beats || 8)}"></label>
      <label><span>Intensity <output>${Math.round(Number(step.intensity || 0) * 100)}%</output></span><input data-step-field="intensity" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.intensity || 0) * 100)}"></label>
      <label><span>Brightness <output>${Math.round(Number(step.brightness ?? step.intensity ?? 0) * 100)}%</output></span><input data-step-field="brightness" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.brightness ?? step.intensity ?? 0) * 100)}"></label>
      <label><span>Motion speed <output>${Math.round(Number(step.motion_speed ?? 0.5) * 100)}%</output></span><input data-step-field="motion_speed" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.motion_speed ?? 0.5) * 100)}"></label>
      <label><span>Travel <output>${Math.round(Number(step.travel_size ?? 1) * 100)}%</output></span><input data-step-field="travel_size" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.travel_size ?? 1) * 100)}"></label>
      <label><span>Activity <output>${Math.round(Number(step.activity_density ?? 1) * 100)}%</output></span><input data-step-field="activity_density" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.activity_density ?? 1) * 100)}"></label>
      <label><span>Beat sync <output>${Math.round(Number(step.beat_sync ?? 1) * 100)}%</output></span><input data-step-field="beat_sync" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.beat_sync ?? 1) * 100)}"></label>
      <label><span>Palette</span><select data-step-field="palette">${paletteOptions}</select></label>
      <label><span>Strobe <output>${Number(step.strobe || 0) ? `${Math.round(Number(step.strobe) * 100)}%` : "Off"}</output></span><input data-step-field="strobe" type="range" min="0" max="100" step="1" value="${Math.round(Number(step.strobe || 0) * 100)}"></label>
      <button data-step-remove="${index}"${app.sequenceDraft.length === 1 ? " disabled" : ""}>Remove</button>
    </div>`;
  }).join("");
}

function syncSequenceDraftFromDom() {
  $$('[data-sequence-step]').forEach((row) => {
    const index = Number(row.dataset.sequenceStep);
    const current = app.sequenceDraft[index];
    if (!current) return;
    row.querySelectorAll('[data-step-field]').forEach((input) => {
      const field = input.dataset.stepField;
      current[field] = field === "duration_beats"
        ? Math.max(1, Number(input.value || 1))
        : ["intensity", "brightness", "motion_speed", "travel_size", "activity_density", "beat_sync", "strobe"].includes(field)
          ? Number(input.value || 0) / 100
          : input.value;
    });
  });
  let startBeat = 0;
  app.sequenceDraft.forEach((step) => {
    step.start_beat = startBeat;
    startBeat += Number(step.duration_beats || 0);
  });
}

function clearSequenceDraft() {
  app.sequenceDraft = [{ routine: "breathe", duration_beats: 8, intensity: 0.72, brightness: 0.72, motion_speed: 0.5, travel_size: 1, activity_density: 1, beat_sync: 1, palette: "", strobe: 0 }];
  app.editingSequenceId = null;
  app.editingPlacementId = null;
  if ($("sequence-name")) $("sequence-name").value = "";
  renderSequenceDraft();
}

async function saveSequenceHere(remote = false) {
  if (remote) {
    app.sequenceDraft = [{
      routine: $("remote-sequence-routine")?.value || "breathe",
      duration_beats: Number($("remote-sequence-duration")?.value || 8),
      intensity: Number($("remote-sequence-intensity")?.value || 75) / 100,
      palette: "",
      strobe: 0,
      start_beat: 0,
    }];
  } else {
    syncSequenceDraftFromDom();
  }
  const scope = remote ? $("remote-sequence-scope")?.value || "movers" : $("sequence-scope")?.value || "movers";
  const clientEventId = window.crypto?.randomUUID?.() || `sequence-${Date.now()}-${Math.random()}`;
  try {
    const result = await api("/api/choreography/save", {
      method: "POST",
      body: {
        sequence_id: remote ? null : app.editingSequenceId,
        placement_id: remote ? null : app.editingPlacementId,
        name: remote ? `${label(app.sequenceDraft[0].routine)} at ${songTime(app.teaching?.position_ms)}` : $("sequence-name")?.value.trim() || "Taught lighting phrase",
        scope,
        section_label: remote ? null : $("sequence-section")?.value || null,
        steps: app.sequenceDraft,
        participant_id: app.participantId,
        participant_name: app.participantName || null,
        client_event_id: clientEventId,
        place: true,
      },
    });
    app.editingSequenceId = result.sequence_id;
    app.editingPlacementId = result.placement_id;
    app.choreographyUndo = { kind: "sequence", id: result.sequence_id };
    if ($("sequence-undo-history")) $("sequence-undo-history").disabled = false;
    await refreshSongTeaching(true);
    toast("Sequence taught", `${label(scope)} · ${result.steps} step${result.steps === 1 ? "" : "s"} · active from the next matching phrase boundary`, "success");
  } catch (error) {
    toast("Sequence could not be saved", error.message, "error");
  }
}

function renderSequenceHistory(teaching = {}) {
  const container = $("sequence-history-list");
  if (!container) return;
  const placementsBySequence = new Map();
  for (const placement of teaching.placements || []) {
    if (!placementsBySequence.has(placement.sequence_id)) placementsBySequence.set(placement.sequence_id, []);
    placementsBySequence.get(placement.sequence_id).push(placement);
  }
  const sequences = (teaching.sequences || []).filter((sequence) => sequence.source === "operator_sequence_editor");
  container.innerHTML = sequences.length ? sequences.map((sequence) => {
    const placements = placementsBySequence.get(sequence.id) || [];
    const placement = placements[0];
    const owner = sequence.participant_name || (sequence.participant_id === app.participantId ? "This device" : "Listener");
    return `<article class="sequence-history-item">
      <div><b>${escapeHtml(sequence.name || "Taught sequence")}</b><span>${label(sequence.fixture_scope)} · ${sequence.steps.length} steps · revision ${sequence.revision} · ${escapeHtml(owner)}</span><small>${placement ? `${songTime(placement.start_ms)} · ${label(placement.section_label || "time anchor")}` : "Not placed"}</small></div>
      <div><button data-sequence-load="${escapeHtml(sequence.id)}">Edit</button>${placement ? `<button data-placement-delete="${escapeHtml(placement.id)}">Remove placement</button>` : ""}<button data-sequence-delete="${escapeHtml(sequence.id)}">Delete</button></div>
    </article>`;
  }).join("") : '<p class="empty-state">No explicit sequences saved for this song. Build one above or use the mobile one-action teacher.</p>';
}

function loadSequenceForEdit(sequenceId) {
  const sequence = (app.teaching?.sequences || []).find((item) => item.id === sequenceId);
  if (!sequence) return;
  app.sequenceDraft = sequence.steps.map((step) => ({
    routine: step.routine,
    duration_beats: Number(step.duration_beats),
    intensity: Number(step.intensity),
    brightness: Number(step.parameters?.brightness ?? step.intensity),
    motion_speed: Number(step.parameters?.motion_speed ?? 0.5),
    travel_size: Number(step.parameters?.travel_size ?? 1),
    activity_density: Number(step.parameters?.activity_density ?? 1),
    beat_sync: Number(step.parameters?.beat_sync ?? 1),
    palette: step.palette || "",
    strobe: typeof step.strobe === "object" ? Number(step.strobe.rate || 0) : Number(step.strobe || 0),
  }));
  app.editingSequenceId = sequence.id;
  app.editingPlacementId = (app.teaching?.placements || []).find((item) => item.sequence_id === sequence.id)?.id || null;
  if ($("sequence-name")) $("sequence-name").value = sequence.name || "";
  if ($("sequence-scope")) $("sequence-scope").value = sequence.fixture_scope || "movers";
  renderSequenceDraft();
}

async function changeChoreographyHistory(kind, action, id) {
  try {
    await api("/api/choreography/history", { method: "POST", body: { kind, action, id, participant_id: app.participantId } });
    app.choreographyUndo = action === "delete" ? { kind, id } : null;
    if ($("sequence-undo-history")) $("sequence-undo-history").disabled = !app.choreographyUndo;
    await refreshSongTeaching(true);
    toast(`${label(action)} complete`, `${label(kind)} memory updated`, "success");
  } catch (error) {
    toast("Song memory was not changed", error.message, "error");
  }
}

async function undoLastFeedback() {
  const feedbackId = app.feedbackReceipts.pop();
  if (!feedbackId) {
    return toast(
      "Nothing to undo",
      "This device has not submitted feedback since the page was opened."
    );
  }
  await deleteFeedback(feedbackId, true);
}

function effectiveCueSummary(output = {}) {
  const strobe = output.strobe?.enabled
    ? `strobe ${percent(output.strobe.rate)}`
    : "strobe off";
  return `${label(output.routine || "waiting")} · speed ${percent(output.motion_speed)} · travel ${percent(output.travel_size)} · activity ${percent(output.activity_density)} · brightness ${percent(output.brightness)} · ${label(output.palette || "automatic palette")} · ${strobe}`;
}

function renderLiveTeachingReference(status = {}) {
  const media = status.media || {};
  const observation = status.observation || {};
  const decision = status.decision || {};
  const expression = decision.expression || {};
  const resolution = status.structure_model?.effective_resolution || {};
  const energyAxis = resolution.axes?.energy || {};
  const position = Number(media.live_position_ms || 0);
  const duration = Number(media.duration_ms || 0);
  const progress = duration > 0 ? clamp(position / duration) : 0;
  setText("teaching-track-title", media.title || "No identified song");
  setText("teaching-track-artist", (media.artists || []).join(", ") || media.album || "Start Live with Spotify connected");
  setText("teaching-position", songTime(position));
  setText("teaching-duration", songTime(duration));
  if ($("teaching-progress-fill")) $("teaching-progress-fill").style.width = `${progress * 100}%`;
  if ($("teaching-progress-marker")) $("teaching-progress-marker").style.left = `${progress * 100}%`;
  setText("teaching-energy", percent(expression.energy));
  setText("teaching-energy-detail", `Loudness ${percent(observation.loudness)} · rhythm density ${percent(observation.rhythm_density)} · bass share ${percent(observation.low_energy)}`);
  setText("teaching-structure", `${label(energyAxis.label || observation.section || "unknown")} · ${percent(energyAxis.decision_confidence ?? observation.section_confidence)}`);
  setText("teaching-structure-detail", `${label(energyAxis.source || "live analyzer")} · ${label(observation.transition_event || "no transition event")}`);
  setText("teaching-reference-source", label(energyAxis.source || "waiting for Live"));
  setText("teaching-rhythm", observation.bpm ? `${Number(observation.bpm).toFixed(1)} BPM` : "No lock");
  setText("teaching-rhythm-detail", `${percent(observation.beat_confidence)} beat confidence · ${percent(observation.beat_phase)} beat phase`);
  const change = Math.max(Number(observation.arrangement_change || 0), Number(observation.novelty || 0), Number(observation.harmonic_change || 0));
  setText("teaching-change", percent(change));
  setText("teaching-change-detail", `Arrangement ${percent(observation.arrangement_change)} · novelty ${percent(observation.novelty)} · harmony ${percent(observation.harmonic_change)}`);
  for (const lane of ["movers", "center"]) {
    const plan = status.choreography?.lanes?.[lane] || {};
    const outputs = plan.effective_outputs || [];
    const first = outputs[0];
    setText(`teaching-${lane}-output`, first ? `${label(first.routine)} · ${outputs.length} fixture${outputs.length === 1 ? "" : "s"}` : "Waiting");
    setText(`teaching-${lane}-detail`, first ? effectiveCueSummary(first) : "No effective cue");
  }
  updateStructureOverviewPlayhead(status);
}

function captureLiveCue(lane) {
  const status = app.status || {};
  const plan = status.choreography?.lanes?.[lane] || {};
  const outputs = plan.effective_outputs || [];
  if (!outputs.length) {
    toast("No live cue to copy", `Lumen has not published an effective ${lane} output yet.`, "error");
    return;
  }
  const first = outputs[0];
  const average = (field, fallback) => outputs.reduce((total, item) => total + Number(item[field] ?? fallback), 0) / outputs.length;
  const strobeRate = Math.max(...outputs.map((item) => item.strobe?.enabled ? Number(item.strobe.rate || 0) : 0));
  app.sequenceDraft = [{
    routine: first.routine === "parked" || first.routine === "hold" ? "breathe" : first.routine,
    duration_beats: Number(plan.active_step?.duration_beats || 8),
    intensity: Number(plan.active_step?.intensity ?? average("brightness", 0.72)),
    brightness: average("brightness", 0.72),
    motion_speed: average("motion_speed", 0.5),
    travel_size: average("travel_size", 1),
    activity_density: average("activity_density", 1),
    beat_sync: average("beat_sync", 1),
    palette: first.palette || "",
    strobe: strobeRate,
    start_beat: 0,
  }];
  app.editingSequenceId = null;
  app.editingPlacementId = null;
  if ($("sequence-scope")) $("sequence-scope").value = lane;
  if ($("sequence-section")) $("sequence-section").value = status.observation?.section || "";
  if ($("sequence-name")) $("sequence-name").value = `${label(first.routine)} response`;
  renderSequenceDraft();
  $("sequence-steps")?.scrollIntoView({ behavior: "smooth", block: "center" });
  toast("Live cue loaded", "Review every characteristic below; it has not been saved or taught yet.", "success");
}

function renderStatus() {
  const status = app.status;
  if (!status) return;
  const engine = status.engine;
  const controls = status.controls;
  const observation = status.observation;
  const decision = status.decision;
  const expression = decision?.expression || { energy: 0, tension: 0, motion: 0, intimacy: 0.5, confidence: 0 };
  const running = Boolean(engine.running);
  const fault = engine.phase === "fault";
  const tempoClock = status.audio?.tempo_clock || {};
  const spectralTempo = tempoClock.spectral || {};
  const tempoAmbiguous = Number(spectralTempo.family_ambiguity || 0) >= 0.62
    && !spectralTempo.octave_promoted;
  const mood = `${label(decision?.gesture || "standing_by")} · ${label(observation?.section || "waiting")} · ${label(decision?.routine || "no_routine")}`;
  setText("analysis-mood", mood);
  setText("analysis-mood-detail", decision?.reason || "Waiting for a musical observation.");
  setText("analysis-energy", percent(expression.energy));
  setText("analysis-motion", percent(expression.motion));
  setText(
    "analysis-timing",
    observation?.bpm
      ? `${Number(observation.bpm).toFixed(1)} BPM · ${percent(observation.beat_confidence)} ${tempoAmbiguous ? "family ambiguity" : "lock"}`
      : tempoAmbiguous ? "Resolving half/double-time family" : "Searching for tempo",
  );
  const branch = status.solutions?.[0]?.branch || "No fixture solution";
  setText("analysis-resolution", branch.replaceAll("/", " → "));
  const structure = status.structure_model || {};
  const cachedTeacher = structure.cached_timeline;
  const student = structure.prediction;
  const selectedByStudent = student && student.selected_axis === "student_energy";
  const acceptedStudentAxes = student
    ? Object.entries(student.accepted_axes || {}).filter(([, accepted]) => accepted).map(([axis]) => axis)
    : [];
  if (cachedTeacher?.axes) {
    const axes = ["functional", "energy", "content"]
      .map((axis) => cachedTeacher.axes[axis])
      .filter(Boolean);
    const primary = cachedTeacher.axes.energy || cachedTeacher.axes.functional || axes[0];
    setText("analysis-structure", `${label(primary?.label || "teacher context")} · ${percent(cachedTeacher.confidence || primary?.confidence)}`);
    setText(
      "analysis-structure-detail",
      `Cached ${axes.map((axis) => `${label(axis.label)} ${percent(axis.confidence)}`).join(" · ")} · offline lookup ${structure.memory_context?.lookup_duration_ms ?? "—"} ms · line-in remains beat authority.`,
    );
  } else if (structure.state === "error") {
    setText("analysis-structure", "Model load error");
    setText("analysis-structure-detail", structure.error || "The saved student artifact could not be loaded.");
  } else if (selectedByStudent) {
    const selectedConfidence = Number(
      student.confidence?.[student.selected_axis === "student_functional" ? "functional" : "energy"] || 0,
    );
    setText("analysis-structure", `${label(student.selected_section)} · ${percent(selectedConfidence)}`);
    setText(
      "analysis-structure-detail",
      `Accepted ${label(student.selected_axis)} context is influencing expression and motion; choreography changes remain phrase-boundary decisions.`,
    );
  } else if (structure.state === "active") {
    if (acceptedStudentAxes.length) {
      setText("analysis-structure", `Live energy · ${acceptedStudentAxes.map(label).join(" + ")}`);
      setText(
        "analysis-structure-detail",
        "Accepted functional/content context can rank phrase-level routines; the live analyzer remains responsible for energy because no student energy label passed its gate.",
      );
    } else {
      setText("analysis-structure", "Live analyzer fallback");
      setText(
        "analysis-structure-detail",
        "The student is active, but its current labels are below the confidence gate and cannot steer choreography or motion.",
      );
    }
  } else {
    setText("analysis-structure", "Live analyzer only");
    setText("analysis-structure-detail", "No validated streaming student is active; Lumen is using its live authored analyzer.");
  }
  const choreography = status.choreography || {};
  for (const lane of ["movers", "center"]) {
    const plan = choreography.lanes?.[lane] || {};
    const step = plan.active_step;
    setText(`analysis-${lane}-plan`, step ? `${label(step.routine)} · ${Number(step.duration_beats || 0).toFixed(0)} beats` : "Waiting");
    setText(
      `analysis-${lane}-plan-detail`,
      step
        ? `${label(plan.active_sequence_source || "authored")} · ${percent(plan.confidence)} selection confidence${(choreography.replan_pending_lanes || []).includes(lane) ? " · feedback queued for next phrase" : " · phrase held"}`
        : "No phrase lease yet.",
    );
  }
  const learning = status.learning || {};
  const evidence = Object.values(learning.applied_feedback_evidence || {});
  const listenerEvidence = evidence.reduce((highest, item) => Math.max(highest, Number(item.listeners || 0)), 0);
  const eventEvidence = evidence.reduce((total, item) => total + Number(item.events || 0), 0);
  setText("analysis-learning", evidence.length ? `${listenerEvidence} listener${listenerEvidence === 1 ? "" : "s"} · ${eventEvidence} weighted events` : "No applied evidence yet");
  setText(
    "analysis-learning-detail",
    `${learning.application_rule || "Feedback waits for a phrase boundary."} Model revision ${learning.model_revision || 0} · ${learning.learned_sequence_candidates || 0} learned sequence candidates.`,
  );

  setText("engine-status", "");
  const engineStatus = $("engine-status");
  if (engineStatus) {
    engineStatus.innerHTML = `<i></i><span>ENGINE</span><b>${escapeHtml(label(engine.phase))}</b>`;
    setStatusClass(engineStatus, fault ? "error" : running ? "active" : "ok");
  }
  setText("rail-state", label(engine.phase));
  setText("rail-detail", fault ? engine.error || "Engine fault" : running ? `${label(engine.mode)} mode · ${formatUptime(engine.uptime_s)}` : "Engine is standing by");
  setText("footer-state", label(engine.phase));
  setText("footer-message", fault ? engine.error : running ? `${label(engine.mode)} mode on ${engine.audio_device}` : "Lumen operator console connected");
  setText("remote-engine-state", label(engine.phase));
  setText("remote-output-state", status.output ? `${status.output.backend} · ${status.output.frames_sent || 0} frames` : "No output active");
  setText("audio-device-name", engine.audio_device);
  renderAudioDiagnostics(status.audio, engine);
  if (app.page === "audio") renderTrainingDataset(status.training || {}, engine);
  const researchSnapshot = app.bootstrap?.research || {};
  const researchBusy = Boolean(
    researchSnapshot.worker?.running || researchSnapshot.preparation?.running
  );
  if ($("research-import-button")) {
    $("research-import-button").disabled = running || researchBusy;
  }
  if ($("research-run-button")) {
    $("research-run-button").disabled = running || researchBusy;
  }
  if ($("research-train-button")) {
    $("research-train-button").disabled = running
      || researchBusy
      || !Boolean(researchSnapshot.training?.train_ready);
  }

  for (const id of ["rail-state", "footer-state"]) {
    const element = $(id);
    if (element) element.title = engine.error || "";
  }
  for (const id of ["footer-lamp"]) {
    const element = $(id);
    if (element) element.className = `status-lamp ${fault ? "fault" : running ? "active" : "ready"}`;
  }

  renderEngineButtons(engine);
  renderControls(controls);
  if (app.page === "rehearsal") renderRehearsal(status.rehearsal || {});
  if (app.page === "rehearsal") renderLiveTeachingReference(status);
  renderMedia(status.media, observation);
  renderExpression(
    decision,
    expression,
    observation,
    tempoClock,
  );
  if (app.page === "system") renderDmx(status);
  if (app.page === "performance") renderEvents(status.events || []);
  if (["performance", "rig"].includes(app.page)) renderTargetSolutions(status.target_solutions || []);
  renderConnection(true);
  updateComponentStatuses();

  if (app.page === "performance") drawPerformanceRoom();
  if (app.page === "rig") drawRig();
  if (app.page === "audio") drawScope();
}

function dbfs(value) {
  const numeric = Math.max(0, Number(value) || 0);
  return Math.max(-120, 20 * Math.log10(Math.max(numeric, 0.000001)));
}

function renderAudioDiagnostics(audio = {}, engine = {}) {
  const metrics = audio.metrics || {};
  const state = audio.state || "inactive";
  const proof = $("audio-proof");
  if (proof) proof.dataset.state = state;
  setText("audio-proof-label", audio.label || "INPUT TEST NOT RUNNING");
  setText("audio-proof-detail", audio.detail || "No capture information is available.");
  setText("audio-dbfs", `${Number(metrics.dbfs ?? -120).toFixed(1)} dBFS`);
  setWidth("raw-level-bar", (Number(metrics.dbfs ?? -120) + 60) / 60);
  setText("pcm-packets", Number(audio.packets_received || 0).toLocaleString());
  setText("pcm-age", audio.last_packet_age_ms === null || audio.last_packet_age_ms === undefined
    ? "—"
    : audio.last_packet_age_ms < 1000
      ? `${Math.round(audio.last_packet_age_ms)} ms`
      : `${(audio.last_packet_age_ms / 1000).toFixed(1)} s`);
  setText("pcm-rate", `${Number(audio.packet_rate_hz || 0).toFixed(1)} /s`);
  setText("pcm-left", `${dbfs(metrics.channel_rms?.[0]).toFixed(1)} dB`);
  setText("pcm-right", `${dbfs(metrics.channel_rms?.[1]).toFixed(1)} dB`);
  setText("pcm-peak", `${Math.round(clamp(metrics.peak) * 100)}% / ${metrics.clipped_samples || 0}`);

  const stateLabel = {
    signal: "Signal",
    quiet: "PCM live",
    missing: "No line signal",
    clipping: "Clipping",
    waiting: "Opening input",
    stale: "Stalled",
    simulated: "Demo source",
    inactive: "Not running",
  }[state] || label(state);
  setText("audio-device-state", stateLabel);
  setText("remote-audio-state", stateLabel);
  setText(
    "remote-audio-level",
    ["signal", "quiet", "missing", "clipping"].includes(state)
      ? `${Number(metrics.dbfs ?? -120).toFixed(1)} dBFS · ${Number(audio.packets_received || 0).toLocaleString()} packets`
      : audio.detail || "Start Monitor on the console",
  );
  $("audio-device-state")?.classList.toggle("online", ["signal", "quiet"].includes(state));
  const testButton = $("audio-input-test-button");
  if (testButton) {
    testButton.textContent = engine.running && engine.mode === "monitor" ? "Stop input test" : "Start input test";
    testButton.disabled = Boolean(engine.running && engine.mode !== "monitor");
  }

  const packetCount = Number(audio.packets_received || 0);
  if (packetCount !== app.lastAudioPacketCount) {
    const heartbeat = $("pcm-heartbeat");
    heartbeat?.classList.remove("tick");
    void heartbeat?.offsetWidth;
    heartbeat?.classList.add("tick");
    app.lastAudioPacketCount = packetCount;
  }
}

function formatBytes(value) {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = bytes;
  let unit = -1;
  do {
    amount /= 1024;
    unit += 1;
  } while (amount >= 1024 && unit < units.length - 1);
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

function renderTrainingDataset(training = {}, engine = {}) {
  const current = training.current || {};
  const history = training.history || {};
  const state = current.state || (training.enabled ? "armed" : "disabled");
  const recording = Boolean(current.recording);
  setText("training-state", recording ? "RECORDING PCM" : label(state));
  setText(
    "training-detail",
    recording
      ? `${formatUptime(current.duration_s)} captured · ${current.codec || "PCM16 WAV"} · ${Number(current.feature_rate_hz || 10).toFixed(0)} semantic frames/s`
      : training.enabled
        ? "Armed for the next Monitor or Live session."
        : "Audio collection is disabled in settings.",
  );
  setText("training-duration", formatUptime(current.duration_s || 0));
  setText("training-segments", Number(current.segments ?? history.segments ?? 0).toLocaleString());
  setText("training-features", Number((history.feature_frames || 0) + (current.semantic_frames || 0)).toLocaleString());
  setText("training-feedback", Number((history.linked_feedback || 0) + (training.current_linked_feedback || 0)).toLocaleString());
  setText("training-annotations", Number((history.annotations || 0) + (training.current_annotations || 0)).toLocaleString());
  setText("training-musical-annotations", Number(history.musical_annotations || 0).toLocaleString());
  setText("training-consensus-anchors", Number(history.consensus_anchors || 0).toLocaleString());
  setText("training-structure-participants", Number(history.structure_participants || 0).toLocaleString());
  setText("training-size", formatBytes(training.total_bytes || history.bytes || 0));
  setText("training-free", formatBytes(training.disk_free_bytes || 0));
  setText("training-dropped", Number(current.dropped_frames || history.dropped_frames || 0).toLocaleString());
  setText("training-path", training.path || "state/training");
  setText("training-export-path", training.last_export || "No manifest built yet");
  const indicator = $("training-record-indicator");
  if (indicator) indicator.classList.toggle("recording", recording);
  const exportButton = $("training-export-button");
  if (exportButton) exportButton.disabled = recording || !Number(history.sessions || 0);
  const settingsButton = $("save-training-settings-button");
  if (settingsButton) settingsButton.disabled = Boolean(engine.running);
}

function renderResearch(research = {}) {
  const database = research.database || {};
  const training = research.training || {};
  const planned = Number(training.recordings_planned || 0);
  const processed = Number(training.recordings_processed || 0);
  const captured = Number(training.recordings_captured ?? planned);
  const eligible = Number(training.recordings_eligible ?? planned);
  const partial = Number(training.recordings_partial || 0);
  const unknown = Number(training.recordings_unknown || 0);
  const jobsComplete = Number(training.eligible_teacher_jobs_complete ?? training.teacher_jobs_complete ?? 0);
  const jobsTotal = Number(training.eligible_teacher_jobs ?? training.teacher_jobs_total ?? 0);
  const heldout = Number(training.split_counts?.validation || 0) + Number(training.split_counts?.test || 0);
  const model = training.model || {};
  const workerError = research.worker?.last_result?.error || "";
  const modelErrors = model.runtime_error ? 1 : 0;
  const errors = Number(training.teacher_errors?.length || 0)
    + Number(training.provenance_errors?.length || 0)
    + modelErrors
    + Number(Boolean(workerError));
  const captureDetails = [`${captured.toLocaleString()} total`];
  if (partial) captureDetails.push(`${partial.toLocaleString()} partial`);
  if (unknown) captureDetails.push(`${unknown.toLocaleString()} unidentified`);
  setText("research-captured", captureDetails.join(" · "));
  setText("research-recordings", `${processed.toLocaleString()} / ${eligible.toLocaleString()}`);
  setText("research-jobs", `${jobsComplete.toLocaleString()} / ${jobsTotal.toLocaleString()}`);
  setText("research-progress", `${Math.round(Number(training.progress || 0) * 100)}%`);
  setText("research-examples", Number(training.usable_examples || 0).toLocaleString());
  setText("research-heldout", heldout.toLocaleString());
  setText("research-errors", errors.toLocaleString());
  const evaluation = model.evaluation || {};
  const modelNotice = model.runtime_notice || "";
  const hasMetric = (value) => value !== null
    && value !== undefined
    && Number.isFinite(Number(value));
  const candidateIsCurrent = model.candidate_provenance_current !== false;
  const staleCandidate = Boolean(model.candidate && !candidateIsCurrent);
  const latestCandidateRejected = Boolean(model.candidate && candidateIsCurrent && evaluation.activated === false);
  setText(
    "research-student-state",
    model.runtime_state === "error"
      ? "Load error"
      : model.runtime_state === "obsolete" && latestCandidateRejected
        ? "Current candidate rejected"
      : model.runtime_state === "obsolete"
        ? "Previous model obsolete"
      : model.active && latestCandidateRejected
        ? "Active · latest candidate rejected"
        : model.active && staleCandidate
          ? "Active · retrain recommended"
        : model.active
          ? "Active"
          : latestCandidateRejected
            ? "Candidate rejected"
            : staleCandidate
              ? "Candidate stale · retrain"
            : model.candidate
              ? "Candidate awaiting validation"
              : "Awaiting training",
  );
  const blockers = training.blockers || [];
  const workerProgress = research.worker?.progress || {};
  const recoveredJobs = research.worker?.recovered_jobs || [];
  const recoveryNote = recoveredJobs.length
    ? ` Recovered ${recoveredJobs.length.toLocaleString()} interrupted job${recoveredJobs.length === 1 ? "" : "s"}; completed work was preserved.`
    : "";
  const preparationRunning = Boolean(research.preparation?.running);
  const cancelRequested = Boolean(research.worker?.cancel_requested);
  const readiness = $("research-readiness");
  if (readiness) {
    if (preparationRunning) {
      readiness.textContent = "Preparing the most recent capture: verifying continuity, song identity, and full-song eligibility before teacher jobs are queued.";
      readiness.className = "research-readiness active";
    } else if (research.worker?.running) {
      const done = Number(workerProgress.processed || 0);
      const batch = Number(workerProgress.planned || 0);
      const eta = training.estimated_remaining_seconds == null
        ? "estimating time after the first completion"
        : `about ${formatUptime(training.estimated_remaining_seconds)} remaining`;
      const jobName = label(workerProgress.current_job_type || "offline job");
      const resources = workerProgress.resources || {};
      const memoryStatus = Number(resources.rss_bytes || 0) > 0
        ? ` Memory ${formatBytes(resources.rss_bytes)}${Number(resources.memory_limit_bytes || 0) > 0 ? ` of ${formatBytes(resources.memory_limit_bytes)} limit` : ""}.`
        : "";
      readiness.textContent = research.worker?.externally_managed
        ? `Running ${jobName} in a separate local Lumen worker.${memoryStatus} This console will not start another analysis, training, or Live process while that worker owns the job.`
        : cancelRequested
        ? `Pausing ${jobName} at its cancellation checkpoint. Completed jobs are retained and the unfinished job will return to the queue.${recoveryNote}`
        : `Running ${jobName}: ${done} of ${batch} jobs finished, ${eta}.${memoryStatus} You may pause and resume without losing completed work.${recoveryNote}`;
      readiness.className = "research-readiness active";
    } else if (workerError) {
      readiness.textContent = `The analysis controller stopped unexpectedly: ${workerError}. The recording queue remains durable; press Analyze new recordings to retry after reviewing the error below.`;
      readiness.className = "research-readiness";
    } else if (latestCandidateRejected) {
      const reasons = evaluation.gate_reasons || [];
      const heldoutName = label(evaluation.held_out_split || "held-out");
      const heldout = evaluation.evaluation?.[evaluation.held_out_split || "test"] || {};
      const energy = heldout.energy || {};
      const functional = heldout.functional || {};
      const content = heldout.content || {};
      const boundary = heldout.boundary || {};
      const notApplicable = new Set(evaluation.not_applicable_axes || []);
      const metricDetails = [];
      if (hasMetric(energy.accuracy) && hasMetric(energy.majority_baseline)) {
        metricDetails.push(`energy ${(Number(energy.accuracy) * 100).toFixed(1)}% versus ${(Number(energy.majority_baseline) * 100 + 0.5).toFixed(1)}% required`);
      }
      if (hasMetric(energy.balanced_accuracy) && hasMetric(energy.balanced_baseline)) {
        const balancedRequired = Math.max(25, Number(energy.balanced_baseline) * 100 + 5);
        metricDetails.push(`energy balanced ${(Number(energy.balanced_accuracy) * 100).toFixed(1)}% versus ${balancedRequired.toFixed(1)}% required`);
      }
      if (hasMetric(content.accuracy) && hasMetric(content.majority_baseline)) {
        metricDetails.push(`content ${(Number(content.accuracy) * 100).toFixed(1)}% versus ${(Number(content.majority_baseline) * 100 + 0.5).toFixed(1)}% required`);
      }
      if (hasMetric(functional.accuracy) && hasMetric(functional.majority_baseline)) {
        metricDetails.push(`functional ${(Number(functional.accuracy) * 100).toFixed(1)}% versus ${(Number(functional.majority_baseline) * 100 + 0.5).toFixed(1)}% required`);
      } else if (notApplicable.has("functional")) {
        metricDetails.push("functional form was not present in this candidate's trusted teacher data");
      }
      const boundaryPrecision = boundary.event_precision ?? boundary.precision;
      const boundaryF1 = boundary.event_f1 ?? boundary.f1;
      if (hasMetric(boundaryPrecision) && hasMetric(boundaryF1)) {
        metricDetails.push(`boundary events ${(Number(boundaryPrecision) * 100).toFixed(1)}% precision / ${(Number(boundaryF1) * 100).toFixed(1)}% F1 at ±${Number(boundary.event_tolerance_ms || 0).toLocaleString()} ms`);
      }
      const resultDetail = metricDetails.length
        ? ` ${heldoutName} results: ${metricDetails.join("; ")}.`
        : reasons.length ? ` ${reasons.join("; ")}.` : "";
      readiness.textContent = model.active
        ? `The validated model remains active; the latest candidate was rejected.${resultDetail}`
        : `Training completed. The candidate was retained for diagnosis but not authorized for Live because held-out qualification did not pass.${resultDetail} Review or correct the held-out song timelines, then retrain after the trusted data changes; repeating training with identical inputs produces the same qualification evidence.`;
      if (modelNotice) readiness.textContent += ` ${modelNotice}`;
      readiness.textContent += recoveryNote;
      readiness.className = "research-readiness";
    } else if (staleCandidate) {
      readiness.textContent = training.activation_ready === false
        ? `The saved candidate predates the current qualification gate. Activation remains locked: ${(training.activation_blockers || []).join("; ")}. Collect complete, identified, previously unseen songs before the next qualifying train; a diagnostic pass may still be run now.`
        : "The saved candidate was trained from an older manifest or qualification gate. Train again so validation uses the current trusted examples, song splits, and event metrics.";
      readiness.textContent += recoveryNote;
      readiness.className = "research-readiness";
    } else if (training.train_ready) {
      readiness.textContent = training.activation_ready === false
        ? `Ready for a diagnostic training pass, but activation will remain locked: ${(training.activation_blockers || []).join("; ")}. Capture complete, identified, previously unseen songs to grow the final test population.`
        : training.collection_complete
        ? "Ready to train. Lumen has trusted training songs and separate held-out songs for validation."
        : `Ready for a preliminary train-and-validate pass. ${Number(training.teacher_jobs_remaining || 0)} teacher jobs remain; retrain after more analysis for broader musical coverage.`;
      readiness.textContent += recoveryNote;
      readiness.className = "research-readiness ready";
    } else {
      readiness.textContent = blockers.length ? `Not ready: ${blockers.join("; ")}.` : "Analyze captured songs to begin.";
      readiness.textContent += recoveryNote;
      readiness.className = "research-readiness";
    }
  }
  const songResults = $("research-song-results");
  const songResultsList = $("research-song-results-list");
  const evaluatedSongs = evaluation.song_evaluation?.[evaluation.held_out_split || "test"] || [];
  if (songResults && songResultsList) {
    songResults.classList.toggle("visible", evaluatedSongs.length > 0);
    if (latestCandidateRejected && evaluatedSongs.length) songResults.open = true;
    setText(
      "research-song-results-summary",
      `${evaluatedSongs.length.toLocaleString()} ${label(evaluation.held_out_split || "held-out")} songs · independent qualification details`,
    );
    songResultsList.innerHTML = evaluatedSongs.map((song) => {
      const metrics = song.metrics || {};
      const energyMetrics = metrics.energy || {};
      const boundaryMetrics = metrics.boundary || {};
      const artist = (song.artists || []).join(", ") || "Unknown artist";
      const energyAccuracy = hasMetric(energyMetrics.accuracy)
        ? `${(Number(energyMetrics.accuracy) * 100).toFixed(1)}% accuracy`
        : "no energy labels";
      const balanced = hasMetric(energyMetrics.balanced_accuracy)
        ? `${(Number(energyMetrics.balanced_accuracy) * 100).toFixed(1)}% balanced`
        : "balanced score unavailable";
      const eventF1 = hasMetric(boundaryMetrics.event_f1)
        ? `${(Number(boundaryMetrics.event_f1) * 100).toFixed(1)}% boundary-event F1`
        : "boundary score unavailable";
      return `<div class="research-song-result">
        <b>${escapeHtml(song.title || song.split_group_id || "Unidentified song")}</b>
        <span>${escapeHtml(artist)} · ${escapeHtml(label(song.review_status || "not reviewed"))}</span>
        <small>Energy ${energyAccuracy} · ${balanced} · ${eventF1} · ${Number(song.examples || 0).toLocaleString()} frames</small>
      </div>`;
    }).join("");
  }
  const errorList = $("research-error-list");
  if (errorList) {
    const items = [
      ...(training.teacher_errors || []),
      ...(training.provenance_errors || []),
      ...(workerError ? [{ job_type: "Analysis controller", error: workerError }] : []),
      ...(model.runtime_error ? [{ job_type: "Streaming student", error: model.runtime_error }] : []),
    ];
    errorList.innerHTML = items.slice(0, 8).map((item) =>
      `<div><b>${escapeHtml(item.job_type || item.teacher_run_id || "Analysis")}</b><span>${escapeHtml(item.error || "Unknown error")}</span></div>`
    ).join("");
    errorList.classList.toggle("visible", items.length > 0);
  }
  const labelBalance = $("research-label-balance");
  if (labelBalance) {
    const axes = training.label_balance || {};
    labelBalance.innerHTML = ["functional", "energy", "content"].map((axis) => {
      const labels = Object.entries(axes[axis] || {}).sort((a, b) => b[1] - a[1]);
      const content = labels.length
        ? labels.map(([name, count]) => `${escapeHtml(label(name))} ${Number(count).toLocaleString()}`).join(" · ")
        : "Awaiting teacher labels";
      return `<div><b>${escapeHtml(label(axis))}</b><span>${content}</span></div>`;
    }).join("");
  }
  const runButton = $("research-run-button");
  const engineRunning = Boolean(app.status?.engine?.running);
  const currentJobType = String(research.worker?.progress?.current_job_type || "");
  if (runButton) {
    runButton.disabled = engineRunning || preparationRunning || Boolean(research.worker?.running);
    runButton.textContent = preparationRunning
      ? "Preparing capture…"
      : research.worker?.running
        ? cancelRequested ? "Pausing analysis…" : "Analysis running…"
        : "Analyze new recordings";
    runButton.classList.toggle(
      "task-pending",
      preparationRunning || Boolean(research.worker?.running && currentJobType !== "student.train"),
    );
    runButton.setAttribute(
      "aria-busy",
      String(preparationRunning || Boolean(research.worker?.running && currentJobType !== "student.train")),
    );
  }
  const cancelButton = $("research-cancel-button");
  if (cancelButton) {
    const cancelSupported = research.worker?.cancel_supported !== false;
    cancelButton.disabled = !Boolean(research.worker?.running) || cancelRequested || !cancelSupported;
    cancelButton.textContent = !cancelSupported && research.worker?.running
      ? "Managed by local worker"
      : cancelRequested ? "Pausing…" : "Pause analysis";
  }
  const trainButton = $("research-train-button");
  if (trainButton) {
    trainButton.disabled = engineRunning || preparationRunning || Boolean(research.worker?.running) || !training.train_ready;
    const trainingNow = Boolean(research.worker?.running && currentJobType === "student.train");
    trainButton.textContent = trainingNow ? "Training and validating…" : "Train and validate";
    trainButton.classList.toggle("task-pending", trainingNow);
    trainButton.setAttribute("aria-busy", String(trainingNow));
  }
  if ($("research-import-button")) {
    $("research-import-button").disabled = engineRunning || preparationRunning || Boolean(research.worker?.running);
  }
  renderOperatorTask(research);
  const list = $("research-components");
  if (!list) return;
  list.innerHTML = (research.components || []).map((component) => {
    const annotation = component.annotations || {};
    const environment = component.environment_status || {};
    const state = annotation.state || component.source?.state || "unknown";
    const className = state === "ready" || state === "imported" || state === "not_applicable"
      ? ""
      : state.includes("awaiting") || state === "missing"
        ? "warn"
        : "error";
    const detail = annotation.message
      || (component.environment
        ? `Teacher environment: ${environment.state || "unknown"}`
        : component.role || "");
    return `<div class="research-component ${className}">
      <b>${escapeHtml(component.display_name || component.component_id)}</b>
      <span>${escapeHtml(label(state))}</span>
      <small>${escapeHtml(detail)}</small>
    </div>`;
  }).join("");
}

async function refreshResearch() {
  if (app.researchRefreshing) return;
  app.researchRefreshing = true;
  try {
    const research = await api("/api/research");
    app.bootstrap.research = research;
    renderResearch(research);
  } catch (error) {
    toast("Research status failed", error.message, "error");
  } finally {
    app.researchRefreshing = false;
  }
}

async function importResearchAnnotations() {
  const button = $("research-import-button");
  const task = beginOperatorTask(
    "Refreshing normalized research annotations",
    "Reading the installed label datasets and rebuilding their normalized local timelines.",
    button,
  );
  try {
    const result = await api("/api/research/import-annotations", {
      method: "POST",
      body: { components: ["edm98", "harmonix", "ccmusic", "salami"] },
    });
    await refreshResearch();
    const imported = (result.results || [])
      .filter((item) => item.state === "imported")
      .map((item) => `${item.component_id}: ${Number(item.timelines || 0).toLocaleString()}`)
      .join(" · ");
    toast("Annotations normalized", imported || "No importable annotations found.", "success");
  } catch (error) {
    toast("Annotation import failed", error.message, "error");
  } finally {
    finishOperatorTask(task);
    if (button) button.disabled = false;
  }
}

async function runResearchJob() {
  const button = $("research-run-button");
  if (button?.disabled) return;
  const task = beginOperatorTask(
    "Preparing recordings for analysis",
    "Checking new captures, reconstructing coherent songs, verifying checksums, and queuing EDMFormer.",
    button,
  );
  if (button) {
    button.textContent = "Preparing capture…";
  }
  try {
    const result = await api("/api/research/analyze", {
      method: "POST",
      body: {},
    });
    const research = result.research;
    app.bootstrap.research = research;
    renderResearch(research);
    if (result.started === false) {
      toast("Analysis already current", result.message || "No structure jobs are queued.", "success");
    } else {
      toast("Recording analysis started", "EDMFormer is running as a resumable offline batch.", "success");
    }
  } catch (error) {
    toast("Research job could not start", error.message, "error");
  } finally {
    await refreshResearch();
    finishOperatorTask(task);
    renderResearch(app.bootstrap?.research || {});
  }
}

async function cancelResearch() {
  const button = $("research-cancel-button");
  const task = beginOperatorTask(
    "Requesting an analysis pause",
    "The current model process will stop at its checkpoint and return unfinished work to the queue.",
    button,
  );
  try {
    const research = await api("/api/research/cancel", { method: "POST", body: {} });
    app.bootstrap.research = research;
    renderResearch(research);
    toast("Analysis pausing", "The current teacher process will stop and its job will return to the queue.", "success");
  } catch (error) {
    toast("Analysis could not pause", error.message, "error");
  } finally {
    finishOperatorTask(task);
    renderResearch(app.bootstrap?.research || {});
  }
}

async function trainStructureStudent() {
  const button = $("research-train-button");
  if (button?.disabled) return;
  const task = beginOperatorTask(
    "Starting neural training and validation",
    "Selecting trusted teacher examples, preserving held-out songs, and preparing the causal model.",
    button,
  );
  try {
    const result = await api("/api/research/train-student", {
      method: "POST",
      body: { epochs: 30 },
    });
    app.bootstrap.research = result.research;
    renderResearch(result.research);
    toast(
      "CPU student training started",
      `${Number(result.queued.examples || 0).toLocaleString()} aligned examples`,
      "success",
    );
  } catch (error) {
    toast("Student training could not start", error.message, "error");
  } finally {
    await refreshResearch();
    finishOperatorTask(task);
    renderResearch(app.bootstrap?.research || {});
  }
}

function formatUptime(seconds) {
  if (seconds === null || seconds === undefined) return "0:00";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function renderEngineButtons(engine) {
  const activeMode = engine.running ? engine.mode : null;
  $("monitor-button")?.classList.toggle("engaged", activeMode === "monitor");
  $("live-button")?.classList.toggle("engaged", activeMode === "live");
  $("demo-button")?.classList.toggle("engaged", activeMode === "demo");
  if ($("stop-button")) $("stop-button").disabled = !engine.running && engine.phase !== "fault";
}

function renderControls(controls) {
  $$("[data-control]").forEach((input) => {
    if (document.activeElement !== input && controls[input.dataset.control] !== undefined) {
      input.value = input.tagName === "SELECT"
        ? controls[input.dataset.control]
        : Math.round(Number(controls[input.dataset.control]) * 100);
    }
  });
  $$("[data-output]").forEach((output) => {
    if (controls[output.dataset.output] !== undefined) output.textContent = percent(controls[output.dataset.output]);
  });
  for (const id of ["blackout-button", "remote-blackout-button"]) {
    const button = $(id);
    if (button) button.setAttribute("aria-pressed", String(Boolean(controls.blackout)));
  }
}

function renderMedia(media, observation) {
  const title = media?.title || "Unidentified line input";
  const artists = media?.artists?.length ? media.artists.join(", ") : "Spotify metadata is optional";
  const source = media?.provider ? media.provider.toUpperCase() : "LINE-IN";
  const position = media?.live_position_ms;
  const duration = media?.duration_ms;
  const progress = duration ? clamp(position / duration) : 0;

  setText("song-title", title);
  setText("song-artist", artists);
  setText("nav-song-title", title);
  setText("nav-song-artist", artists);
  setText("remote-song-title", title);
  setText("remote-song-artist", artists);
  setText("media-source-badge", source);
  setText("remote-track-source", source);
  setText("track-context", media?.album || "Audio identity");
  setText("track-position", formatTime(position));
  setText("track-duration", formatTime(duration));
  setText("nav-position", formatTime(position));
  setText("nav-duration", formatTime(duration));
  setWidth("track-progress", progress);
  setWidth("nav-progress", progress);
  setWidth("remote-track-progress", progress);
  setText("remote-feedback-time", position !== null && position !== undefined ? `At ${formatTime(position)}` : "This moment");
  if (media?.provider === "spotify") {
    setText("spotify-position", formatTime(position));
    setText("spotify-duration", formatTime(duration));
    if ($("spotify-seek") && document.activeElement !== $("spotify-seek")) {
      $("spotify-seek").value = duration ? Math.round(clamp(position / duration) * 1000) : 0;
    }
  }

  const bpm = observation.bpm ? Number(observation.bpm).toFixed(1) : "—";
  const section = label(observation.section || "waiting");
  setText("fact-bpm", bpm);
  setText("fact-beat", percent(observation.beat_confidence));
  setText("fact-section", section);
  setText("remote-bpm", bpm);
  setText("remote-section", section.toUpperCase());
}

async function refreshSpotifyConsole(showErrors = false, query = null) {
  if (app.spotifyRefreshing) return;
  app.spotifyRefreshing = true;
  try {
    const requestedQuery = query === null ? (app.spotify?.query || "") : query;
    const parameters = new URLSearchParams();
    if (requestedQuery) parameters.set("q", requestedQuery);
    if (app.spotifyPlaylistId) parameters.set("playlist_id", app.spotifyPlaylistId);
    app.spotify = await api(`/api/spotify${parameters.size ? `?${parameters}` : ""}`);
    app.spotifyFetchedAt = Date.now();
    rememberSpotifyView();
    renderSpotifyConsole();
  } catch (error) {
    if (showErrors) toast("Spotify console unavailable", error.message, "error");
    const message = $("spotify-search-message");
    if (message) message.textContent = error.message;
  } finally {
    app.spotifyRefreshing = false;
  }
}

function rememberSpotifyView() {
  const view = { query: app.spotify?.query || "", playlistId: app.spotifyPlaylistId || "" };
  const current = app.spotifyHistory[app.spotifyHistoryIndex];
  if (current && current.query === view.query && current.playlistId === view.playlistId) return;
  app.spotifyHistory = app.spotifyHistory.slice(0, app.spotifyHistoryIndex + 1);
  app.spotifyHistory.push(view);
  app.spotifyHistoryIndex = app.spotifyHistory.length - 1;
}

function navigateSpotifyHistory(delta) {
  const next = app.spotifyHistoryIndex + delta;
  if (next < 0 || next >= app.spotifyHistory.length) return;
  app.spotifyHistoryIndex = next;
  const view = app.spotifyHistory[next];
  app.spotifyPlaylistId = view.playlistId;
  if ($("spotify-search-input")) $("spotify-search-input").value = view.query;
  refreshSpotifyConsole(true, view.query);
}

function renderSpotifyConsole() {
  const spotify = app.spotify || {
    connected: false,
    devices: [],
    playlists: [],
    playlist_tracks: [],
    results: [],
  };
  $("spotify-console-setup")?.classList.toggle("hidden", Boolean(spotify.connected));
  $("spotify-console-connected")?.classList.toggle("hidden", !spotify.connected);
  if (!spotify.connected) return;
  renderRemoteSpotify(spotify);

  const playback = spotify.playback || {};
  const track = playback.track || {};
  const profile = spotify.profile || {};
  const diagnostics = spotify.diagnostics || {};
  const activeDevice = playback.device || spotify.devices.find((device) => device.is_active) || null;
  setText("spotify-player-state", playback.is_playing ? "PLAYING" : track.name ? "PAUSED" : "IDLE");
  setText(
    "spotify-account-name",
    `${profile.display_name || profile.id || "Spotify account"}${profile.product ? ` · ${label(profile.product)}` : ""}`,
  );
  setText("spotify-track-title", track.name || "Choose music from Spotify");
  setText("spotify-track-artists", track.artists?.length ? track.artists.join(", ") : "Search below or use your usual Spotify app.");
  setText("spotify-album", track.album || "No active playback");
  setText("spotify-position", formatTime(playback.progress_ms));
  setText("spotify-duration", formatTime(track.duration_ms));
  setText("spotify-device-name", activeDevice?.name ? `${activeDevice.name} · active in Spotify` : "No active Spotify route");
  setText(
    "spotify-route-description",
    activeDevice?.name
      ? `Commands follow ${activeDevice.name}; Lumen does not select this computer automatically.`
      : "Start playback or choose Chromecast Audio in Spotify, then refresh.",
  );
  setText(
    "spotify-control-scope",
    spotify.control_authorized && spotify.library_authorized ? "ACCOUNT READY" : "RECONNECT REQUIRED",
  );

  const cover = $("spotify-cover-image");
  if (cover) {
    if (track.image_url) {
      cover.src = track.image_url;
      cover.alt = `${track.album || track.name || "Spotify"} cover`;
    } else {
      cover.removeAttribute("src");
      cover.alt = "";
    }
  }
  $("spotify-cover-placeholder")?.classList.toggle("hidden", Boolean(track.image_url));
  if ($("spotify-play-button")) $("spotify-play-button").textContent = playback.is_playing ? "❚❚" : "▶";
  if ($("spotify-seek") && document.activeElement !== $("spotify-seek")) {
    $("spotify-seek").value = track.duration_ms
      ? Math.round(clamp(Number(playback.progress_ms || 0) / Number(track.duration_ms)) * 1000)
      : 0;
  }

  const deviceSelect = $("spotify-device-select");
  if (deviceSelect) {
    if (
      app.spotifyTransferDeviceId
      && !spotify.devices.some((device) => device.id === app.spotifyTransferDeviceId)
    ) {
      app.spotifyTransferDeviceId = "";
    }
    deviceSelect.innerHTML = [
      `<option value="" ${app.spotifyTransferDeviceId ? "" : "selected"}>Follow Spotify active route${activeDevice?.name ? ` · ${escapeHtml(activeDevice.name)}` : ""}</option>`,
      ...spotify.devices.map(
        (device) => `<option value="${escapeHtml(device.id || "")}" ${device.id === app.spotifyTransferDeviceId ? "selected" : ""}>Transfer to ${escapeHtml(device.name || "Unnamed device")} · ${escapeHtml(device.type || "device")}${device.is_active ? " · currently active" : ""}</option>`,
      ),
    ].join("");
  }
  const selectedDevice = spotify.devices.find((device) => device.id === app.spotifyTransferDeviceId) || activeDevice;
  if ($("spotify-volume") && document.activeElement !== $("spotify-volume")) {
    $("spotify-volume").value = Number(selectedDevice?.volume_percent ?? 50);
    $("spotify-volume").disabled = !selectedDevice?.supports_volume;
  }

  $$(
    "#spotify-previous-button, #spotify-play-button, #spotify-next-button, #spotify-seek, #spotify-volume",
  ).forEach((control) => {
    if (control.id !== "spotify-volume" || selectedDevice?.supports_volume) {
      control.disabled = !spotify.control_authorized;
    }
  });
  if ($("spotify-transfer-button")) {
    $("spotify-transfer-button").disabled = !spotify.control_authorized || !app.spotifyTransferDeviceId;
  }

  $("spotify-library-auth")?.classList.toggle("hidden", Boolean(spotify.library_authorized));
  $("spotify-library-browser")?.classList.toggle("scope-missing", !spotify.library_authorized);
  const playlistSelect = $("spotify-playlist-select");
  if (playlistSelect) {
    playlistSelect.innerHTML = [
      `<option value="">Choose a playlist…</option>`,
      ...(spotify.playlists || []).map(
        (playlist) => `<option value="${escapeHtml(playlist.id || "")}" ${playlist.id === app.spotifyPlaylistId ? "selected" : ""}>${escapeHtml(playlist.name || "Untitled playlist")} · ${Number(playlist.track_count || 0)} tracks</option>`,
      ),
    ].join("");
    playlistSelect.disabled = !spotify.library_authorized;
  }
  const selectedPlaylist = (spotify.playlists || []).find(
    (playlist) => playlist.id === app.spotifyPlaylistId,
  ) || spotify.selected_playlist || null;
  if ($("spotify-play-playlist-button")) {
    $("spotify-play-playlist-button").disabled = !selectedPlaylist?.uri || !spotify.control_authorized;
  }
  const openPlaylistLink = $("spotify-open-playlist-link");
  if (openPlaylistLink) {
    openPlaylistLink.classList.toggle("hidden", !selectedPlaylist?.spotify_url);
    if (selectedPlaylist?.spotify_url) openPlaylistLink.href = selectedPlaylist.spotify_url;
  }

  const message = $("spotify-search-message");
  if (message) {
    if (!spotify.library_authorized) {
      message.textContent = "Reconnect once for playlist access. Track search and current-playback metadata remain available.";
    } else if (spotify.query) {
      message.textContent = `${spotify.results.length} result${spotify.results.length === 1 ? "" : "s"} for “${spotify.query}”. Playback follows Spotify's active device.`;
    } else if (app.spotifyPlaylistId && spotify.playlist_error) {
      message.textContent = `${spotify.playlist_error} Open this playlist in Spotify to browse it there.`;
    } else if (app.spotifyPlaylistId) {
      message.textContent = `${spotify.playlist_tracks.length} track${spotify.playlist_tracks.length === 1 ? "" : "s"} loaded from ${selectedPlaylist?.name || "the selected playlist"}.`;
    } else {
      message.textContent = `${spotify.playlists.length} playlist${spotify.playlists.length === 1 ? "" : "s"} available. Choose one or search for a song.`;
    }
  }
  const results = $("spotify-results");
  if (results) {
    const displayedTracks = spotify.query
      ? (spotify.results || [])
      : app.spotifyPlaylistId
        ? (spotify.playlist_tracks || [])
        : null;
    results.innerHTML = displayedTracks
      ? displayedTracks
        .map((item) => `<div class="spotify-result">
          ${item.image_url ? `<img src="${escapeHtml(item.image_url)}" alt="">` : `<span class="result-cover"></span>`}
          <div class="spotify-result-copy">
            <b>${escapeHtml(item.name || "Untitled")}${item.explicit ? " · E" : ""}</b>
            <span>${escapeHtml(item.artists?.join(", ") || "Unknown artist")}</span>
            <small>${escapeHtml(item.album || "")} · ${formatTime(item.duration_ms)}</small>
          </div>
          <div class="spotify-result-actions">
            <button data-spotify-play="${escapeHtml(item.uri || "")}" ${spotify.control_authorized ? "" : "disabled"}>Play</button>
            <button data-spotify-queue="${escapeHtml(item.uri || "")}" ${spotify.control_authorized ? "" : "disabled"}>Queue</button>
            ${item.spotify_url ? `<a href="${escapeHtml(item.spotify_url)}" target="_blank" rel="noreferrer">Spotify ↗</a>` : ""}
          </div>
        </div>`)
        .join("")
      : (spotify.playlists || [])
        .map((playlist) => `<div class="spotify-result">
          ${playlist.image_url ? `<img src="${escapeHtml(playlist.image_url)}" alt="">` : `<span class="result-cover"></span>`}
          <div class="spotify-result-copy">
            <b>${escapeHtml(playlist.name || "Untitled playlist")}</b>
            <span>${escapeHtml(playlist.owner || "Spotify playlist")}</span>
            <small>${Number(playlist.track_count || 0)} tracks</small>
          </div>
          <div class="spotify-result-actions">
            <button data-spotify-browse="${escapeHtml(playlist.id || "")}">Browse</button>
            <button data-spotify-context="${escapeHtml(playlist.uri || "")}" ${spotify.control_authorized ? "" : "disabled"}>Play</button>
            ${playlist.spotify_url ? `<a href="${escapeHtml(playlist.spotify_url)}" target="_blank" rel="noreferrer">Spotify ↗</a>` : ""}
          </div>
        </div>`)
        .join("");
    $$("[data-spotify-play]", results).forEach((button) => {
      button.addEventListener("click", () => spotifyCommand("play", app.spotifyPlaylistId && selectedPlaylist?.uri
        ? {
          context_uri: selectedPlaylist.uri,
          offset_uri: button.dataset.spotifyPlay,
        }
        : { uri: button.dataset.spotifyPlay }));
    });
    $$("[data-spotify-queue]", results).forEach((button) => {
      button.addEventListener("click", () => spotifyCommand("queue", { uri: button.dataset.spotifyQueue }));
    });
    $$("[data-spotify-context]", results).forEach((button) => {
      button.addEventListener("click", () => spotifyCommand("play", { context_uri: button.dataset.spotifyContext }));
    });
    $$("[data-spotify-browse]", results).forEach((button) => {
      button.addEventListener("click", () => {
        app.spotifyPlaylistId = button.dataset.spotifyBrowse || "";
        if ($("spotify-search-input")) $("spotify-search-input").value = "";
        app.spotify = { ...spotify, query: "" };
        refreshSpotifyConsole(true, "");
      });
    });
  }

  setText(
    "spotify-diagnostic-metadata",
    track.name
      ? `${track.artists?.join(", ") || "Unknown artist"} — ${track.name} · ${formatTime(playback.progress_ms)} / ${formatTime(track.duration_ms)}`
      : "Connected; Spotify reports no current track.",
  );
  setText(
    "spotify-diagnostic-devices",
    diagnostics.available_device_count
      ? `${diagnostics.available_device_count}: ${(diagnostics.available_device_names || []).join(", ")}`
      : "Spotify returned no API-controllable devices.",
  );
  setText("spotify-diagnostic-route", "Follow Spotify active device; no forced device ID");
  setText(
    "spotify-diagnostic-command",
    diagnostics.last_command
      ? `${diagnostics.last_command.ok ? "Accepted" : "Failed"} · ${label(diagnostics.last_command.action)} · ${diagnostics.last_command.message}`
      : "None sent since Lumen started.",
  );
  setText("spotify-diagnostic-note", diagnostics.api_note || "");
}

function renderRemoteSpotify(spotify) {
  const playback = spotify.playback || {};
  const track = playback.track || {};
  setText("remote-spotify-state", playback.is_playing ? "PLAYING" : track.name ? "PAUSED" : "IDLE");
  setText("remote-spotify-title", track.name || "No active Spotify track");
  setText("remote-spotify-artist", track.artists?.join(", ") || "Connect Spotify on the console");
  if ($("remote-spotify-play")) $("remote-spotify-play").textContent = playback.is_playing ? "❚❚" : "▶";
  const select = $("remote-spotify-playlist");
  if (select) {
    select.innerHTML = `<option value="">Choose playlist…</option>${(spotify.playlists || []).map((playlist) => `<option value="${escapeHtml(playlist.id || "")}" ${playlist.id === app.spotifyPlaylistId ? "selected" : ""}>${escapeHtml(playlist.name || "Untitled playlist")}</option>`).join("")}`;
    select.disabled = !spotify.library_authorized;
  }
  if ($("remote-spotify-message")) $("remote-spotify-message").textContent = spotify.message || (spotify.library_authorized ? "Choose a playlist or use the transport controls." : "Connect Spotify with playlist access on the desktop console.");
}

function selectedSpotifyDeviceId() {
  return app.spotifyTransferDeviceId || "";
}

async function spotifyCommand(action, values = {}) {
  try {
    await api("/api/spotify/control", {
      method: "POST",
      body: { action, device_id: selectedSpotifyDeviceId(), ...values },
    });
    if (action === "transfer") app.spotifyTransferDeviceId = "";
    window.setTimeout(() => refreshSpotifyConsole(false), 350);
  } catch (error) {
    toast("Spotify command failed", error.message, "error");
    window.setTimeout(() => refreshSpotifyConsole(false), 100);
  }
}

function renderExpression(decision, expression, observation, tempoClock = {}) {
  const gesture = decision ? label(decision.gesture) : "Standing by";
  const reason = decision?.reason || "Start Monitor, Perform, or Demo to begin interpretation.";
  const confidence = decision?.confidence || 0;
  setText("current-gesture", gesture);
  setText("decision-reason", reason);
  setText("expression-confidence", `${percent(confidence)} CONFIDENCE`);
  setText("audio-expression-gesture", gesture);
  setText("audio-expression-reason", reason);
  setText("audio-expression-confidence", `${percent(confidence)} CONFIDENCE`);
  setText("remote-gesture", gesture);
  setText("remote-reason", reason);
  setText("remote-confidence", percent(confidence));
  setText("remote-energy", percent(expression.energy));
  if ($("remote-energy-orb")) $("remote-energy-orb").style.setProperty("--energy", Math.round(expression.energy * 100));

  for (const name of ["energy", "tension", "motion", "intimacy"]) {
    setWidth(`meter-${name}`, expression[name]);
    setText(`value-${name}`, Number(expression[name]).toFixed(2));
    setWidth(`audio-meter-${name}`, expression[name]);
    setText(`audio-value-${name}`, Number(expression[name]).toFixed(2));
  }

  for (const [name, value] of [
    ["low", observation.low_energy],
    ["mid", observation.mid_energy],
    ["high", observation.high_energy],
    ["onset", observation.onset_strength],
  ]) {
    setText(`band-${name}`, Number(value || 0).toFixed(2));
    setWidth(`bar-${name}`, value || 0);
  }
  const bpm = observation.bpm ? Number(observation.bpm).toFixed(1) : "—";
  setText("beat-bpm-large", bpm);
  setText("beat-confidence-detail", percent(observation.beat_confidence));
  setText("beat-phase-detail", Number(observation.beat_phase || 0).toFixed(2));
  setText("novelty-detail", Number(observation.novelty || 0).toFixed(2));
  setText("beat-pulse-detail", percent(observation.beat_pulse || 0));
  const spectralTempo = tempoClock.spectral || {};
  const familyAmbiguous = Number(spectralTempo.family_ambiguity || 0) >= 0.62
    && !spectralTempo.octave_promoted;
  setText(
    "beat-lock-badge",
    familyAmbiguous
      ? "AMBIGUOUS"
      : observation.beat_confidence >= 0.5 ? "LOCKED" : "SEARCHING",
  );
  setText("beat-clock-source", label(tempoClock.source || "none"));
  setText(
    "beat-candidate-detail",
    Number(spectralTempo.candidate_bpm || 0) > 0
      ? `${Number(spectralTempo.candidate_bpm).toFixed(1)} BPM`
      : "—",
  );
  setText(
    "beat-spectral-lock",
    Number(spectralTempo.locked_bpm || 0) > 0
      ? `${Number(spectralTempo.locked_bpm).toFixed(1)} BPM`
      : "—",
  );
  setText(
    "beat-octave-detail",
    spectralTempo.octave_promoted ? "Fast pulse supported" : "Standard pulse",
  );
  const circumference = 2 * Math.PI * 66;
  const phase = clamp(observation.beat_phase);
  const previousBeatPhase = app.lastBeatPhase;
  if ($("beat-dial-progress")) $("beat-dial-progress").style.strokeDashoffset = String(circumference * (1 - phase));
  if (previousBeatPhase === null) {
    app.beatDisplayTurns = phase;
  } else {
    let delta = phase - previousBeatPhase;
    if (delta < -0.5) delta += 1;
    if (delta > 0.5) delta -= 1;
    app.beatDisplayTurns += delta;
  }
  app.lastBeatPhase = phase;
  if ($("beat-hand")) $("beat-hand").style.transform = `rotate(${app.beatDisplayTurns * 360}deg)`;
  const beatPulse = Number(observation.beat_pulse || 0);
  const beatArrived = (
    beatPulse >= 0.50
    && app.lastBeatPulse < 0.50
  ) || (
    observation.beat_confidence >= 0.35
    && previousBeatPhase !== null
    && phase < previousBeatPhase
  );
  if (beatArrived) {
    const lamp = $("beat-receive-lamp");
    lamp?.classList.remove("pulse");
    void lamp?.offsetWidth;
    lamp?.classList.add("pulse");
  }
  app.lastBeatPhase = phase;
  app.lastBeatPulse = beatPulse;
}

function renderDmx(status) {
  const map = new Map((status.dmx?.active_channels || []).map((item) => [item.channel, item.value]));
  $$(".dmx-cell").forEach((cell) => {
    const value = map.get(Number(cell.dataset.channel)) || 0;
    if (cell.dataset.value === String(value)) return;
    cell.dataset.value = String(value);
    cell.classList.toggle("active", value > 0);
    cell.style.setProperty("--dmx-value", value);
    cell.style.setProperty("--dmx-hue", 175 + (Number(cell.dataset.channel) % 5) * 17);
    cell.title = `Channel ${cell.dataset.channel}: ${value}`;
  });
  setText("dmx-frame-count", `${status.output?.frames_sent || 0} FRAMES`);
}

function renderEvents(events) {
  const container = $("event-log");
  if (!container) return;
  const signature = events.slice(0, 40).map((event) => `${event.time}|${event.kind}|${event.message}`).join("\n");
  if (container.dataset.signature === signature) return;
  container.dataset.signature = signature;
  if (!events.length) {
    container.innerHTML = `<div class="event-row"><time>--:--:--</time><em>system</em><span>No engine events yet.</span></div>`;
    return;
  }
  container.innerHTML = events
    .slice(0, 40)
    .map((event) => `<div class="event-row ${escapeHtml(event.kind)}"><time>${escapeHtml(event.time)}</time><em>${escapeHtml(event.kind)}</em><span>${escapeHtml(event.message)}</span></div>`)
    .join("");
}

function renderConnection(connected) {
  const element = $("remote-connection");
  if (!element) return;
  element.classList.toggle("online", connected);
  element.querySelector("span").textContent = connected ? "Connected" : "Reconnecting";
}

function updateComponentStatuses() {
  const engine = app.status?.engine;
  const system = app.system;
  if (!engine || !system) return;
  const audio = $("audio-status");
  const hasAudio = Boolean(system.audio?.cards?.length);
  const audioState = app.status?.audio?.state;
  if (audio) {
    const text = audioState === "signal" ? `${Number(app.status.audio.metrics?.dbfs ?? -120).toFixed(0)} dBFS`
      : audioState === "quiet" ? "PCM live"
      : audioState === "clipping" ? "Clipping"
      : audioState === "waiting" ? "Opening"
      : audioState === "stale" ? "Stalled"
      : hasAudio ? "Available" : "Not reported";
    audio.querySelector("b").textContent = text;
    setStatusClass(
      audio,
      audioState === "signal" || audioState === "quiet" ? "active"
        : audioState === "clipping" || audioState === "stale" ? "error"
        : hasAudio ? "ok" : "warn",
    );
  }
  const dmx = $("dmx-status");
  const hasDmx = Boolean(system.dmx?.ft232r_devices?.length && system.dmx?.native_driver_ready);
  if (dmx) {
    dmx.querySelector("b").textContent = engine.mode === "live" && engine.running ? "Transmitting" : hasDmx ? "Ready" : "Check device";
    setStatusClass(dmx, engine.mode === "live" && engine.running ? "active" : hasDmx ? "ok" : "warn");
  }
  const spotify = $("spotify-status");
  const spotifyReady = system.spotify?.client_id_configured && system.spotify?.token_present;
  if (spotify) {
    spotify.querySelector("b").textContent = app.status.media?.provider === "spotify" ? "Now playing" : spotifyReady ? "Connected" : "Optional";
    setStatusClass(spotify, app.status.media?.provider === "spotify" ? "active" : spotifyReady ? "ok" : "warn");
  }
}

function renderFixtureList(filter = "") {
  const container = $("fixture-list");
  if (!container) return;
  const normalized = filter.trim().toLowerCase();
  const items = fixtures().filter((fixture) => `${fixture.name} ${fixture.profile_key} ${fixture.address}`.toLowerCase().includes(normalized));
  container.innerHTML = items
    .map((fixture) => {
      const profile = profileFor(fixture.profile_key);
      return `<button class="fixture-row ${fixture.id === app.selectedFixtureId ? "active" : ""}" data-fixture-id="${escapeHtml(fixture.id)}">
        <span class="fixture-symbol ${fixture.kind === "auxiliary" ? "aux" : ""}"></span>
        <span><strong>${escapeHtml(fixture.name)}</strong><small>${escapeHtml(profile?.label || fixture.profile_key)}</small></span>
        <code>U${fixture.universe || 0}.${fixture.address}</code>
      </button>`;
    })
    .join("");
  $$("[data-fixture-id]", container).forEach((button) => {
    button.addEventListener("click", () => selectFixture(button.dataset.fixtureId));
  });
}

function selectFixture(id) {
  app.selectedFixtureId = id;
  renderFixtureList($("fixture-filter")?.value || "");
  const fixture = selectedFixture();
  if (!fixture) {
    $("empty-selection")?.classList.remove("hidden");
    $("fixture-form")?.classList.add("hidden");
    return;
  }
  $("empty-selection")?.classList.add("hidden");
  $("fixture-form")?.classList.remove("hidden");
  setText("selection-kind", fixture.kind === "moving" ? "MOVING HEAD" : "AUXILIARY");
  const profile = profileFor(fixture.profile_key);
  setText("fixture-profile-label", profile?.label || fixture.profile_key);
  $("fixture-name").value = fixture.name || fixture.id;
  setText("fixture-id", fixture.id);
  $("fixture-universe").value = fixture.universe || 0;
  $("fixture-address").value = fixture.address;
  ["x", "y", "z"].forEach((axis, index) => {
    $(`fixture-${axis}`).value = Number(fixture.position_m?.[index] || 0).toFixed(4);
  });
  ["rx", "ry", "rz"].forEach((axis, index) => {
    $(`fixture-${axis}`).value = Number(fixture.housing_rotation_deg?.[index] || 0).toFixed(2);
  });
  const calibrationFieldset = $("calibration-fieldset");
  calibrationFieldset?.classList.toggle("hidden", !fixture.calibration);
  if (fixture.calibration) {
    $$("[data-calibration]").forEach((input) => {
      const value = fixture.calibration[input.dataset.calibration];
      if (value !== undefined) input.value = value;
    });
    const points = calibrationPointsFor(fixture);
    Object.entries(points).forEach(([key, value]) => {
      const input = $(`[data-calibration="${key}"]`);
      if (input) input.value = value;
    });
    renderCalibrationPoints();
  }
  if (fixture.kind === "moving") {
    const points = calibrationPointsFor(fixture);
    const pan = points.home_pan_dmx;
    const tilt = points.home_tilt_dmx;
    if ($("calibration-pan")) $("calibration-pan").value = pan;
    if ($("calibration-tilt")) $("calibration-tilt").value = tilt;
    updateCalibrationJogReadouts();
  }
  const channelMap = $("fixture-channel-map");
  if (channelMap) {
    channelMap.innerHTML = Object.entries(profile?.channels || {})
      .map(([name, offset]) => `<span>${escapeHtml(name)} · ${Number(fixture.address) + Number(offset) - 1}</span>`)
      .join("");
  }
  drawRig();
}

function calibrationPointsFor(fixture) {
  const calibration = fixture?.calibration || {};
  const options = fixture?.source_metadata?.options || {};
  const coarse = (value, fallback) => Math.max(
    0,
    Math.min(255, Math.round(Number(value ?? fallback))),
  );
  const panMinimum = Number(calibration.pan_dmx_min_u16 ?? 0) / 257;
  const panMaximum = Number(calibration.pan_dmx_max_u16 ?? 65535) / 257;
  const tiltMinimum = Number(calibration.tilt_dmx_min_u16 ?? 0) / 257;
  const tiltMaximum = Number(calibration.tilt_dmx_max_u16 ?? 65535) / 257;
  const panReversed = Number(calibration.pan_direction ?? 1) < 0;
  const tiltReversed = Number(calibration.tilt_direction ?? 1) < 0;
  return {
    pan_left_dmx: coarse(
      calibration.pan_left_dmx ?? options.room_pan_left_dmx,
      panReversed ? panMaximum : panMinimum,
    ),
    pan_right_dmx: coarse(
      calibration.pan_right_dmx ?? options.room_pan_right_dmx,
      panReversed ? panMinimum : panMaximum,
    ),
    tilt_high_dmx: coarse(
      calibration.tilt_high_dmx ?? options.room_tilt_high_dmx,
      tiltReversed ? tiltMinimum : tiltMaximum,
    ),
    tilt_low_dmx: coarse(
      calibration.tilt_low_dmx ?? options.room_tilt_low_dmx,
      tiltReversed ? tiltMaximum : tiltMinimum,
    ),
    home_pan_dmx: coarse(
      calibration.home_pan_dmx ?? options.home_pan_dmx,
      (panMinimum + panMaximum) / 2,
    ),
    home_tilt_dmx: coarse(
      calibration.home_tilt_dmx ?? options.home_tilt_dmx,
      (tiltMinimum + tiltMaximum) / 2,
    ),
  };
}

function calibrationFieldNumber(name, fallback = 0) {
  return Number($(`[data-calibration="${name}"]`)?.value ?? fallback);
}

function fixtureTravelLabel(value) {
  const position = Math.max(0, Math.min(255, Math.round(Number(value))));
  return `${Math.round(position / 255 * 100)}%`;
}

function renderCalibrationPoints() {
  const bindings = {
    "capture-pan-left": "pan_left_dmx",
    "capture-pan-home": "home_pan_dmx",
    "capture-pan-right": "pan_right_dmx",
    "capture-tilt-high": "tilt_high_dmx",
    "capture-tilt-home": "home_tilt_dmx",
    "capture-tilt-low": "tilt_low_dmx",
  };
  Object.entries(bindings).forEach(([id, key]) => {
    setText(id, fixtureTravelLabel(calibrationFieldNumber(key, 128)));
  });
  const panDirection = calibrationFieldNumber("pan_direction", 1) < 0 ? "reversed" : "normal";
  const tiltDirection = calibrationFieldNumber("tilt_direction", 1) < 0 ? "reversed" : "normal";
  setText(
    "calibration-direction-note",
    `Software axis detection: pan ${panDirection}; tilt ${tiltDirection}. Reversed is valid and does not require turning the fixture.`,
  );
}

function updateCalibrationJogReadouts() {
  setText("calibration-pan-value", fixtureTravelLabel($("calibration-pan")?.value || 128));
  setText("calibration-tilt-value", fixtureTravelLabel($("calibration-tilt")?.value || 128));
  setText("calibration-speed-value", $("calibration-speed")?.value || 192);
}

function deriveCalibrationAxis(axis, previous) {
  const fixture = selectedFixture();
  const profile = profileFor(fixture?.profile_key);
  const totalDegrees = Number(axis === "pan" ? profile?.pan_degrees : profile?.tilt_degrees) || (axis === "pan" ? 540 : 270);
  const firstKey = axis === "pan" ? "pan_left_dmx" : "tilt_low_dmx";
  const secondKey = axis === "pan" ? "pan_right_dmx" : "tilt_high_dmx";
  const homeKey = axis === "pan" ? "home_pan_dmx" : "home_tilt_dmx";
  const endpointA = calibrationFieldNumber(firstKey, 0);
  const endpointB = calibrationFieldNumber(secondKey, 255);
  if (endpointA === endpointB) throw new Error(`${axis === "pan" ? "Left and right" : "High and low"} positions must differ.`);
  const rawMinimum = Math.min(endpointA, endpointB);
  const rawMaximum = Math.max(endpointA, endpointB);
  const direction = endpointB >= endpointA ? 1 : -1;
  const oldDirection = previous.direction || 1;
  const oldHomeAngle = previous.home / 255 * totalDegrees;
  const canonicalHome = (oldHomeAngle - previous.offset) / oldDirection;
  const newHomeAngle = calibrationFieldNumber(homeKey, 128) / 255 * totalDegrees;
  const set = (key, value) => { const input = $(`[data-calibration="${key}"]`); if (input) input.value = value; };
  set(`${axis}_dmx_min_u16`, Math.round(rawMinimum * 257));
  set(`${axis}_dmx_max_u16`, Math.round(rawMaximum * 257));
  set(`${axis}_min_deg`, rawMinimum / 255 * totalDegrees);
  set(`${axis}_max_deg`, rawMaximum / 255 * totalDegrees);
  set(`${axis}_direction`, direction);
  set(`${axis}_offset_deg`, newHomeAngle - direction * canonicalHome);
}

function captureCalibrationPoint(kind) {
  const axis = kind.startsWith("pan_") ? "pan" : "tilt";
  const previous = {
    direction: calibrationFieldNumber(`${axis}_direction`, 1),
    offset: calibrationFieldNumber(`${axis}_offset_deg`, 0),
    home: calibrationFieldNumber(axis === "pan" ? "home_pan_dmx" : "home_tilt_dmx", 128),
  };
  const value = Number($(axis === "pan" ? "calibration-pan" : "calibration-tilt").value);
  const fieldByKind = {
    pan_left: "pan_left_dmx",
    pan_home: "home_pan_dmx",
    pan_right: "pan_right_dmx",
    tilt_high: "tilt_high_dmx",
    tilt_home: "home_tilt_dmx",
    tilt_low: "tilt_low_dmx",
  };
  const field = $(`[data-calibration="${fieldByKind[kind]}"]`);
  if (!field) return;
  field.value = value;
  deriveCalibrationAxis(axis, previous);
  app.calibrationCaptures[kind] = value;
  renderCalibrationPoints();
}

function selectedMoverId() {
  const fixture = selectedFixture();
  return fixture?.kind === "moving" ? fixture.id : fixtures().find((item) => item.kind === "moving")?.id;
}

async function sendCalibration(active, values = {}) {
  const fixtureId = selectedMoverId();
  if (!fixtureId) return;
  try {
    if (active && !app.status?.engine?.running) {
      app.status = await api("/api/engine/start", { method: "POST", body: { mode: "live" } });
    }
    const position = active ? {
      pan_dmx: Number($("calibration-pan")?.value || 128),
      tilt_dmx: Number($("calibration-tilt")?.value || 128),
      speed: Number($("calibration-speed")?.value || 192),
    } : {};
    await api("/api/calibration", { method: "POST", body: { fixture_id: fixtureId, active, ...position, ...values } });
    app.calibrationActive = active;
    setText("calibration-toggle", active ? "Stop calibration" : "Start calibration");
    ["calibration-pan", "calibration-tilt", "calibration-speed"].forEach((id) => { if ($(id)) $(id).disabled = !active; });
    $$('[data-calibration-capture]').forEach((button) => { button.disabled = !active; });
  } catch (error) { toast("Calibration unavailable", error.message, "error"); }
}

function calibrationJog() {
  if (!app.calibrationActive) return;
  sendCalibration(true, {
    pan_dmx: Number($("calibration-pan")?.value || 128),
    tilt_dmx: Number($("calibration-tilt")?.value || 128),
    speed: Number($("calibration-speed")?.value || 192),
  });
}

function installFeedbackButton(button) {
  button.dataset.manualConfirmation = "true";
  let start = null;
  let cancelled = false;
  let touchArmed = false;
  button.addEventListener("pointerdown", (event) => {
    start = { x: event.clientX, y: event.clientY, pointerType: event.pointerType };
    cancelled = event.pointerType === "touch" && Date.now() < app.touchBlockedUntil;
    touchArmed = false;
  });
  button.addEventListener("pointermove", (event) => {
    if (!start) return;
    const distance = Math.hypot(event.clientX - start.x, event.clientY - start.y);
    if (distance > 12) cancelled = true;
  });
  button.addEventListener("pointercancel", () => { cancelled = true; touchArmed = false; });
  button.addEventListener("pointerup", () => {
    if (start?.pointerType === "touch") touchArmed = !cancelled;
  });
  button.addEventListener("click", () => {
    // Mobile browsers can synthesize a click after a scroll/wake gesture.
    // A touch feedback click is accepted only when its pointer stayed on the
    // button; mouse/keyboard clicks remain immediate on the desktop.
    if (start?.pointerType === "touch" && !touchArmed) return;
    touchArmed = false;
    confirmButtonPress(button);
    const surface = button.closest(".remote-section") ? "remote" : "desktop";
    sendFeedback(button.dataset.feedback, button.dataset.value, null, surface);
  });
}

function installWakeProtectedAction(button, action) {
  if (!button) return;
  button.dataset.manualConfirmation = "true";
  let start = null;
  let allowed = false;
  button.addEventListener("pointerdown", (event) => {
    start = { x: event.clientX, y: event.clientY, type: event.pointerType };
    allowed = event.pointerType !== "touch" || Date.now() >= app.touchBlockedUntil;
  });
  button.addEventListener("pointermove", (event) => {
    if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) > 12) {
      allowed = false;
    }
  });
  button.addEventListener("pointercancel", () => { allowed = false; });
  button.addEventListener("click", (event) => {
    if (start?.type === "touch" && !allowed) {
      event.preventDefault();
      return;
    }
    allowed = false;
    confirmButtonPress(button);
    action(event);
  });
}

async function saveSelectedFixture() {
  const fixture = selectedFixture();
  if (!fixture) {
    toast("No fixture selected", "Choose a fixture from the inventory first.", "error");
    return;
  }
  const payload = {
    fixture_id: fixture.id,
    name: $("fixture-name").value,
    universe: Number($("fixture-universe").value),
    address: Number($("fixture-address").value),
    position_m: ["x", "y", "z"].map((axis) => Number($(`fixture-${axis}`).value)),
    housing_rotation_deg: ["rx", "ry", "rz"].map((axis) => Number($(`fixture-${axis}`).value)),
  };
  if (fixture.calibration) {
    payload.calibration = {};
    $$("[data-calibration]").forEach((input) => {
      payload.calibration[input.dataset.calibration] = Number(input.value);
    });
  }
  setText("rig-save-state", "Saving…");
  const resumeMode = app.status?.engine?.running
    ? (app.status.engine.mode || "live")
    : null;
  try {
    if (app.calibrationActive) await sendCalibration(false);
    if (app.status?.engine?.running) {
      app.status = await api("/api/engine/stop", { method: "POST", body: {} });
      const deadline = Date.now() + 8000;
      while (app.status?.engine?.running && Date.now() < deadline) {
        await new Promise((resolve) => window.setTimeout(resolve, 100));
        app.status = await api("/api/status");
      }
      if (app.status?.engine?.running) {
        throw new Error("The engine did not finish stopping in time to save the rig.");
      }
    }
    app.bootstrap = await api("/api/rig/fixture", { method: "POST", body: payload });
    app.status = app.bootstrap.status;
    app.memory = app.bootstrap.memory;
    app.system = app.bootstrap.system;
    if (resumeMode) {
      app.status = await api("/api/engine/start", { method: "POST", body: { mode: resumeMode } });
    }
    setText("rig-save-state", "Saved to active rig");
    renderFixtureList($("fixture-filter")?.value || "");
    selectFixture(fixture.id);
    toast("Fixture saved", `${payload.name} is now part of the active Lumen rig.`, "success");
  } catch (error) {
    if (resumeMode && !app.status?.engine?.running) {
      try {
        app.status = await api("/api/engine/start", { method: "POST", body: { mode: resumeMode } });
      } catch (_resumeError) { /* The original save error is the useful message. */ }
    }
    setText("rig-save-state", "Save failed");
    toast("Could not save fixture", error.message, "error");
  }
}

async function solveTargetFromFields() {
  const body = {
    x: Number($("target-x").value),
    y: Number($("target-y").value),
    z: Number($("target-z").value),
  };
  try {
    const result = await api("/api/target", { method: "POST", body });
    if (app.status) {
      app.status.selected_target = result.target;
      app.status.target_solutions = result.solutions;
    }
    renderTargetSolutions(result.solutions);
    drawRig();
  } catch (error) {
    toast("Target could not be solved", error.message, "error");
  }
}

function renderTargetSolutions(solutions) {
  const container = $("solution-strip");
  if (!container) return;
  container.innerHTML = solutions
    .map((solution) => solution.reachable
      ? `<div class="solution-card"><b>${escapeHtml(solution.fixture_name || shortFixtureName(solution.fixture_id))}</b><span>pan ${Number(solution.pan_deg).toFixed(2)}° · tilt ${Number(solution.tilt_deg).toFixed(2)}°</span><code>${Number(solution.aim_error_deg).toFixed(3)}° err</code></div>`
      : `<div class="solution-card unreachable"><b>${escapeHtml(solution.fixture_name || shortFixtureName(solution.fixture_id))}</b><span>${escapeHtml(solution.error || "Target unreachable")}</span><code>NO SOLVE</code></div>`)
    .join("");
}

function shortFixtureName(id) {
  const fixture = fixtures().find((item) => item.id === id);
  return fixture?.name || String(id).slice(0, 8);
}

function configureCanvas(canvas) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return null;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * ratio);
  const height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function roomProjection(canvasSize, view, position) {
  const room = app.bootstrap.rig.room;
  const pad = 38;
  const availableWidth = Math.max(10, canvasSize.width - pad * 2);
  const availableHeight = Math.max(10, canvasSize.height - pad * 2);
  let aRange;
  let bRange;
  let a;
  let b;
  if (view === "3d") {
    const camera = app.roomCamera;
    const scale = Math.min(availableWidth / room.width_m, availableHeight / room.height_m) * 0.78 * camera.zoom;
    let [px, py, pz] = position;
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    [px, py] = [px * cy - py * sy, px * sy + py * cy];
    const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    [py, pz] = [py * cp - pz * sp, py * sp + pz * cp];
    const cx = canvasSize.width / 2;
    const floor = canvasSize.height * 0.80;
    const x = cx + px * scale + camera.panX;
    const y = floor - pz * scale + py * scale * 0.32 + camera.panY;
    return { x, y, scale, rect: { x: pad, y: pad, width: availableWidth, height: availableHeight } };
  } else if (view === "front") {
    aRange = room.width_m;
    bRange = room.height_m;
    a = position[0] + room.width_m / 2;
    b = position[2];
  } else if (view === "side") {
    aRange = room.depth_m;
    bRange = room.height_m;
    a = position[1] + room.depth_m / 2;
    b = position[2];
  } else {
    aRange = room.width_m;
    bRange = room.depth_m;
    a = position[0] + room.width_m / 2;
    b = position[1] + room.depth_m / 2;
  }
  const scale = Math.min(availableWidth / aRange, availableHeight / bRange);
  const projectedWidth = aRange * scale;
  const projectedHeight = bRange * scale;
  const originX = (canvasSize.width - projectedWidth) / 2;
  const originY = (canvasSize.height - projectedHeight) / 2;
  return {
    x: originX + a * scale,
    y: originY + projectedHeight - b * scale,
    scale,
    rect: { x: originX, y: originY, width: projectedWidth, height: projectedHeight },
  };
}

function roomPositionFromCanvas(canvas, view, clientX, clientY, z = 1.2) {
  const rect = canvas.getBoundingClientRect();
  const room = app.bootstrap.rig.room;
  const projection = roomProjection({ width: rect.width, height: rect.height }, view, [0, 0, 0]);
  const normalizedA = clamp((clientX - rect.left - projection.rect.x) / projection.rect.width);
  const normalizedB = clamp(1 - (clientY - rect.top - projection.rect.y) / projection.rect.height);
  if (view === "front") return [(normalizedA - 0.5) * room.width_m, 0, normalizedB * room.height_m];
  if (view === "side") return [0, (normalizedA - 0.5) * room.depth_m, normalizedB * room.height_m];
  return [(normalizedA - 0.5) * room.width_m, (normalizedB - 0.5) * room.depth_m, z];
}

function drawRoom(canvas, view, interactive = false) {
  const configured = configureCanvas(canvas);
  if (!configured || !app.bootstrap || !app.status) return;
  const { context: ctx, width, height } = configured;
  ctx.clearRect(0, 0, width, height);
  if (view === "3d") drawRoom3d(ctx, width, height, interactive);
  else drawRoomOrthographic(ctx, width, height, view, interactive);
}

function drawRoomOrthographic(ctx, width, height, view, interactive = false) {
  const boundary = roomProjection({ width, height }, view, [0, 0, 0]);
  ctx.save();
  ctx.strokeStyle = "rgba(102, 163, 160, .35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(boundary.rect.x, boundary.rect.y, boundary.rect.width, boundary.rect.height);
  ctx.setLineDash([4, 5]);
  ctx.strokeStyle = "rgba(102, 163, 160, .12)";
  for (let i = 1; i < 10; i += 1) {
    const x = boundary.rect.x + boundary.rect.width * i / 10;
    const y = boundary.rect.y + boundary.rect.height * i / 10;
    ctx.beginPath(); ctx.moveTo(x, boundary.rect.y); ctx.lineTo(x, boundary.rect.y + boundary.rect.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(boundary.rect.x, y); ctx.lineTo(boundary.rect.x + boundary.rect.width, y); ctx.stroke();
  }
  ctx.setLineDash([]);

  const liveSolutionMap = new Map((app.status.solutions || []).map((solution) => [solution.fixture_id, solution]));
  const target = app.status.selected_target || { x: 0, y: 0, z: 1.2 };
  const targetPoint = roomProjection({ width, height }, view, [target.x, target.y, target.z]);
  const resolvedTargetByFixture = new Map();
  for (const solution of app.status.solutions || []) {
    resolvedTargetByFixture.set(solution.fixture_id, solution.target || target);
  }

  for (const fixture of fixtures()) {
    if (fixture.kind !== "moving") continue;
    const fixturePoint = roomProjection({ width, height }, view, fixture.position_m);
    const solution = liveSolutionMap.get(fixture.id);
    const beamTarget = solution?.target || resolvedTargetByFixture.get(fixture.id) || target;
    const beamPoint = roomProjection({ width, height }, view, [beamTarget.x, beamTarget.y, beamTarget.z]);
    const gradient = ctx.createLinearGradient(fixturePoint.x, fixturePoint.y, beamPoint.x, beamPoint.y);
    gradient.addColorStop(0, "rgba(102, 220, 211, .65)");
    gradient.addColorStop(1, "rgba(77, 130, 171, .12)");
    ctx.strokeStyle = gradient;
    ctx.lineWidth = solution ? 1.8 : 1;
    ctx.beginPath();
    ctx.moveTo(fixturePoint.x, fixturePoint.y);
    ctx.lineTo(beamPoint.x, beamPoint.y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#e2b464";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(targetPoint.x, targetPoint.y, 8, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(targetPoint.x - 12, targetPoint.y);
  ctx.lineTo(targetPoint.x + 12, targetPoint.y);
  ctx.moveTo(targetPoint.x, targetPoint.y - 12);
  ctx.lineTo(targetPoint.x, targetPoint.y + 12);
  ctx.stroke();

  for (const fixture of fixtures()) {
    const point = roomProjection({ width, height }, view, fixture.position_m);
    const selected = fixture.id === app.selectedFixtureId;
    ctx.save();
    ctx.translate(point.x, point.y);
    if (selected) {
      ctx.strokeStyle = "#fff0a9";
      ctx.lineWidth = 1;
      ctx.strokeRect(-10, -10, 20, 20);
    }
    ctx.fillStyle = fixture.kind === "moving" ? "#68d7cf" : "#d8a35e";
    ctx.strokeStyle = "#0a1113";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (fixture.kind === "moving") {
      ctx.arc(0, 0, selected ? 6 : 5, 0, Math.PI * 2);
    } else {
      ctx.rect(-5, -5, 10, 10);
    }
    ctx.fill();
    ctx.stroke();
    if (interactive || selected) {
      ctx.fillStyle = selected ? "#fff1b0" : "#829793";
      ctx.font = "8px DejaVu Sans";
      ctx.fillText(`${fixture.name.slice(0, 18)} · ${fixture.address}`, 10, 3);
    }
    ctx.restore();
  }

  ctx.fillStyle = "rgba(125, 153, 150, .72)";
  ctx.font = "8px DejaVu Sans Mono";
  const room = app.bootstrap.rig.room;
  const dimensions = view === "plan"
    ? `${room.width_m.toFixed(2)} m × ${room.depth_m.toFixed(2)} m`
    : view === "front"
      ? `${room.width_m.toFixed(2)} m × ${room.height_m.toFixed(2)} m`
      : `${room.depth_m.toFixed(2)} m × ${room.height_m.toFixed(2)} m`;
  ctx.fillText(`${view.toUpperCase()} · ${dimensions}`, boundary.rect.x + 5, boundary.rect.y + 13);
  ctx.restore();
}

function drawRoom3d(ctx, width, height, interactive = false) {
  const room = app.bootstrap.rig.room;
  const project = (p) => roomProjection({ width, height }, "3d", [p.x ?? p[0], p.y ?? p[1], p.z ?? p[2]]);
  const corner = (x, y, z) => project([x, y, z]);
  const w = room.width_m / 2, d = room.depth_m / 2, h = room.height_m;
  const floor = [corner(-w, -d, 0), corner(w, -d, 0), corner(w, d, 0), corner(-w, d, 0)];
  const ceiling = [corner(-w, -d, h), corner(w, -d, h), corner(w, d, h), corner(-w, d, h)];
  const line = (a, b, color = "rgba(102,163,160,.30)", dash = []) => {
    ctx.save(); ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.beginPath();
    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.restore();
  };
  ctx.fillStyle = "rgba(25,48,51,.45)"; ctx.beginPath(); ctx.moveTo(floor[0].x, floor[0].y);
  floor.slice(1).forEach((p) => ctx.lineTo(p.x, p.y)); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "rgba(38,67,69,.18)"; ctx.beginPath(); ctx.moveTo(floor[2].x, floor[2].y); ctx.lineTo(floor[3].x, floor[3].y); ctx.lineTo(ceiling[3].x, ceiling[3].y); ctx.lineTo(ceiling[2].x, ceiling[2].y); ctx.closePath(); ctx.fill();
  for (let i = 0; i < 4; i += 1) { line(floor[i], floor[(i + 1) % 4]); line(ceiling[i], ceiling[(i + 1) % 4], "rgba(102,163,160,.16)"); line(floor[i], ceiling[i], "rgba(102,163,160,.22)"); }
  for (let i = 1; i < 6; i += 1) {
    const x = -w + (2 * w * i / 6); line(corner(x, -d, 0), corner(x, d, 0), "rgba(102,163,160,.12)", [3, 5]);
    const y = -d + (2 * d * i / 6); line(corner(-w, y, 0), corner(w, y, 0), "rgba(102,163,160,.12)", [3, 5]);
  }
  const live = new Map((app.status.solutions || []).map((s) => [s.fixture_id, s]));
  for (const fixture of fixtures()) {
    if (fixture.kind !== "moving") continue;
    const fp = project(fixture.position_m); const target = live.get(fixture.id)?.target || app.status.selected_target || { x: 0, y: 0, z: 1.2 }; const tp = project(target);
    ctx.save();
    const beam = ctx.createLinearGradient(fp.x, fp.y, tp.x, tp.y);
    beam.addColorStop(0, "rgba(130,245,230,.72)"); beam.addColorStop(1, "rgba(64,125,139,.08)");
    ctx.strokeStyle = beam; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(fp.x, fp.y); ctx.lineTo(tp.x, tp.y); ctx.stroke(); ctx.restore();
  }
  const target = project(app.status.selected_target || { x: 0, y: 0, z: 1.2 });
  ctx.strokeStyle = "#e2b464"; ctx.beginPath(); ctx.arc(target.x, target.y, 7, 0, Math.PI * 2); ctx.stroke();
  for (const fixture of fixtures()) {
    const p = project(fixture.position_m); const selected = fixture.id === app.selectedFixtureId;
    ctx.fillStyle = fixture.kind === "moving" ? "#68d7cf" : "#d8a35e"; ctx.beginPath(); ctx.arc(p.x, p.y, selected ? 6 : 4, 0, Math.PI * 2); ctx.fill();
    if (interactive || selected) { ctx.fillStyle = selected ? "#fff1b0" : "#829793"; ctx.font = "8px DejaVu Sans"; ctx.fillText(`${fixture.name.slice(0, 18)} · ${fixture.address}`, p.x + 9, p.y + 3); }
  }
  ctx.fillStyle = "rgba(125,153,150,.72)"; ctx.font = "8px DejaVu Sans Mono"; ctx.fillText(`3D · ${room.width_m.toFixed(2)} × ${room.depth_m.toFixed(2)} × ${room.height_m.toFixed(2)} m`, 43, 51);
}

function drawPerformanceRoom() {
  drawRoom($("performance-room-canvas"), app.roomView, false);
}

function drawRig() {
  drawRoom($("rig-canvas"), app.rigView, true);
}

function drawScope() {
  const configured = configureCanvas($("scope-canvas"));
  if (!configured) return;
  const { context: ctx, width, height } = configured;
  ctx.clearRect(0, 0, width, height);
  const values = app.status?.analysis_history || [];
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(100, 222, 213, .85)");
  gradient.addColorStop(1, "rgba(69, 126, 163, .42)");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1.4;
  ctx.shadowColor = "rgba(100, 222, 213, .4)";
  ctx.shadowBlur = 4;
  ctx.beginPath();
  if (!values.length) {
    ctx.moveTo(0, height * 0.5);
    ctx.lineTo(width, height * 0.5);
  }
  values.forEach((sample, index) => {
    const value = clamp((Number(sample.dbfs) + 60) / 60);
    const x = index / Math.max(1, values.length - 1) * width;
    const y = height - value * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(208, 163, 92, .78)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  values.forEach((sample, index) => {
    const x = index / Math.max(1, values.length - 1) * width;
    const y = height - clamp(sample.energy) * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "rgba(100, 222, 213, .9)";
  ctx.font = "8px DejaVu Sans Mono";
  ctx.fillText("INPUT dBFS", 10, 13);
  ctx.fillStyle = "rgba(208, 163, 92, .9)";
  ctx.fillText("EXPRESSION ENERGY", 88, 13);
  const observation = app.status?.observation || {};
  const bands = [
    ["LOW", observation.low_energy || 0, "#55c4bb"],
    ["MID", observation.mid_energy || 0, "#6e9fc1"],
    ["HIGH", observation.high_energy || 0, "#d0a35c"],
  ];
  bands.forEach(([name, value, color], index) => {
    const x = 20 + index * (width - 40) / 3;
    const barWidth = (width - 60) / 3;
    const barHeight = clamp(value) * height * 0.34;
    ctx.fillStyle = `${color}22`;
    ctx.fillRect(x, height - 18 - barHeight, barWidth, barHeight);
    ctx.strokeStyle = `${color}88`;
    ctx.strokeRect(x, height - 18 - barHeight, barWidth, barHeight);
    ctx.fillStyle = `${color}aa`;
    ctx.font = "7px DejaVu Sans Mono";
    ctx.fillText(name, x + 3, height - 6);
  });
}

function buildDmxHeatmap() {
  const container = $("dmx-heatmap");
  if (!container || container.children.length) return;
  const fragment = document.createDocumentFragment();
  for (let channel = 1; channel <= 512; channel += 1) {
    const cell = document.createElement("i");
    cell.className = "dmx-cell";
    cell.dataset.channel = channel;
    cell.title = `Channel ${channel}: 0`;
    fragment.append(cell);
  }
  container.append(fragment);
}

function renderMemory(memory) {
  if (!memory) return;
  setText("memory-song-count", memory.totals?.songs || 0);
  setText("memory-feedback-count", memory.totals?.feedback || 0);
  setText("memory-decision-count", memory.totals?.decisions || 0);
  renderSongLibrary();
  const history = $("feedback-history");
  if (history) {
    history.innerHTML = memory.recent_feedback?.length
      ? memory.recent_feedback.map((item) => `<article class="feedback-history-item">
          <div><b>${escapeHtml(label(item.label))}</b><time>${escapeHtml(formatElapsed(item.created_unix_ms))} <button class="feedback-undo" data-delete-feedback="${item.id}">Undo</button></time></div>
          <span>${escapeHtml(item.song_title || "Unidentified session")} · ${formatTime(item.position_ms)}${item.gesture ? ` · ${escapeHtml(item.gesture)}` : ""}${item.section ? ` · ${escapeHtml(item.section)}` : ""}</span>
          ${item.note ? `<p>${escapeHtml(item.note)}</p>` : ""}
        </article>`).join("")
      : `<div class="empty-selection"><b>No feedback yet</b><p>Use the performance console or phone remote to teach Lumen.</p></div>`;
  }
}

async function deleteFeedback(feedbackId, participantScoped = false) {
  try {
    await api("/api/feedback/delete", {
      method: "POST",
      body: {
        feedback_id: Number(feedbackId),
        participant_id: participantScoped ? app.participantId : null,
      },
    });
    app.memory = await api("/api/memory");
    renderMemory(app.memory);
    toast("Feedback removed", "Its preference weight was recomputed.", "success");
  } catch (error) { toast("Feedback could not be removed", error.message, "error"); }
}

function renderSongLibrary(filter = "") {
  const container = $("song-library");
  if (!container || !app.memory) return;
  const normalized = filter.trim().toLowerCase();
  const songs = (app.memory.songs || []).filter((song) => `${song.title} ${(song.artists || []).join(" ")} ${song.provider}`.toLowerCase().includes(normalized));
  container.innerHTML = songs.length
    ? songs.map((song) => `<div class="song-row">
        <div><strong>${escapeHtml(song.title || "Untitled recording")}</strong><small>${escapeHtml((song.artists || []).join(", ") || song.album || "No artist identity")}</small></div>
        <span class="provider-pill">${escapeHtml(song.provider)}</span>
        <span>${Number(song.play_count || 0)}</span>
        <span>${Number(song.feedback_count || 0)}</span>
        <span>${escapeHtml(formatElapsed(song.last_seen_unix_ms))}</span>
      </div>`).join("")
    : `<div class="empty-selection"><b>No recordings yet</b><p>Identified songs and line-in sessions will appear here.</p></div>`;
}

function renderSystem(system) {
  if (!system) return;
  const audioReport = system.audio?.report || "No ALSA capture report was returned.";
  setText("audio-hardware-report", audioReport);
  const addresses = system.network?.addresses || [];
  const remoteAddress = addresses.length
    ? `http://${addresses[0]}:${window.location.port || "4042"}/remote`
    : `http://<${system.network?.host_name || "lumen-pc"}-ip>:${window.location.port || "4042"}/remote`;
  setText("remote-address", remoteAddress);
  $("remote-address").dataset.address = remoteAddress;
  const cards = [];
  const ftdi = system.dmx?.ft232r_devices || [];
  cards.push({
    name: "FT232R / Open-DMX",
    detail: ftdi.length ? (ftdi[0].usb_path || "USB 0403:6001") : "Adapter not reported",
    sub: system.dmx?.native_driver_ready ? "libftdi transport ready" : "libftdi transport unavailable",
    ok: Boolean(ftdi.length && system.dmx?.native_driver_ready),
  });
  cards.push({
    name: "Line audio input",
    detail: system.audio?.cards?.[0] || "No capture card reported",
    sub: system.audio?.arecord || "arecord unavailable",
    ok: Boolean(system.audio?.cards?.length),
  });
  cards.push({
    name: "Active lighting rig",
    detail: `${fixtures().length} fixtures · ${new Set(fixtures().map((item) => item.universe || 0)).size} universe`,
    sub: app.bootstrap.rig.name,
    ok: true,
  });
  cards.push({
    name: "Spotify identity",
    detail: system.spotify?.client_id_configured ? "Client configured" : "Client ID not configured",
    sub: system.spotify?.token_present ? "Local token present" : "Optional integration disconnected",
    ok: Boolean(system.spotify?.client_id_configured && system.spotify?.token_present),
  });
  const container = $("hardware-cards");
  if (container) {
    container.innerHTML = cards.map((card) => `<div class="hardware-card ${card.ok ? "ok" : "warn"}"><i></i><b>${escapeHtml(card.name)}</b><span>${escapeHtml(card.detail)}</span><small>${escapeHtml(card.sub)}</small></div>`).join("");
  }
  const spotifyPhase = system.spotify?.phase || "disconnected";
  setText("spotify-connect-state", spotifyPhase.toUpperCase());
  setText("spotify-client-summary", system.spotify?.client_id_masked || "Not configured");
  const spotifyStatus = $("spotify-integration-status");
  if (spotifyStatus) {
    spotifyStatus.classList.remove("connected", "fault");
    if (system.spotify?.error) {
      spotifyStatus.classList.add("fault");
      spotifyStatus.textContent = system.spotify.error;
    } else if (system.spotify?.token_present) {
      spotifyStatus.classList.add("connected");
      spotifyStatus.textContent = "Spotify playback identity is connected on this computer.";
    } else if (spotifyPhase === "connecting") {
      spotifyStatus.textContent = "Authorization is waiting in the desktop browser.";
    } else {
      spotifyStatus.textContent = "No Spotify token is stored on this computer.";
    }
  }
}

function renderOperatorSettings(settings) {
  if ($("audio-device-input")) $("audio-device-input").value = settings.audio_device || "default";
  if ($("training-capture-enabled")) $("training-capture-enabled").checked = settings.training_capture_enabled !== false;
  if ($("training-max-gb")) $("training-max-gb").value = Number(settings.training_max_gb || 100);
  if ($("spotify-client-id")) {
    $("spotify-client-id").value = "";
    $("spotify-client-id").placeholder = settings.spotify_client_id_masked
      ? `Configured as ${settings.spotify_client_id_masked} — paste a new ID to replace`
      : "Paste the client ID from your private Spotify app";
  }
}

function renderServiceDetails() {
  setText("service-address", `${window.location.hostname}:${window.location.port || "4042"}`);
  const list = $("service-details");
  if (!list || !app.bootstrap) return;
  list.innerHTML = [
    ["Version", app.bootstrap.project.version],
    ["Host", app.system?.network?.host_name || "—"],
    ["Active rig", app.bootstrap.rig.name],
    ["Memory database", "Private local SQLite"],
    ["Desktop route", window.location.origin + "/"],
  ].map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

async function startEngine(mode) {
  const button = $(`${mode}-button`);
  const task = beginOperatorTask(
    `Starting ${label(mode)} mode`,
    mode === "live" ? "Opening line-in and the physical DMX output." : "Opening the audio engine and preparing the selected output.",
    button,
  );
  try {
    app.status = await api("/api/engine/start", { method: "POST", body: { mode } });
    renderStatus();
    toast(`${label(mode)} mode starting`, mode === "live" ? "Opening line-in and the FT232R DMX interface." : mode === "monitor" ? "Listening to line-in with virtual output." : "Running the built-in demonstration.", "success");
  } catch (error) {
    toast(`Could not start ${mode} mode`, error.message, "error");
  } finally {
    finishOperatorTask(task);
  }
}

async function stopEngine() {
  const task = beginOperatorTask(
    "Stopping the engine",
    "Closing the active session and finalizing its recorded audio and synchronized context.",
    $("stop-button"),
  );
  try {
    app.status = await api("/api/engine/stop", { method: "POST", body: {} });
    renderStatus();
  } catch (error) {
    toast("Could not stop engine", error.message, "error");
  } finally {
    finishOperatorTask(task);
  }
}

async function shutdownLumen() {
  const confirmed = window.confirm(
    "Shut down the Lumen engine and local operator service?"
  );
  if (!confirmed) return;
  try {
    const result = await api("/api/service/shutdown", { method: "POST", body: {} });
    window.clearTimeout(app.polling);
    app.polling = null;
    $("shutdown-screen")?.classList.remove("hidden");
    document.title = "Lumen is shut down";
    toast("Lumen is shutting down", result.message);
  } catch (error) {
    toast("Could not shut down Lumen", error.message, "error");
  }
}

async function patchControl(control, value) {
  try {
    app.status = await api("/api/control", { method: "POST", body: { [control]: value } });
    renderStatus();
  } catch (error) {
    toast("Control change failed", error.message, "error");
  }
}

function queueControl(control, value) {
  $$(`[data-output="${control}"]`).forEach((output) => { output.textContent = percent(value); });
  if (app.status?.controls) app.status.controls[control] = value;
  window.clearTimeout(app.controlTimer);
  app.controlTimer = window.setTimeout(() => patchControl(control, value), 85);
}

async function applyPreset(preset) {
  try {
    app.status = await api("/api/preset", { method: "POST", body: { preset } });
    $$("[data-preset]").forEach((button) => button.classList.toggle("active", button.dataset.preset === preset));
    renderStatus();
    toast(`${label(preset)} influence`, "The live response has been reshaped.", "success");
  } catch (error) {
    toast("Preset could not be applied", error.message, "error");
  }
}

async function sendFeedback(labelValue, value, note = null, surface = "desktop") {
  const selector = $(surface === "remote" ? "remote-feedback-scope" : "feedback-scope");
  const lifetimeSelector = $(surface === "remote" ? "remote-feedback-lifetime" : "feedback-lifetime");
  const selected = selector?.value || "overall";
  const lifetime = lifetimeSelector?.value || "cue";
  const [scope, fixtureId, groupId] = selected.startsWith("fixture:")
    ? ["fixture", selected.slice(8), null]
    : selected.startsWith("group:")
      ? ["group", null, selected.slice(6)]
      : ["overall", null, null];
  try {
    const clientEventId = window.crypto?.randomUUID?.()
      || `event-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    const result = await api("/api/feedback", {
      method: "POST",
      body: {
        label: labelValue,
        value: Number(value),
        note,
        scope,
        fixture_id: fixtureId,
        group_id: groupId,
        lifetime,
        participant_id: app.participantId,
        participant_name: app.participantName || null,
        client_event_id: clientEventId,
      },
    });
    if (result.feedback_id && result.created !== false) {
      app.feedbackReceipts.push(Number(result.feedback_id));
      app.feedbackReceipts = app.feedbackReceipts.slice(-30);
    }
    toast(
      result.created === false ? "Feedback already received" : "Feedback remembered",
      result.created === false
        ? "A network retry was ignored; its learning weight was not duplicated."
        : `${note || label(labelValue)} · ${result.participant_agreement || 1} listener agreement · applies at a phrase boundary`,
      "success",
    );
    app.memory = await api("/api/memory");
    renderMemory(app.memory);
    if ($("feedback-note")) $("feedback-note").value = "";
    if ($("remote-feedback-note")) $("remote-feedback-note").value = "";
  } catch (error) {
    toast("Feedback could not be saved", error.message, "error");
  }
}

async function sendTrainingAnnotation(kind, labelValue, surface = "desktop") {
  const selector = $(surface === "remote" ? "remote-feedback-scope" : "feedback-scope");
  const selected = selector?.value || "overall";
  const [scope, fixtureId, groupId] = kind === "musical_context"
    ? ["overall", null, null]
    : selected.startsWith("fixture:")
      ? ["fixture", selected.slice(8), null]
      : selected.startsWith("group:")
        ? ["group", null, selected.slice(6)]
        : ["overall", null, null];
  const note = $(surface === "remote" ? "remote-feedback-note" : "feedback-note")?.value.trim() || null;
  try {
    const clientEventId = window.crypto?.randomUUID?.()
      || `annotation-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    const result = await api("/api/training/annotation", {
      method: "POST",
      body: {
        kind,
        label: labelValue,
        scope,
        fixture_id: fixtureId,
        group_id: groupId,
        note,
        intensity: 1,
        participant_id: app.participantId,
        participant_name: app.participantName || null,
        client_event_id: clientEventId,
      },
    });
    toast(
      "Training label saved",
      `${label(labelValue)}${result.linked_to_audio ? " · linked to PCM" : " · no active PCM link"}${result.structure_consensus ? " · consensus updates after this run" : ""}`,
      "success",
    );
  } catch (error) {
    toast("Training label could not be saved", error.message, "error");
  }
}

async function rescanSystem(button = null) {
  const task = beginOperatorTask(
    "Scanning connected hardware",
    "Refreshing ALSA, FTDI, network, and Spotify status.",
    button,
  );
  try {
    toast("Scanning hardware", "Refreshing ALSA, FTDI, network, and Spotify state.");
    app.system = await api("/api/system");
    renderSystem(app.system);
    updateComponentStatuses();
  } catch (error) {
    toast("Hardware scan failed", error.message, "error");
  } finally {
    finishOperatorTask(task);
  }
}

async function saveAudioDevice() {
  const audioDevice = $("audio-device-input")?.value.trim();
  if (!audioDevice) return;
  try {
    const result = await api("/api/settings", { method: "POST", body: { audio_device: audioDevice } });
    app.status = result.status;
    app.bootstrap.settings = result.settings;
    renderStatus();
    renderOperatorSettings(result.settings);
    toast("Audio input saved", `Lumen will capture from ${audioDevice}.`, "success");
  } catch (error) {
    toast("Audio input was not changed", error.message, "error");
  }
}

async function saveTrainingSettings() {
  try {
    const result = await api("/api/settings", {
      method: "POST",
      body: {
        training_capture_enabled: Boolean($("training-capture-enabled")?.checked),
        training_max_gb: Number($("training-max-gb")?.value || 100),
      },
    });
    app.status = result.status;
    app.bootstrap.settings = result.settings;
    renderOperatorSettings(result.settings);
    renderStatus();
    toast("Training capture settings saved", "They apply when Monitor or Live mode starts.", "success");
  } catch (error) {
    toast("Training settings were not changed", error.message, "error");
  }
}

async function exportTrainingData() {
  const task = beginOperatorTask(
    "Building the training manifest",
    "Validating captured sessions and writing the versioned local dataset index.",
    $("training-export-button"),
  );
  try {
    const result = await api("/api/training/export", { method: "POST", body: {} });
    if (app.status?.training) app.status.training.last_export = result.path;
    renderStatus();
    toast("Training manifest built", result.path, "success");
  } catch (error) {
    toast("Training export failed", error.message, "error");
  } finally {
    finishOperatorTask(task);
  }
}

async function connectSpotify() {
  const clientId = $("spotify-client-id")?.value.trim() || "";
  try {
    const result = await api("/api/spotify/connect", { method: "POST", body: { client_id: clientId } });
    toast("Spotify authorization started", result.message, "success");
    setText("spotify-connect-state", "CONNECTING");
    if ($("spotify-integration-status")) $("spotify-integration-status").textContent = result.message;
  } catch (error) {
    toast("Spotify connection could not start", error.message, "error");
  }
}

function toggleBlackout() {
  const current = Boolean(app.status?.controls?.blackout);
  patchControl("blackout", !current);
}

function installHandlers() {
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (button && button.dataset.manualConfirmation !== "true") {
      confirmButtonPress(button);
    }
  });
  for (const id of ["feedback-listener-name", "remote-listener-name"]) {
    $(id)?.addEventListener("change", (event) => {
      updateParticipantName(event.target.value);
    });
  }
  $$("[data-nav]").forEach((button) => button.addEventListener("click", () => setPage(button.dataset.nav)));
  $("monitor-button")?.addEventListener("click", () => startEngine("monitor"));
  $("live-button")?.addEventListener("click", () => startEngine("live"));
  $("demo-button")?.addEventListener("click", () => startEngine("demo"));
  $("stop-button")?.addEventListener("click", stopEngine);
  $("blackout-button")?.addEventListener("click", toggleBlackout);
  $("remote-blackout-button")?.addEventListener("click", toggleBlackout);
  $("spotify-back-button")?.addEventListener("click", () => navigateSpotifyHistory(-1));
  $("spotify-forward-button")?.addEventListener("click", () => navigateSpotifyHistory(1));
  $("fresh-gesture-button")?.addEventListener("click", requestFreshGesture);
  $("rehearsal-routines")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-rehearsal-routine]");
    if (button) patchRehearsal({ routine: button.dataset.rehearsalRoutine, tour: false });
  });
  $("rehearsal-start")?.addEventListener("click", startRehearsal);
  $("rehearsal-stop")?.addEventListener("click", stopEngine);
  $("rehearsal-previous")?.addEventListener("click", () => stepRehearsal(-1));
  $("rehearsal-next")?.addEventListener("click", () => stepRehearsal(1));
  $("rehearsal-tour")?.addEventListener("click", () => patchRehearsal({ tour: !Boolean(app.status?.rehearsal?.tour) }));
  $("rehearsal-restart")?.addEventListener("click", async () => {
    if (app.status?.engine?.mode === "rehearsal" && app.status?.engine?.running) {
      await stopEngine();
      window.setTimeout(startRehearsal, 250);
    } else startRehearsal();
  });
  ["rehearsal-output", "rehearsal-scope", "rehearsal-palette", "rehearsal-isolate"].forEach((id) => {
    $(id)?.addEventListener("change", () => patchRehearsal(rehearsalFormValues()));
  });
  ["rehearsal-bpm", "rehearsal-size", "rehearsal-intensity", "rehearsal-strobe"].forEach((id) => {
    $(id)?.addEventListener("input", () => {
      setText("rehearsal-bpm-value", `${$("rehearsal-bpm").value} BPM`);
      setText("rehearsal-size-value", `${$("rehearsal-size").value}%`);
      setText("rehearsal-intensity-value", `${$("rehearsal-intensity").value}%`);
      setText("rehearsal-strobe-value", Number($("rehearsal-strobe").value) ? `${$("rehearsal-strobe").value}%` : "Off");
      queueRehearsal(rehearsalFormValues());
    });
  });
  $$('[data-motion-control]').forEach((input) => {
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      const values = motionFormValues();
      setText("motion-cycle-value", `${values.cycle_beats.toFixed(0)} beats`);
      setText("motion-pan-size-value", percent(values.pan_size));
      setText("motion-tilt-size-value", percent(values.tilt_size));
      setText("motion-pan-center-value", percent(values.pan_center));
      setText("motion-tilt-center-value", percent(values.tilt_center));
      queueMotionRoutine();
    });
  });
  $("motion-editor-scope")?.addEventListener("change", (event) => {
    app.motionEditorScope = event.target.value;
    renderMotionEditor(app.status?.rehearsal?.motion_editor || {});
  });
  $$('[data-center-motion-control]').forEach((input) => {
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      const values = motionFormValues();
      setText("center-cycle-value", `${values.cycle_beats.toFixed(0)} beats`);
      setText("center-body-travel-value", percent(values.body_travel));
      setText("center-body-speed-value", `${values.body_speed.toFixed(2)}×`);
      setText("center-body-phase-value", percent(values.body_phase));
      setText("center-arm-1-travel-value", percent(values.arm_1_travel));
      setText("center-arm-1-speed-value", `${values.arm_1_speed.toFixed(2)}×`);
      setText("center-arm-1-phase-value", percent(values.arm_1_phase));
      setText("center-arm-2-travel-value", percent(values.arm_2_travel));
      setText("center-arm-2-speed-value", `${values.arm_2_speed.toFixed(2)}×`);
      setText("center-arm-2-phase-value", percent(values.arm_2_phase));
      setText("center-laser-level-value", percent(values.laser_level));
      setText("center-strip-program-value", values.strip_program.toFixed(0));
      setText("center-strip-speed-value", percent(values.strip_speed));
      setText("center-strobe-level-value", values.strobe_level ? percent(values.strobe_level) : "Off");
      setText("center-intensity-value", percent(values.intensity));
      setText("center-blackout-accent-value", values.blackout_accent ? percent(values.blackout_accent) : "Off");
      queueMotionRoutine();
    });
  });
  $("motion-reset-defaults")?.addEventListener("click", () => patchMotionRoutine({}, "reset"));
  $$('[data-capture-live-cue]').forEach((button) => {
    button.addEventListener("click", () => captureLiveCue(button.dataset.captureLiveCue));
  });
  $("sequence-add-step")?.addEventListener("click", () => {
    syncSequenceDraftFromDom();
    const previous = app.sequenceDraft.at(-1) || {};
    app.sequenceDraft.push({
      routine: previous.routine || "breathe",
      duration_beats: 8,
      intensity: Number(previous.intensity ?? 0.72),
      brightness: Number(previous.brightness ?? previous.intensity ?? 0.72),
      motion_speed: Number(previous.motion_speed ?? 0.5),
      travel_size: Number(previous.travel_size ?? 1),
      activity_density: Number(previous.activity_density ?? 1),
      beat_sync: Number(previous.beat_sync ?? 1),
      palette: previous.palette || "",
      strobe: Number(previous.strobe || 0),
    });
    renderSequenceDraft();
  });
  $("sequence-save-here")?.addEventListener("click", () => saveSequenceHere(false));
  installWakeProtectedAction(
    $("remote-sequence-save"),
    () => saveSequenceHere(true),
  );
  $("sequence-clear")?.addEventListener("click", clearSequenceDraft);
  $("sequence-undo-history")?.addEventListener("click", () => {
    const undo = app.choreographyUndo;
    if (undo) changeChoreographyHistory(undo.kind, "undo", undo.id);
  });
  $("sequence-steps")?.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-step-remove]");
    if (!remove || app.sequenceDraft.length <= 1) return;
    syncSequenceDraftFromDom();
    app.sequenceDraft.splice(Number(remove.dataset.stepRemove), 1);
    renderSequenceDraft();
  });
  $("sequence-steps")?.addEventListener("input", (event) => {
    const input = event.target.closest("[data-step-field]");
    if (!input) return;
    const row = input.closest("[data-sequence-step]");
    const output = input.closest("label")?.querySelector("output");
    if (output && ["intensity", "brightness", "motion_speed", "travel_size", "activity_density", "beat_sync", "strobe"].includes(input.dataset.stepField)) {
      output.textContent = Number(input.value) ? `${input.value}%` : "Off";
    }
    if (input.dataset.stepField === "duration_beats") {
      syncSequenceDraftFromDom();
    } else if (row) {
      syncSequenceDraftFromDom();
    }
  });
  $("sequence-steps")?.addEventListener("change", (event) => {
    if (event.target?.dataset?.stepField === "duration_beats") {
      syncSequenceDraftFromDom();
      renderSequenceDraft();
    }
  });
  $("sequence-history-list")?.addEventListener("click", (event) => {
    const load = event.target.closest("[data-sequence-load]");
    const deleteSequence = event.target.closest("[data-sequence-delete]");
    const deletePlacement = event.target.closest("[data-placement-delete]");
    if (load) loadSequenceForEdit(load.dataset.sequenceLoad);
    else if (deleteSequence) changeChoreographyHistory("sequence", "delete", deleteSequence.dataset.sequenceDelete);
    else if (deletePlacement) changeChoreographyHistory("placement", "delete", deletePlacement.dataset.placementDelete);
  });
  $("structure-timeline-list")?.addEventListener("click", (event) => {
    const review = event.target.closest("[data-timeline-review]");
    const correct = event.target.closest("[data-timeline-correct]");
    const save = event.target.closest("[data-timeline-save]");
    if (review) {
      reviewStructureTimeline(
        review.dataset.timelineId,
        review.dataset.timelineReview,
      );
    } else if (correct) {
      app.editingStructureTimelineId = (
        app.editingStructureTimelineId === correct.dataset.timelineCorrect
          ? null
          : correct.dataset.timelineCorrect
      );
      renderStructureTimelines(app.structureLibrary || app.teaching || {});
    } else if (save) {
      saveStructureCorrection(save.dataset.timelineSave);
    }
  });
  $("structure-song-select")?.addEventListener("change", (event) => {
    const recordingId = event.target.value;
    if (!recordingId) return;
    app.editingStructureTimelineId = null;
    app.selectedStructureRecordingId = recordingId;
    void refreshStructureLibrary(recordingId);
  });
  $("structure-song-search")?.addEventListener("input", () => {
    renderStructureSongOptions();
    renderStructureCatalogTable();
  });
  $("structure-song-filter")?.addEventListener("change", () => {
    renderStructureSongOptions();
    renderStructureCatalogTable();
  });
  $("structure-catalog-body")?.addEventListener("click", (event) => {
    const row = event.target.closest("[data-structure-recording]");
    const recordingId = row?.dataset.structureRecording;
    if (!recordingId) return;
    app.editingStructureTimelineId = null;
    app.selectedStructureRecordingId = recordingId;
    void refreshStructureLibrary(recordingId);
  });
  $("structure-song-playing")?.addEventListener("click", () => {
    const recordingId = app.teaching?.recording_id;
    if (!recordingId) {
      toast("No recognized playing song", "Start a Spotify track that Lumen has analyzed, or select one from the library.", "error");
      return;
    }
    if (!(app.structureLibrary?.catalog || []).some((item) => item.recording_id === recordingId)) {
      toast("Playing song has no timeline", "Analyze the complete recording before reviewing it.", "error");
      return;
    }
    app.editingStructureTimelineId = null;
    app.selectedStructureRecordingId = recordingId;
    void refreshStructureLibrary(recordingId);
  });
  $("structure-overview")?.addEventListener("click", (event) => {
    const segment = event.target.closest("[data-overview-segment]");
    if (!segment) return;
    const row = $("structure-timeline-list")?.querySelector(
      `[data-structure-segment="${segment.dataset.overviewSegment}"]`,
    );
    row?.scrollIntoView({ behavior: "smooth", block: "center" });
    row?.classList.add("selected");
    window.setTimeout(() => row?.classList.remove("selected"), 1800);
  });

  $$("[data-control]").forEach((input) => {
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => queueControl(input.dataset.control, input.tagName === "SELECT" ? input.value : Number(input.value) / 100));
  });
  $("calibration-toggle")?.addEventListener("click", () => sendCalibration(!app.calibrationActive));
  ["calibration-pan", "calibration-tilt", "calibration-speed"].forEach((id) => {
    $(id)?.addEventListener("input", () => {
      updateCalibrationJogReadouts();
      calibrationJog();
    });
  });
  $$('[data-calibration-capture]').forEach((button) => {
    button.addEventListener('click', () => {
      try {
        captureCalibrationPoint(button.dataset.calibrationCapture);
      } catch (error) {
        toast("Calibration point rejected", error.message, "error");
      }
    });
  });
  $$('[data-calibration-preview]').forEach((button) => {
    button.addEventListener('click', () => {
      if (!app.calibrationActive) {
        toast("Start calibration first", "The selected mover must be in its visible calibration state.", "error");
        return;
      }
      const kind = button.dataset.calibrationPreview;
      const fieldByKind = {
        pan_left: "pan_left_dmx", pan_home: "home_pan_dmx", pan_right: "pan_right_dmx",
        tilt_high: "tilt_high_dmx", tilt_home: "home_tilt_dmx", tilt_low: "tilt_low_dmx",
      };
      const axis = kind.startsWith("pan_") ? "pan" : "tilt";
      const slider = $(axis === "pan" ? "calibration-pan" : "calibration-tilt");
      if (slider) slider.value = calibrationFieldNumber(fieldByKind[kind], 128);
      updateCalibrationJogReadouts();
      calibrationJog();
    });
  });
  $$("[data-preset]").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  $$("[data-feedback]").forEach((button) => {
    installFeedbackButton(button);
  });
  $("feedback-note-button")?.addEventListener("click", () => {
    const note = $("feedback-note").value.trim();
    if (note) sendFeedback("operator_note", 0, note, "desktop");
  });
  $("feedback-undo-button")?.addEventListener("click", undoLastFeedback);
  installWakeProtectedAction($("remote-feedback-undo"), undoLastFeedback);
  installWakeProtectedAction($("remote-feedback-note-button"), () => {
    const note = $("remote-feedback-note").value.trim();
    if (note) sendFeedback("operator_note", 0, note, "remote");
  });
  $("remote-note-open")?.addEventListener("click", () => {
    $("remote-note-box").classList.toggle("hidden");
    if (!$("remote-note-box").classList.contains("hidden")) $("remote-feedback-note").focus();
  });
  $("feedback-context-button")?.addEventListener("click", () => {
    sendTrainingAnnotation("musical_context", $("feedback-context-label")?.value, "desktop");
  });
  $("feedback-action-button")?.addEventListener("click", () => {
    sendTrainingAnnotation("preferred_action", $("feedback-action-label")?.value, "desktop");
  });
  installWakeProtectedAction($("remote-context-button"), () => {
    sendTrainingAnnotation("musical_context", $("remote-context-label")?.value, "remote");
  });
  installWakeProtectedAction($("remote-action-button"), () => {
    sendTrainingAnnotation("preferred_action", $("remote-action-label")?.value, "remote");
  });
  $("feedback-history")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-feedback]");
    if (button) deleteFeedback(button.dataset.deleteFeedback);
  });

  $$("[data-room-view]").forEach((button) => {
    button.addEventListener("click", () => {
      app.roomView = button.dataset.roomView;
      $$("[data-room-view]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      drawPerformanceRoom();
    });
  });
  $$("[data-rig-view]").forEach((button) => {
    button.addEventListener("click", () => {
      app.rigView = button.dataset.rigView;
      $$("[data-rig-view]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      drawRig();
    });
  });

  $("fixture-filter")?.addEventListener("input", (event) => renderFixtureList(event.target.value));
  $("memory-filter")?.addEventListener("input", (event) => renderSongLibrary(event.target.value));
  $("save-fixture-button")?.addEventListener("click", saveSelectedFixture);
  $("solve-target-button")?.addEventListener("click", solveTargetFromFields);
  $("center-room-button")?.addEventListener("click", () => {
    $("target-x").value = 0;
    $("target-y").value = 0;
    $("target-z").value = 1.2;
    solveTargetFromFields();
  });
  installRigPointerHandlers();

  $("rescan-system-button")?.addEventListener("click", () => rescanSystem($("rescan-system-button")));
  $("system-rescan-button")?.addEventListener("click", () => rescanSystem($("system-rescan-button")));
  $("shutdown-lumen-button")?.addEventListener("click", shutdownLumen);
  $("audio-input-test-button")?.addEventListener("click", () => {
    if (app.status?.engine?.running && app.status.engine.mode === "monitor") stopEngine();
    else startEngine("monitor");
  });
  $("save-audio-device-button")?.addEventListener("click", saveAudioDevice);
  $("save-training-settings-button")?.addEventListener("click", saveTrainingSettings);
  $("training-export-button")?.addEventListener("click", exportTrainingData);
  $("research-refresh-button")?.addEventListener("click", refreshResearch);
  $("research-import-button")?.addEventListener("click", importResearchAnnotations);
  $("research-run-button")?.addEventListener("click", runResearchJob);
  $("research-cancel-button")?.addEventListener("click", cancelResearch);
  $("research-train-button")?.addEventListener("click", trainStructureStudent);
  $("spotify-connect-button")?.addEventListener("click", connectSpotify);
  $("spotify-refresh-button")?.addEventListener("click", () => refreshSpotifyConsole(true));
  $("remote-spotify-refresh")?.addEventListener("click", () => refreshSpotifyConsole(true));
  $("remote-spotify-previous")?.addEventListener("click", () => spotifyCommand("previous"));
  $("remote-spotify-play")?.addEventListener("click", () => spotifyCommand(app.spotify?.playback?.is_playing ? "pause" : "play"));
  $("remote-spotify-next")?.addEventListener("click", () => spotifyCommand("next"));
  $("remote-spotify-playlist")?.addEventListener("change", (event) => {
    app.spotifyPlaylistId = event.target.value || "";
    refreshSpotifyConsole(true, "");
  });
  $("remote-spotify-play-playlist")?.addEventListener("click", () => {
    const playlist = (app.spotify?.playlists || []).find((item) => item.id === app.spotifyPlaylistId);
    if (playlist?.uri) spotifyCommand("play", { context_uri: playlist.uri });
  });
  $("spotify-search-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    app.spotifyPlaylistId = "";
    refreshSpotifyConsole(true, $("spotify-search-input")?.value.trim() || "");
  });
  $("spotify-reconnect-button")?.addEventListener("click", connectSpotify);
  $("spotify-playlist-select")?.addEventListener("change", (event) => {
    app.spotifyPlaylistId = event.target.value || "";
    if ($("spotify-search-input")) $("spotify-search-input").value = "";
    refreshSpotifyConsole(true, "");
  });
  $("spotify-play-playlist-button")?.addEventListener("click", () => {
    const playlist = (app.spotify?.playlists || []).find(
      (candidate) => candidate.id === app.spotifyPlaylistId,
    );
    if (playlist?.uri) spotifyCommand("play", { context_uri: playlist.uri });
  });
  $("spotify-previous-button")?.addEventListener("click", () => spotifyCommand("previous"));
  $("spotify-play-button")?.addEventListener("click", () => {
    spotifyCommand(app.spotify?.playback?.is_playing ? "pause" : "play");
  });
  $("spotify-next-button")?.addEventListener("click", () => spotifyCommand("next"));
  $("spotify-transfer-button")?.addEventListener("click", () => spotifyCommand("transfer", {
    play: Boolean(app.spotify?.playback?.is_playing),
  }));
  $("spotify-seek")?.addEventListener("change", (event) => {
    const duration = Number(app.spotify?.playback?.track?.duration_ms || 0);
    if (duration) spotifyCommand("seek", { position_ms: Math.round(duration * Number(event.target.value) / 1000) });
  });
  $("spotify-volume")?.addEventListener("change", (event) => {
    spotifyCommand("volume", { volume_percent: Number(event.target.value) });
  });
  $("spotify-device-select")?.addEventListener("change", (event) => {
    app.spotifyTransferDeviceId = event.target.value || "";
    renderSpotifyConsole();
  });
  $("copy-spotify-redirect-button")?.addEventListener("click", async () => {
    const redirect = "http://127.0.0.1:8765/callback";
    try {
      await navigator.clipboard.writeText(redirect);
      toast("Spotify redirect URI copied", redirect, "success");
    } catch {
      toast("Spotify redirect URI", redirect);
    }
  });
  $("copy-remote-address")?.addEventListener("click", async () => {
    const address = $("remote-address").dataset.address || $("remote-address").textContent;
    try {
      await navigator.clipboard.writeText(address);
      toast("Remote address copied", address, "success");
    } catch {
      toast("Remote address", address);
    }
  });

  const dialog = $("hotkey-dialog");
  $("hotkey-help-button")?.addEventListener("click", () => dialog?.showModal());
  $("close-hotkey-dialog")?.addEventListener("click", () => dialog?.close());

  $$("[data-remote-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      $$("[data-remote-tab]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      const target = button.dataset.remoteTab === "feedback" ? $$(".remote-section")[1] : $$(".remote-section")[0];
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  document.addEventListener("keydown", handleHotkey);
}

function installRigPointerHandlers() {
  const canvas = $("rig-canvas");
  if (!canvas) return;
  canvas.addEventListener("pointerdown", (event) => {
    if (!app.bootstrap) return;
    if (app.rigView === "3d") {
      app.pointer = { dragging: true, moved: false, fixtureId: null, mode: event.ctrlKey ? "rotate" : "pan", x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    let nearest = null;
    let nearestDistance = 14;
    for (const fixture of fixtures()) {
      const point = roomProjection({ width: rect.width, height: rect.height }, app.rigView, fixture.position_m);
      const distance = Math.hypot(event.clientX - rect.left - point.x, event.clientY - rect.top - point.y);
      if (distance < nearestDistance) {
        nearest = fixture;
        nearestDistance = distance;
      }
    }
    app.pointer = { dragging: Boolean(nearest), moved: false, fixtureId: nearest?.id || null };
    if (nearest) {
      selectFixture(nearest.id);
      canvas.setPointerCapture(event.pointerId);
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!app.bootstrap) return;
    if (app.rigView === "3d" && app.pointer.dragging) {
      const dx = event.clientX - (app.pointer.x || event.clientX);
      const dy = event.clientY - (app.pointer.y || event.clientY);
      app.pointer.x = event.clientX; app.pointer.y = event.clientY;
      if (Math.abs(dx) + Math.abs(dy) > 1) app.pointer.moved = true;
      if (app.pointer.mode === "rotate") {
        app.roomCamera.yaw += dx * 0.008;
        app.roomCamera.pitch = clamp(app.roomCamera.pitch + dy * 0.008, -1.2, 1.2);
      } else {
        app.roomCamera.panX += dx;
        app.roomCamera.panY += dy;
      }
      drawRig();
      return;
    }
    const position = roomPositionFromCanvas(canvas, app.rigView, event.clientX, event.clientY, Number($("target-z")?.value || 1.2));
    setText("cursor-x", position[0].toFixed(2));
    setText("cursor-y", position[1].toFixed(2));
    setText("cursor-z", position[2].toFixed(2));
    if (!app.pointer.dragging || !app.pointer.fixtureId || app.rigView !== "plan") return;
    app.pointer.moved = true;
    $("fixture-x").value = position[0].toFixed(4);
    $("fixture-y").value = position[1].toFixed(4);
    const fixture = selectedFixture();
    if (fixture) {
      fixture.position_m[0] = position[0];
      fixture.position_m[1] = position[1];
      drawRig();
      setText("rig-save-state", "Unsaved position change");
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (app.rigView === "3d" && app.pointer.dragging) {
      app.pointer = { dragging: false, moved: app.pointer.moved, fixtureId: null };
      return;
    }
    if (!app.pointer.moved && !app.pointer.fixtureId) {
      const position = roomPositionFromCanvas(canvas, app.rigView, event.clientX, event.clientY, Number($("target-z")?.value || 1.2));
      $("target-x").value = position[0].toFixed(3);
      $("target-y").value = position[1].toFixed(3);
      if (app.rigView !== "plan") $("target-z").value = position[2].toFixed(3);
      solveTargetFromFields();
    }
    app.pointer = { dragging: false, moved: false, fixtureId: null };
  });
  canvas.addEventListener("wheel", (event) => {
    if (app.rigView !== "3d") return;
    event.preventDefault();
    app.roomCamera.zoom = clamp(app.roomCamera.zoom - event.deltaY * 0.001, 0.55, 1.8);
    drawRig();
  }, { passive: false });
}

async function requestFreshGesture() {
  try {
    app.status = await api("/api/gesture/fresh", { method: "POST", body: {} });
    renderStatus();
    toast("Fresh gesture requested", "Lumen may choose a new visual idea on the current musical evidence.", "success");
  } catch (error) {
    toast("Fresh gesture unavailable", error.message, "error");
  }
}

function handleHotkey(event) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName) || event.target?.isContentEditable) return;
  if (/^[1-7]$/.test(event.key)) {
    setPage(["performance", "rehearsal", "rig", "audio", "memory", "music", "system"][Number(event.key) - 1]);
    event.preventDefault();
  } else if (event.key.toLowerCase() === "b") {
    toggleBlackout();
    event.preventDefault();
  } else if (event.key.toLowerCase() === "m") {
    startEngine("monitor");
    event.preventDefault();
  } else if (event.key.toLowerCase() === "l") {
    startEngine("live");
    event.preventDefault();
  } else if (event.key === "Escape") {
    if ($("hotkey-dialog")?.open) $("hotkey-dialog").close();
    else stopEngine();
  } else if (event.key === " ") {
    requestFreshGesture();
    event.preventDefault();
  } else if (event.key === "[" || event.key === "]") {
    const current = app.status?.controls?.intensity ?? 0.5;
    const next = clamp(current + (event.key === "]" ? 0.05 : -0.05));
    queueControl("intensity", next);
    renderControls({ ...app.status.controls, intensity: next });
    event.preventDefault();
  }
}

function updateClock() {
  const now = new Date();
  const clock = now.toLocaleTimeString([], { hour12: false });
  setText("clock", clock);
  setText("footer-time", clock.slice(0, 5));
  renderOperatorTask();
  if (app.page === "audio") {
    const age = app.lastStatusReceivedAt ? Date.now() - app.lastStatusReceivedAt : Infinity;
    setText(
      "analysis-rate",
      age > 750
        ? `DISPLAY STALE · ${Math.round(age)} ms`
        : `PCM 10 Hz · UI ${Math.round(app.statusLatencyMs)} ms`,
    );
  }
}

initialize();
