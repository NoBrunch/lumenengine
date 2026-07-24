"use strict";

const app = {
  remote: window.location.pathname === "/remote",
  bootstrap: null,
  status: null,
  system: null,
  memory: null,
  page: "performance",
  selectedFixtureId: null,
  roomView: "plan",
  rigView: "plan",
  scope: Array.from({ length: 150 }, () => 0),
  pointer: { dragging: false, moved: false, fixtureId: null },
  polling: null,
  controlTimer: null,
  disconnected: false,
  pollCount: 0,
};

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

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function setWidth(id, value) {
  const element = $(id);
  if (element) element.style.width = `${clamp(value) * 100}%`;
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
  if (!["performance", "rig", "audio", "memory", "system"].includes(name)) return;
  app.page = name;
  $$(".workspace-page").forEach((page) => page.classList.toggle("active", page.dataset.page === name));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.nav === name));
  if (name === "rig") window.setTimeout(drawRig, 30);
  if (name === "performance") window.setTimeout(drawPerformanceRoom, 30);
  if (name === "audio") window.setTimeout(drawScope, 30);
}

async function initialize() {
  if (app.remote) document.body.classList.add("remote-mode");
  installHandlers();
  updateClock();
  window.setInterval(updateClock, 1000);
  try {
    app.bootstrap = await api("/api/bootstrap");
    app.status = app.bootstrap.status;
    app.system = app.bootstrap.system;
    app.memory = app.bootstrap.memory;
    renderBootstrap();
    renderStatus();
    app.disconnected = false;
    $("loading-screen").classList.add("loaded");
  } catch (error) {
    $("loading-screen").classList.add("loaded");
    toast("Could not open Lumen", error.message, "error");
    renderConnection(false);
  }
  app.polling = window.setInterval(pollStatus, 250);
  window.addEventListener("resize", () => {
    drawPerformanceRoom();
    drawRig();
    drawScope();
  });
}

async function pollStatus() {
  try {
    const status = await api("/api/status");
    app.status = status;
    if (app.disconnected) toast("Lumen reconnected", "Live state is available again.", "success");
    app.disconnected = false;
    renderStatus();
    renderConnection(true);
    app.pollCount += 1;
    if (app.pollCount % 20 === 0) {
      app.system = await api("/api/system");
      renderSystem(app.system);
      updateComponentStatuses();
    }
  } catch {
    if (!app.disconnected) toast("Connection interrupted", "Trying to reconnect to the local Lumen service.", "error");
    app.disconnected = true;
    renderConnection(false);
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
  renderSystem(app.system);
  renderOperatorSettings(app.bootstrap.settings || {});
  renderMemory(app.memory);
  buildDmxHeatmap();
  renderServiceDetails();
  drawPerformanceRoom();
  drawRig();
  drawScope();
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
  setText("audio-device-state", running && engine.mode !== "demo" ? "Capturing" : running ? "Demo source" : "Not running");
  $("audio-device-state")?.classList.toggle("online", running);

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
  renderMedia(status.media, observation);
  renderExpression(decision, expression, observation);
  renderDmx(status);
  renderEvents(status.events || []);
  renderTargetSolutions(status.target_solutions || []);
  renderConnection(true);
  updateComponentStatuses();

  app.scope.push(clamp(observation.loudness) * (0.55 + 0.45 * Math.sin(Date.now() / 70)));
  if (app.scope.length > 150) app.scope.shift();
  if (app.page === "performance") drawPerformanceRoom();
  if (app.page === "rig") drawRig();
  if (app.page === "audio") drawScope();
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
      input.value = Math.round(Number(controls[input.dataset.control]) * 100);
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

  const bpm = observation.bpm ? Number(observation.bpm).toFixed(1) : "—";
  const section = label(observation.section || "waiting");
  setText("fact-bpm", bpm);
  setText("fact-beat", percent(observation.beat_confidence));
  setText("fact-section", section);
  setText("remote-bpm", bpm);
  setText("remote-section", section.toUpperCase());
}

function renderExpression(decision, expression, observation) {
  const gesture = decision ? label(decision.gesture) : "Standing by";
  const reason = decision?.reason || "Start Monitor, Perform, or Demo to begin interpretation.";
  const confidence = decision?.confidence || 0;
  setText("current-gesture", gesture);
  setText("decision-reason", reason);
  setText("expression-confidence", `${percent(confidence)} CONFIDENCE`);
  setText("remote-gesture", gesture);
  setText("remote-reason", reason);
  setText("remote-confidence", percent(confidence));
  setText("remote-energy", percent(expression.energy));
  if ($("remote-energy-orb")) $("remote-energy-orb").style.setProperty("--energy", Math.round(expression.energy * 100));

  for (const name of ["energy", "tension", "motion", "intimacy"]) {
    setWidth(`meter-${name}`, expression[name]);
    setText(`value-${name}`, Number(expression[name]).toFixed(2));
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
  setText("beat-lock-badge", observation.beat_confidence >= 0.5 ? "LOCKED" : "SEARCHING");
  const circumference = 2 * Math.PI * 66;
  const phase = clamp(observation.beat_phase);
  if ($("beat-dial-progress")) $("beat-dial-progress").style.strokeDashoffset = String(circumference * (1 - phase));
  if ($("beat-hand")) $("beat-hand").style.transform = `rotate(${phase * 360}deg)`;
}

function renderDmx(status) {
  const map = new Map((status.dmx?.active_channels || []).map((item) => [item.channel, item.value]));
  $$(".dmx-cell").forEach((cell) => {
    const value = map.get(Number(cell.dataset.channel)) || 0;
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
  if (audio) {
    audio.querySelector("b").textContent = engine.running && engine.mode !== "demo" ? "Capturing" : hasAudio ? "Available" : "Not reported";
    setStatusClass(audio, engine.running && engine.mode !== "demo" ? "active" : hasAudio ? "ok" : "warn");
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
  }
  const channelMap = $("fixture-channel-map");
  if (channelMap) {
    channelMap.innerHTML = Object.entries(profile?.channels || {})
      .map(([name, offset]) => `<span>${escapeHtml(name)} · ${Number(fixture.address) + Number(offset) - 1}</span>`)
      .join("");
  }
  drawRig();
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
  try {
    app.bootstrap = await api("/api/rig/fixture", { method: "POST", body: payload });
    app.status = app.bootstrap.status;
    app.memory = app.bootstrap.memory;
    app.system = app.bootstrap.system;
    setText("rig-save-state", "Saved to active rig");
    renderFixtureList($("fixture-filter")?.value || "");
    selectFixture(fixture.id);
    toast("Fixture saved", `${payload.name} is now part of the active Lumen rig.`, "success");
  } catch (error) {
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
  if (view === "front") {
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
  const values = app.scope;
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(100, 222, 213, .85)");
  gradient.addColorStop(1, "rgba(69, 126, 163, .42)");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1.4;
  ctx.shadowColor = "rgba(100, 222, 213, .4)";
  ctx.shadowBlur = 4;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = index / Math.max(1, values.length - 1) * width;
    const modulation = Math.sin(index * 1.8 + performance.now() / 80) * value * height * 0.16;
    const y = height * 0.5 - modulation - (value - 0.5) * height * 0.18;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
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
          <div><b>${escapeHtml(label(item.label))}</b><time>${escapeHtml(formatElapsed(item.created_unix_ms))}</time></div>
          <span>${escapeHtml(item.song_title || "Unidentified session")} · ${formatTime(item.position_ms)}</span>
          ${item.note ? `<p>${escapeHtml(item.note)}</p>` : ""}
        </article>`).join("")
      : `<div class="empty-selection"><b>No feedback yet</b><p>Use the performance console or phone remote to teach Lumen.</p></div>`;
  }
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
  try {
    app.status = await api("/api/engine/start", { method: "POST", body: { mode } });
    renderStatus();
    toast(`${label(mode)} mode starting`, mode === "live" ? "Opening line-in and the FT232R DMX interface." : mode === "monitor" ? "Listening to line-in with virtual output." : "Running the built-in demonstration.", "success");
  } catch (error) {
    toast(`Could not start ${mode} mode`, error.message, "error");
  }
}

async function stopEngine() {
  try {
    app.status = await api("/api/engine/stop", { method: "POST", body: {} });
    renderStatus();
  } catch (error) {
    toast("Could not stop engine", error.message, "error");
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

async function sendFeedback(labelValue, value, note = null) {
  try {
    await api("/api/feedback", { method: "POST", body: { label: labelValue, value: Number(value), note } });
    toast("Feedback remembered", note || label(labelValue), "success");
    app.memory = await api("/api/memory");
    renderMemory(app.memory);
    if ($("feedback-note")) $("feedback-note").value = "";
    if ($("remote-feedback-note")) $("remote-feedback-note").value = "";
  } catch (error) {
    toast("Feedback could not be saved", error.message, "error");
  }
}

async function rescanSystem() {
  try {
    toast("Scanning hardware", "Refreshing ALSA, FTDI, network, and Spotify state.");
    app.system = await api("/api/system");
    renderSystem(app.system);
    updateComponentStatuses();
  } catch (error) {
    toast("Hardware scan failed", error.message, "error");
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
  $$("[data-nav]").forEach((button) => button.addEventListener("click", () => setPage(button.dataset.nav)));
  $("monitor-button")?.addEventListener("click", () => startEngine("monitor"));
  $("live-button")?.addEventListener("click", () => startEngine("live"));
  $("demo-button")?.addEventListener("click", () => startEngine("demo"));
  $("stop-button")?.addEventListener("click", stopEngine);
  $("blackout-button")?.addEventListener("click", toggleBlackout);
  $("remote-blackout-button")?.addEventListener("click", toggleBlackout);
  $("fresh-gesture-button")?.addEventListener("click", requestFreshGesture);

  $$("[data-control]").forEach((input) => {
    input.addEventListener("input", () => queueControl(input.dataset.control, Number(input.value) / 100));
  });
  $$("[data-preset]").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  $$("[data-feedback]").forEach((button) => {
    button.addEventListener("click", () => sendFeedback(button.dataset.feedback, button.dataset.value));
  });
  $("feedback-note-button")?.addEventListener("click", () => {
    const note = $("feedback-note").value.trim();
    if (note) sendFeedback("operator_note", 0, note);
  });
  $("remote-feedback-note-button")?.addEventListener("click", () => {
    const note = $("remote-feedback-note").value.trim();
    if (note) sendFeedback("operator_note", 0, note);
  });
  $("remote-note-open")?.addEventListener("click", () => {
    $("remote-note-box").classList.toggle("hidden");
    if (!$("remote-note-box").classList.contains("hidden")) $("remote-feedback-note").focus();
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

  $("rescan-system-button")?.addEventListener("click", rescanSystem);
  $("system-rescan-button")?.addEventListener("click", rescanSystem);
  $("save-audio-device-button")?.addEventListener("click", saveAudioDevice);
  $("spotify-connect-button")?.addEventListener("click", connectSpotify);
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
    if (!app.pointer.moved && !app.pointer.fixtureId) {
      const position = roomPositionFromCanvas(canvas, app.rigView, event.clientX, event.clientY, Number($("target-z")?.value || 1.2));
      $("target-x").value = position[0].toFixed(3);
      $("target-y").value = position[1].toFixed(3);
      if (app.rigView !== "plan") $("target-z").value = position[2].toFixed(3);
      solveTargetFromFields();
    }
    app.pointer = { dragging: false, moved: false, fixtureId: null };
  });
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
  if (/^[1-5]$/.test(event.key)) {
    setPage(["performance", "rig", "audio", "memory", "system"][Number(event.key) - 1]);
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
}

initialize();
