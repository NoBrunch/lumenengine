from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OperatorInterfaceContractTests(unittest.TestCase):
    def test_phone_remote_has_no_spotify_playback_controls(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        for element_id in (
            "remote-spotify-previous",
            "remote-spotify-play",
            "remote-spotify-next",
            "remote-spotify-refresh",
            "remote-spotify-playlist",
            "remote-spotify-play-playlist",
        ):
            self.assertNotIn(f'id="{element_id}"', html)
            self.assertNotIn(f'$("{element_id}")', script)
        self.assertNotIn("function renderRemoteSpotify", script)

    def test_remote_layout_and_exact_strobe_teaching_match_operator_request(
        self,
    ) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        for removed_id in (
            "remote-gesture",
            "remote-song-teaching",
            "remote-link-card",
            "remote-link-state",
        ):
            self.assertNotIn(f'id="{removed_id}"', html)
        self.assertNotIn("Teach a specific song action", html)
        self.assertNotIn('id="remote-action-button"', html)
        self.assertNotIn('id="remote-action-label"', html)
        self.assertNotIn("Shape the response", html[html.index('id="remote-app"'):])
        self.assertIn('class="embedded-shape-controls"', html)
        for group in ("movers", "center"):
            for prefix in ("remote", "rehearsal"):
                self.assertIn(f'id="{prefix}-{group}-strobe"', html)
                self.assertIn(f'id="{prefix}-{group}-strobe-number"', html)
        self.assertIn('max="255"', html)
        self.assertIn('api("/api/strobe-control"', script)
        self.assertIn("settled,", script)
        self.assertIn("700,", script)
        self.assertIn("interaction_unix_ms", script)

    def test_rehearsal_palette_and_color_studio_are_removed(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

        for element_id in (
            "rehearsal-palette",
            "center-color-pattern",
            "color-wheel-canvas",
            "color-brightness-input",
            "color-name-input",
            "palette-name-input",
            "palette-color-picker",
            "palette-save-button",
        ):
            self.assertNotIn(f'id="{element_id}"', html)
        self.assertNotIn("Color Studio", html)

    def test_live_work_status_uses_durable_totals_and_active_preparation_polling(
        self,
    ) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn("<span>IMPORTED</span>", html)
        self.assertIn('id="link-queued-detail"', html)
        self.assertIn("queue.recent_imports", script)
        self.assertIn("verified lifetime total", script)
        self.assertIn("queue.failed_attention", script)
        self.assertIn("retained historical", script)
        self.assertIn("void refreshResearch();", script)
        self.assertIn("app.bootstrap?.research?.preparation?.running", script)
        self.assertIn("app.bootstrap?.research?.preparation?.pending", script)
        self.assertIn(
            "app.bootstrap?.research?.student_preparation?.running", script
        )
        self.assertIn(
            'researchRunning ? 10 : app.page === "audio" ? 50 : 300',
            script,
        )
        self.assertIn("research.preparation?.started_unix_ms", script)
        self.assertIn("Preparing student-training snapshot", script)
        self.assertIn("studentPreparationRunning", script)
        self.assertIn("student_link?.running", script)
        self.assertIn("student_local_validation", script)
        self.assertIn(
            "researchServerTask(research) || app.operatorTask", script
        )

    def test_long_tasks_have_a_console_wide_live_status_region(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="operator-task"', html)
        self.assertIn('role="status"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn("function beginOperatorTask", script)
        self.assertIn("function researchServerTask", script)
        self.assertIn("Training and validating the structure model", script)
        self.assertIn("Analyzing recordings with EDMFormer", script)

    def test_hidden_task_strip_cannot_displace_workspace_or_footer(self) -> None:
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('"task"\n    "workspace"\n    "statusbar"', stylesheet)
        self.assertIn(".operator-task { grid-area: task; }", stylesheet)
        self.assertIn(".desktop-workspace { grid-area: workspace; }", stylesheet)
        self.assertIn(".desktop-statusbar { grid-area: statusbar; }", stylesheet)
        self.assertIn(
            "grid-template-rows: 51px 27px 72px auto minmax(0, 1fr) 25px",
            stylesheet,
        )

    def test_every_button_receives_visible_press_confirmation(self) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("function confirmButtonPress", script)
        self.assertIn('document.addEventListener("click"', script)
        self.assertIn("button.press-confirmed", stylesheet)
        self.assertIn("button.task-pending", stylesheet)

    def test_small_fixed_pixel_fonts_were_raised_two_pixels(self) -> None:
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )
        pixel_sizes = [
            int(value)
            for value in re.findall(r"font-size:\s*(\d+)px", stylesheet)
        ]
        shorthand_pixel_sizes = [
            int(value)
            for value in re.findall(r"font:\s*(\d+)px", stylesheet)
        ]

        self.assertTrue(pixel_sizes)
        self.assertTrue(shorthand_pixel_sizes)
        self.assertGreaterEqual(min(pixel_sizes), 9)
        self.assertGreaterEqual(min(shorthand_pixel_sizes), 8)
        self.assertIn("body { font-size: 17px; }", stylesheet)

    def test_compute_node_decision_is_a_durable_project_document(self) -> None:
        document = (
            ROOT / "docs" / "threadripper-compute-node.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Lumen PC — live authority", document)
        self.assertIn("Threadripper — offline compute node", document)
        self.assertIn("must never share a writable SQLite database", document)
        self.assertIn("held-out validation gates", document)

    def test_rejected_student_report_does_not_prescribe_blind_retraining(
        self,
    ) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

        self.assertIn(
            "held-out qualification did not pass",
            script,
        )
        self.assertIn(
            "repeating training with identical inputs produces the same "
            "qualification evidence",
            script,
        )
        self.assertIn(
            "functional form was not present in this candidate's trusted "
            "teacher data",
            script,
        )
        self.assertIn('id="research-song-results"', html)
        self.assertIn("boundaryMetrics.event_f1", script)
        self.assertIn("energy.balanced_accuracy", script)
        self.assertNotIn(
            "Analyze more complete songs before retraining",
            script,
        )

    def test_audio_input_panel_mirrors_live_expressive_state(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('class="audio-expression-state"', html)
        self.assertIn('id="audio-expression-gesture"', html)
        self.assertIn('id="audio-expression-confidence"', html)
        for axis in ("energy", "tension", "motion", "intimacy"):
            self.assertIn(f'id="audio-meter-{axis}"', html)
            self.assertIn(f'id="audio-value-{axis}"', html)
        self.assertIn('setText("audio-expression-gesture", gesture)', script)
        self.assertIn(
            'setText("audio-expression-confidence", '
            '`${percent(confidence)} CONFIDENCE`)',
            script,
        )
        self.assertIn('setWidth(`audio-meter-${name}`, expression[name])', script)

    def test_timeline_editor_has_readable_times_and_review_advances(self) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('class="structure-time-readout"', script)
        self.assertIn("Opened the next song awaiting review", script)
        self.assertIn('data-timeline-review="unreviewed"', script)
        self.assertIn("QUALITY CHECKS", script)
        self.assertIn("This quality queue does not block training", markup)
        self.assertIn(
            "Training may use a technically valid completed teacher result",
            markup,
        )
        self.assertIn(".sequence-editor-panel {", stylesheet)
        self.assertIn("grid-column: 1 / 3;", stylesheet)
        self.assertIn(".structure-time-readout { min-width: 13rem; }", stylesheet)

    def test_sequence_and_gesture_editors_expose_simple_workflows(self) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

        self.assertIn('data-sequence-template="build"', markup)
        self.assertIn('class="sequence-step-advanced"', script)
        self.assertIn('data-step-duplicate="${index}"', script)
        self.assertIn('data-step-move="-1"', script)
        self.assertIn('data-play-gesture-routine="${escapeHtml(routine.id)}"', script)
        self.assertIn("2 · Movements Lumen may choose", markup)

    def test_desktop_panels_float_move_and_resize_from_every_edge(self) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("function installPanelWorkspace", script)
        self.assertIn("function beginPanelMove", script)
        self.assertIn("function beginPanelResize", script)
        self.assertIn("function panelResizeEdge", script)
        self.assertIn('edge.includes("w")', script)
        self.assertIn('edge.includes("n")', script)
        self.assertIn("lumen.panel.${key}.v2", script)
        self.assertIn(".panel-floating {", stylesheet)

    def test_lumen_link_has_a_complete_operator_dashboard(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('data-page="link"', html)
        self.assertIn('data-nav="link"', html)
        for element_id in (
            "link-state-badge",
            "link-bridge",
            "link-jobs",
            "link-cpu-meter",
            "link-memory-meter",
            "link-disk-meter",
            "link-gpu-meter",
            "link-events",
            "link-capabilities",
            "link-setup-commands",
            "link-test-button",
            "link-enable-button",
            "link-pause-button",
            "link-action-feedback",
            "link-engine-grid",
            "link-axis-authority",
            "link-artifact-flow",
            "link-candidate-state",
            "link-validation-state",
            "link-approved-axes",
            "link-activation-state",
            "link-model-blockers",
        ):
            self.assertIn(f'id="{element_id}"', html)
        for removed_remote_link_id in (
            "remote-link-state",
            "remote-link-node",
            "remote-link-progress",
            "remote-link-model",
        ):
            self.assertNotIn(f'id="{removed_remote_link_id}"', html)
        self.assertIn("function renderLink", script)
        self.assertNotIn('"/api/link/status?summary=1"', script)
        self.assertIn('api("/api/link/status")', script)
        self.assertIn("api(`/api/link/${action}`", script)
        self.assertIn('runLinkAction("test"', script)
        self.assertIn('app.link?.enabled ? "disable" : "enable"', script)
        self.assertIn('app.link?.paused ? "resume" : "pause"', script)
        self.assertIn('enabled ? "Disable link" : "Enable link"', script)
        self.assertIn('state === "incompatible" ? "degraded"', script)
        self.assertIn("connection.detail", script)
        self.assertIn("a silently disabled control made revision drift", script)
        self.assertIn("function setLinkActionFeedback", script)
        self.assertIn("Testing…", script)
        self.assertIn("Connection test completed: ${reason}", script)
        self.assertIn('toast("Lumen Link is not ready", reason, "error")', script)
        self.assertIn("failed: ${error.message}", script)

    def test_center_fixture_preview_matches_characterized_mechanics(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="center-motion-canvas"', html)
        self.assertNotIn('id="calibration-center-motion-canvas"', html)
        self.assertIn("Center rotor · 300°", html)
        self.assertIn("Side pod A · 180°", html)
        self.assertIn("One complete movement cycle", html)
        self.assertIn("fixed ceiling base has no motion track", html)
        self.assertIn('id="motion-center-preview" class="center-motion-preview hidden"', html)
        self.assertIn("function centerPreviewCoordinates", script)
        self.assertIn("mechanics.center_rotation_deg", script)
        self.assertIn("mechanics.pod_rotation_deg", script)
        self.assertIn('"CENTER ROTATION"', script)
        self.assertIn('"POD A TILT"', script)
        self.assertIn('$("motion-center-preview")?.classList.toggle', script)
        self.assertIn('currentIndeterminate', script)
        self.assertIn('"ACTIVE"', script)
        self.assertIn("function linkJobPhase", script)
        self.assertIn("function renderLinkArtifactFlow", script)
        self.assertIn("function renderLinkPipeline", script)
        self.assertIn("Training and held-out validation", script)
        self.assertIn("Artifact verified · awaiting Standby import", script)
        self.assertIn('candidate_only: "CANDIDATE ONLY"', script)
        for stage in (
            "student_feature_preparation",
            "student_training",
            "student_validation",
            "student_artifacts",
        ):
            self.assertIn(f'"{stage}"', script)
        self.assertIn("function linkJobIsIndeterminate", script)
        self.assertIn('progressKind === "indeterminate"', script)
        self.assertIn("function linkJobImportState", script)
        self.assertIn("link.recent_imports", script)
        self.assertIn('local_import_state: "imported"', script)
        self.assertIn("queue.locally_imported", script)
        self.assertIn("Locally verified and imported into Lumen", script)
        self.assertIn("Remote result complete · local import not confirmed", script)
        self.assertIn('id="link-completed-detail"', html)
        self.assertIn("remote finished · import unconfirmed", script)

    def test_lumen_link_discloses_boundaries_and_capabilities(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn("Live audio and DMX remain on this PC", html)
        self.assertIn("never share a writable database", html)
        self.assertIn(
            "Pausing stops new dispatch; an active remote process continues",
            html,
        )
        self.assertIn("Secrets are never shown", html)
        self.assertIn("same private shared secret", html)
        self.assertIn("without dispatching a job", html)
        self.assertNotIn("identity fingerprints", html)
        self.assertNotIn("Test a bundle", html)
        self.assertIn("link.capabilities || remote.capabilities || link.job_types", script)
        self.assertIn('available ? "available" : "unavailable"', script)

    def test_lumen_link_is_responsive_and_has_keyboard_navigation(self) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn(".link-layout {", stylesheet)
        self.assertIn("@media (max-width: 1100px)", stylesheet)
        self.assertIn(".link-setup-body { grid-template-columns: 1fr; }", stylesheet)
        self.assertIn(".remote-link-path {", stylesheet)
        self.assertIn('setText("remote-link-state", statusLabel)', script)
        self.assertNotIn("app.remote && app.pollCount % 100 === 0", script)
        self.assertIn('if (/^[1-8]$/.test(event.key))', script)
        self.assertIn('"music", "link", "system"', script)
        self.assertIn("workspacePages.includes(requestedWorkspacePage)", script)
        self.assertIn(
            'grid-template-rows: 51px 27px 72px auto minmax(0, 1fr) 25px',
            stylesheet,
        )
        self.assertIn('.workspace-page[data-page="link"] .panel-note,', stylesheet)
        self.assertIn("font-size: .8rem;", stylesheet)
        self.assertIn(
            ".link-model-blockers { max-height: 5.2rem; overflow: auto;",
            stylesheet,
        )
        self.assertIn(
            ".remote-link-facts { display: grid; grid-template-columns: repeat(4, 1fr);",
            stylesheet,
        )


if __name__ == "__main__":
    unittest.main()
