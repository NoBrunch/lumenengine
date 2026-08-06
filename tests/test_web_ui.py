from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OperatorInterfaceContractTests(unittest.TestCase):
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
        stylesheet = (ROOT / "web" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('class="structure-time-readout"', script)
        self.assertIn("Opened the next song awaiting review", script)
        self.assertIn('data-timeline-review="unreviewed"', script)
        self.assertIn(".sequence-editor-panel {", stylesheet)
        self.assertIn("grid-column: 1 / 3;", stylesheet)
        self.assertIn(".structure-time-readout { min-width: 13rem; }", stylesheet)

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


if __name__ == "__main__":
    unittest.main()
