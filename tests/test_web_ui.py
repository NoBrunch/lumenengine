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


if __name__ == "__main__":
    unittest.main()
