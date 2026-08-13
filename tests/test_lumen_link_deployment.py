from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LumenLinkDeploymentTests(unittest.TestCase):
    def test_worker_template_is_valid_and_truthful(self) -> None:
        template = json.loads(
            (ROOT / "config/lumen-link/worker.wsl.example.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(template["schema"], "lumen_link_worker_config_v1")
        self.assertEqual(template["listen"]["port"], 8765)
        self.assertEqual(
            template["listen"]["allowed_clients"], ["192.168.50.2"]
        )
        self.assertEqual(
            template["execution"]["job_types"],
            ["teacher.edmformer", "teacher.songformer", "student.train"],
        )
        self.assertEqual(template["execution"]["gated_job_types"], [])
        self.assertEqual(template["execution"]["maximum_parallel_jobs"], 6)
        self.assertEqual(
            template["authentication"]["scheme"], "hmac-sha256"
        )
        self.assertEqual(template["python"]["core"], "3.12.8")
        self.assertEqual(template["python"]["edmformer"], "3.12.8")
        self.assertEqual(template["python"]["songformer"], "3.10.20")

    def test_shell_deployment_scripts_parse(self) -> None:
        for relative in ("scripts/lumen-link", "scripts/lumen-link-wsl"):
            result = subprocess.run(
                ["bash", "-n", str(ROOT / relative)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_lumen_side_defaults_are_read_only(self) -> None:
        network = subprocess.run(
            [str(ROOT / "scripts/lumen-link"), "network"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(network.returncode, 0, network.stderr)
        self.assertIn("DRY-RUN:", network.stdout)
        self.assertIn("192.168.50.2/24", network.stdout)
        self.assertIn("ipv4.never-default", network.stdout)
        self.assertIn("connection.autoconnect-priority", network.stdout)
        self.assertIn("connection.autoconnect-retries", network.stdout)
        self.assertNotIn("--apply", network.stdout)

        pairing = subprocess.run(
            [str(ROOT / "scripts/lumen-link"), "pair"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(pairing.returncode, 0, pairing.stderr)
        self.assertIn("DRY-RUN:", pairing.stdout)
        self.assertIn("value never displayed", pairing.stdout)

        script = (ROOT / "scripts/lumen-link").read_text(encoding="utf-8")
        self.assertIn(
            'expected = ["teacher.edmformer", "teacher.songformer", "student.train"]',
            script,
        )
        self.assertIn("not ready for every Lumen Link job", script)

    def test_wsl_service_invokes_actual_cli_contract(self) -> None:
        script = (ROOT / "scripts/lumen-link-wsl").read_text(encoding="utf-8")
        self.assertIn('link-node "${worker_arguments[@]}"', script)
        self.assertIn('"--max-memory-gib"', script)
        self.assertIn('"--maximum-parallel-jobs"', script)
        self.assertNotIn("link-worker --config", script)
        self.assertIn("CORE_PYTHON_VERSION=3.12.8", script)
        self.assertIn("SONGFORMER_PYTHON_VERSION=3.10.20", script)
        self.assertIn("PYENV_REVISION=v2.6.27", script)
        self.assertIn(".local/share/lumen-link/pyenv", script)
        self.assertIn('"$pinned_python" -m venv', script)
        self.assertIn("verify_running_capabilities", script)
        self.assertIn("Worker service is verified but STOPPED", script)
        self.assertIn("startup_update", script)
        self.assertIn("update_if_needed", script)
        self.assertIn('git -C "$PROJECT_ROOT" fetch --quiet origin main', script)
        self.assertIn('queue.get("queued")', script)
        self.assertIn("must drain before update", script)
        self.assertIn('git -C "$PROJECT_ROOT" pull --ff-only', script)
        self.assertIn("startup-finalize --apply", script)
        self.assertIn('systemctl --user stop lumen-link-worker.service', script)
        self.assertIn(
            'expected = ["teacher.edmformer", "teacher.songformer", "student.train"]',
            script,
        )
        self.assertNotIn(
            'job_types") != ["teacher.edmformer"]', script
        )
        service = (
            ROOT / "config/lumen-link/lumen-link-worker.service"
        ).read_text(encoding="utf-8")
        self.assertIn("scripts/lumen-link-wsl run", service)
        self.assertIn("UMask=0077", service)
        self.assertIn("Restart=always", service)

        startup_without_apply = subprocess.run(
            [str(ROOT / "scripts/lumen-link-wsl"), "startup"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(startup_without_apply.returncode, 2)
        self.assertIn("startup requires --apply", startup_without_apply.stderr)

    def test_windows_setup_has_explicit_apply_and_restricted_firewall(self) -> None:
        script = (ROOT / "scripts/lumen-link-windows.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("[switch]$Apply", script)
        self.assertIn('if (-not $Apply)', script)
        self.assertIn("Lumen Link Dashboard.lnk", script)
        self.assertIn("Lumen Link Dashboard.url", script)
        self.assertIn('$DashboardAddress = "127.0.0.1"', script)
        self.assertIn("$DashboardLauncher", script)
        self.assertIn("function Wait-LumenLinkWorker", script)
        self.assertIn("desktop shortcut checking for a safe source update", script)
        self.assertIn("Invoke-WebRequest -UseBasicParsing", script)
        self.assertIn("Lumen Link startup failed", script)
        self.assertIn(
            '$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File',
            script,
        )
        self.assertIn('$shortcut.IconLocation = "$DashboardIcon,0"', script)
        self.assertIn("./scripts/lumen-link-wsl startup --apply", script)
        self.assertIn("./scripts/lumen-link-wsl update-if-needed --apply", script)
        self.assertIn("checking Git, configuration and research deployment", script)
        self.assertIn("checking for a newer idle-safe Lumen revision", script)
        self.assertIn("AddMinutes(5)", script)
        self.assertIn("Stop-ScheduledTask -TaskName $TaskName", script)
        self.assertIn("Start-ScheduledTask -TaskName $TaskName", script)
        self.assertIn("Start-Sleep -Seconds 30", script)
        self.assertIn("-RemoteAddress $LumenAddress", script)
        self.assertIn("-LocalAddress $ThreadripperAddress", script)
        self.assertIn("Lumen Link WSL Network Refresh", script)

    def test_handoff_names_privacy_and_authority_checkpoints(self) -> None:
        handoff = (ROOT / "docs/lumen-link-codex-handoff.md").read_text(
            encoding="utf-8"
        )
        for expected in (
            "Pause before Windows administrator or Linux",
            "Never request, print, commit",
            "teacher.edmformer",
            "teacher.songformer",
            "student.train",
            "~/lumenengine",
            "mode `600`",
        ):
            self.assertIn(expected, handoff)
        self.assertIn("There is no per-song selector", handoff)
        self.assertIn("press **Disable link**", handoff)
        self.assertIn("candidate remains inactive", handoff)

        guide = (ROOT / "docs/lumen-link-wsl-deployment.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("up to six teacher slots", guide)
        self.assertIn("Other queued", guide)
        self.assertIn("Lumen Link Dashboard", guide)
        self.assertNotIn("select one completed EDM recording", guide)
        self.assertIn("physical acceptance work", guide)
        self.assertIn("all three job types as `READY`", guide)
        self.assertIn("not the writable Lumen database", guide)
        self.assertIn("candidate model and evaluation report", guide)
        self.assertNotIn("remain clearly gated", guide)


if __name__ == "__main__":
    unittest.main()
