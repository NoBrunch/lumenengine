from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LumenWebAccessDeploymentTests(unittest.TestCase):
    def test_deployment_script_is_explicit_and_keeps_secrets_local(self) -> None:
        script_path = ROOT / "scripts/lumen-web-access"
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('CLOUDFLARED_VERSION="2026.8.2"', script)
        self.assertIn("CLOUDFLARED_SHA256=", script)
        self.assertIn("--token-file", script)
        self.assertIn("read -r -s", script)
        self.assertIn("--host 0.0.0.0", script)
        self.assertIn("ProtectSystem=strict", script)
        self.assertIn("Restart=always", script)
        self.assertIn("cloudflareaccess\\.com", script)
        self.assertIn("direct local-network access", script)
        self.assertIn("write_stdin_file 0600", script)
        self.assertNotIn("install -m 0600 /dev/stdin", script)
        self.assertNotIn("--token ey", script)

    def test_status_is_read_only_before_configuration(self) -> None:
        result = subprocess.run(
            [str(ROOT / "scripts/lumen-web-access"), "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Lumen web access", result.stdout)
        self.assertNotIn("tunnel-token", result.stdout)

    def test_guide_covers_full_authority_and_access_first(self) -> None:
        guide = (ROOT / "docs/lumen-web-access.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("complete Lumen operator console", guide)
        self.assertIn("full operational authority", guide)
        self.assertIn("owner's exact email address", guide)
        self.assertIn("Protect with Access", guide)
        self.assertIn("before publishing the tunnel route", guide)
        self.assertIn("never paste that token into chat", guide)
        self.assertNotIn("port forward", guide.casefold())
