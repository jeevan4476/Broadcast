"""Deterministic tests for GitHub Actions CI workflow, Bun policy, and shell scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
UI_PACKAGE_JSON = REPO_ROOT / "ui" / "package.json"
UI_NPMRC = REPO_ROOT / "ui" / ".npmrc"
SHELL_SCRIPTS = [
    REPO_ROOT / "nv-broadcast",
    REPO_ROOT / "install-deps.sh",
]


class TestCIWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(CI_WORKFLOW.is_file(), f"Missing CI workflow: {CI_WORKFLOW}")
        self.workflow_text = CI_WORKFLOW.read_text(encoding="utf-8")

    def test_triggers_and_runner(self) -> None:
        self.assertRegex(self.workflow_text, r"(?m)^on:\s*$")
        self.assertRegex(self.workflow_text, r"(?m)^\s+push:\s*$")
        self.assertRegex(self.workflow_text, r"(?m)^\s+pull_request:\s*$")
        self.assertIn("runs-on: ubuntu-latest", self.workflow_text)

    def test_toolchain_setup_steps(self) -> None:
        self.assertIn("uses: actions/checkout@v4", self.workflow_text)
        self.assertIn("uses: actions/setup-python@v5", self.workflow_text)
        self.assertIn('python-version: "3.12"', self.workflow_text)
        self.assertIn("pip install pillow", self.workflow_text)
        self.assertIn("uses: oven-sh/setup-bun@v2", self.workflow_text)
        self.assertIn('bun-version: "1.4.2"', self.workflow_text)

    def test_lockfile_enforcement_step(self) -> None:
        self.assertIn("package-lock.json", self.workflow_text)
        self.assertIn("yarn.lock", self.workflow_text)
        self.assertIn("pnpm-lock.yaml", self.workflow_text)
        self.assertIn(".git", self.workflow_text)
        self.assertIn("node_modules", self.workflow_text)

    def test_python_bun_and_shellcheck_checks(self) -> None:
        self.assertIn(
            "python3 -m unittest discover -s tests -p 'test_*.py' -v",
            self.workflow_text,
        )
        self.assertIn('Bun.Transpiler({ loader: "js" })', self.workflow_text)
        self.assertIn(".scan(", self.workflow_text)
        self.assertIn("bun test", self.workflow_text)
        self.assertIn("bash -n nv-broadcast install-deps.sh", self.workflow_text)
        self.assertIn("shellcheck nv-broadcast install-deps.sh", self.workflow_text)

    def test_workflow_never_invokes_npm_or_npx(self) -> None:
        self.assertIsNone(
            re.search(r"\b(?:npm|npx)\b", self.workflow_text),
            "CI workflow must not invoke npm or npx",
        )


class TestBunEnforcement(unittest.TestCase):
    def test_ui_package_json_enforces_bun(self) -> None:
        self.assertTrue(UI_PACKAGE_JSON.is_file(), f"Missing {UI_PACKAGE_JSON}")
        pkg = json.loads(UI_PACKAGE_JSON.read_text(encoding="utf-8"))

        package_manager = pkg.get("packageManager", "")
        self.assertTrue(
            isinstance(package_manager, str) and package_manager.startswith("bun@"),
            f"Expected Bun packageManager, got {package_manager!r}",
        )

        engines = pkg.get("engines", {})
        self.assertIsInstance(engines, dict)
        self.assertIn("bun", engines)
        for forbidden in ("npm", "yarn", "pnpm"):
            self.assertIn(forbidden, engines)
            self.assertNotEqual(engines[forbidden], "*")

        preinstall = pkg.get("scripts", {}).get("preinstall", "")
        self.assertIn("bun/*", preinstall)
        self.assertIn("npm_config_user_agent", preinstall)

    def test_ui_npmrc_sets_engine_strict(self) -> None:
        self.assertTrue(UI_NPMRC.is_file(), f"Missing {UI_NPMRC}")
        lines = [
            line.strip()
            for line in UI_NPMRC.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("engine-strict=true", lines)

    def test_no_forbidden_lockfiles_in_repo(self) -> None:
        forbidden_names = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
        found: list[Path] = []
        for root, dirs, files in os.walk(REPO_ROOT, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules"}]
            for filename in files:
                if filename in forbidden_names:
                    found.append(Path(root) / filename)
        self.assertEqual(found, [], f"Forbidden lockfiles present: {found}")


class TestShellScripts(unittest.TestCase):
    def test_scripts_exist_and_are_executable(self) -> None:
        for script in SHELL_SCRIPTS:
            self.assertTrue(script.is_file(), f"Missing script: {script}")
            self.assertTrue(os.access(script, os.X_OK), f"Script is not executable: {script}")

    def test_bash_syntax_check_passes(self) -> None:
        proc = subprocess.run(
            ["bash", "-n", *(str(script) for script in SHELL_SCRIPTS)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"bash -n failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertEqual(proc.stderr, "")

    def test_install_deps_help(self) -> None:
        proc = subprocess.run(
            [str(REPO_ROOT / "install-deps.sh"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--fedora", proc.stdout)
        self.assertIn("--load-cam", proc.stdout)
        self.assertIn("--venv", proc.stdout)


if __name__ == "__main__":
    unittest.main()
