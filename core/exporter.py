"""
Stage 7b — Self-verifying test suite export.
[Jinja2 Template] passed journeys -> standalone pytest-playwright files
[pytest-playwright] RUN EXPORTED TESTS (no AI) — drop any file that fails.
100% of delivered tests run green on their own. Zero lock-in: plain code.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from . import config
from .contracts import TestCase

# pytest exit codes we can reason about (see pytest docs, "Exit codes")
_EXIT_OK = 0            # every test passed
_EXIT_TESTS_FAILED = 1  # ran, but some test failed
_EXIT_USAGE = 4         # bad CLI args — e.g. --base-url without pytest-base-url
_EXIT_NO_TESTS = 5      # nothing collected


class Exporter:
    def __init__(self, run_dir: Path, target_url: str) -> None:
        self.run_dir = run_dir
        self.target_url = target_url.rstrip("/")
        self.jinja = Environment(
            loader=FileSystemLoader(str(config.JINJA_DIR)), autoescape=False)
        self.out_dir = run_dir / "exported_tests"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48]

    def export(self, passed: list[TestCase]) -> list[str]:
        files: list[str] = []
        tpl = self.jinja.get_template("exported_test.py.j2")
        for tc in passed:
            body = tpl.render(tc=tc, target_url=self.target_url,
                              slug=self._slug(tc.test_id + "_" + tc.title))
            f = self.out_dir / f"test_{self._slug(tc.test_id + '_' + tc.title)}.py"
            f.write_text(body, encoding="utf-8")
            files.append(f.name)
        return files

    def self_verify(self, files: list[str]) -> tuple[list[str], str]:
        """Run the just-exported suite and keep only files that truly passed.

        Verification relies on pytest's machine-readable JUnit XML report and on
        its exit code — never on scraping console text. Scraping silently reports
        a *green* suite whenever pytest could not actually run (wrong interpreter,
        missing plugin, usage error), which would ship unverified tests as
        "proven". Anything we could not positively confirm as passing is dropped.
        """
        if not files:
            return [], "no tests exported"
        # diagnostics live beside pytest_self_run.log, NOT inside exported_tests —
        # the delivered test folder must contain only runnable .py files
        report = self.run_dir / "pytest_self_run.xml"
        report.unlink(missing_ok=True)
        # sys.executable, not "python3": on Windows the launcher is "python"/"py"
        # and the tests belong to the interpreter running QArena.
        cmd = [sys.executable, "-m", "pytest", str(self.out_dir), "-q",
               "--base-url", self.target_url, "-p", "no:cacheprovider",
               "--tb=line", f"--junitxml={report}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=300,
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            log = (proc.stdout or "") + (proc.stderr or "")
            rc = proc.returncode
        except Exception as exc:
            return [], f"pytest failed to run: {exc.__class__.__name__}: {exc}"

        passed_files, note = self._read_junit(report)
        if rc == _EXIT_USAGE:
            note = (f"pytest could not start (exit {rc}) — most likely "
                    f"pytest-playwright/pytest-base-url is not installed in "
                    f"{sys.executable}. NOTHING is shipped as verified.\n{note}")
        elif rc == _EXIT_NO_TESTS:
            note = f"pytest collected no tests (exit {rc}).{note}"
        elif rc not in (_EXIT_OK, _EXIT_TESTS_FAILED):
            note = f"pytest did not complete normally (exit {rc}).{note}"

        keep: list[str] = []
        for name in files:
            ok = name in passed_files
            if ok:
                keep.append(name)
            # dropped files stay on disk with a .dropped suffix for inspection
            try:
                (self.out_dir / name).rename(self.out_dir / (name + ".dropped"))
            except OSError:
                pass
        dropped_note = (f"kept {len(keep)}/{len(files)} exported tests "
                        f"(only positively-passing files are delivered)")
        return keep, f"{dropped_note}\n{note}\n{log[-1800:]}"

    def _read_junit(self, report: Path) -> tuple[set[str], str]:
        """Return {exported filename} whose every testcase passed, plus a note."""
        if not report.exists():
            return set(), ("no JUnit report was produced — the suite did not run, "
                           "so no test is claimed as verified")
        try:
            root = ET.parse(report).getroot()
        except Exception as exc:
            return set(), f"JUnit report unreadable ({exc})"
        suites = root.iter("testsuite") if root.tag != "testsuite" else [root]
        by_file: dict[str, list[bool]] = {}
        for suite in suites:
            for case in suite.iter("testcase"):
                fname = Path(case.get("file") or "").name
                if not fname:
                    continue
                bad = any(case.find(t) is not None for t in ("failure", "error"))
                skipped = case.find("skipped") is not None
                by_file.setdefault(fname, []).append(not bad and not skipped)
        return ({f for f, results in by_file.items() if results and all(results)},
                f"JUnit report parsed: {len(by_file)} file(s) executed")
