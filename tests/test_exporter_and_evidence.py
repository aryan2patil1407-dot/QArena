"""Stage 7b / evidence-pack tests.

Incident (found by running the exporter): self_verify() decided pass/fail by
scraping pytest's console output for lines matching ``^(FAILED|ERROR)\\s+``.
Whenever pytest could not actually run — missing interpreter, missing
pytest-playwright, usage error — the pattern matched nothing, every file was
"kept", and the run advertised a green suite that had never executed.

The tests below pin the repaired contract: only positively-passing files ship.
"""
from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import pytest

from core.contracts import ExpectedContract, StepSpec, TestCase
from core.exporter import Exporter


def a_case(test_id: str = "TC_001", name: str = "Add to Cart") -> TestCase:
    return TestCase(test_id=test_id, journey_type="happy_path",
                    title="Add item to cart",
                    steps=[StepSpec(step=1, action="goto", target="/"),
                           StepSpec(step=2, action="click", role="button", name=name)],
                    expected=ExpectedContract(badge_equals=1))


def test_export_renders_valid_python(tmp_path):
    ex = Exporter(tmp_path, "https://target.test")
    files = ex.export([a_case()])
    assert files and files[0].endswith(".py")
    src = (tmp_path / "exported_tests" / files[0]).read_text(encoding="utf-8")
    ast.parse(src)                                     # must be runnable Python
    assert 'name="Add to Cart"' in src
    assert "expect(page.locator(\"#cart-badge\")).to_have_text(\"1\")" in src


def test_export_handles_unicode_titles(tmp_path):
    tc = a_case()
    tc.title = "Total went below zero \u2014 qty \u22125 \u2192 negative"
    ex = Exporter(tmp_path, "https://target.test")
    files = ex.export([tc])
    src = (tmp_path / "exported_tests" / files[0]).read_text(encoding="utf-8")
    ast.parse(src)
    assert "\u2192" in src


def test_unverifiable_suite_ships_nothing(tmp_path):
    """pytest cannot run (no browser) -> nothing may be presented as verified."""
    ex = Exporter(tmp_path, "http://127.0.0.1:9/")     # unroutable target
    files = ex.export([a_case()])
    keep, log = ex.self_verify(files)
    assert keep == [], f"unverified tests must not ship; got {keep}"
    assert "kept 0/1" in log
    dropped = list((tmp_path / "exported_tests").glob("*.dropped"))
    assert dropped, "dropped tests stay on disk for inspection"
    assert "JUnit" in log or "no JUnit report" in log


def test_self_verify_reports_pytest_usage_failure_honestly(tmp_path, monkeypatch):
    """Reproduces the '--base-url: unrecognized arguments' case (pytest-playwright
    missing): exit code 4, no JUnit XML, therefore no shipped test."""
    import subprocess
    import sys as _sys

    ex = Exporter(tmp_path, "https://target.test")
    files = ex.export([a_case()])

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 4,
            stdout="ERROR: usage: python -m pytest [options]\n"
                   "python -m pytest: error: unrecognized arguments: --base-url\n",
            stderr="")

    monkeypatch.setattr("core.exporter.subprocess.run", fake_run)
    keep, log = ex.self_verify(files)
    assert keep == []
    assert "could not start" in log or "did not complete" in log
    assert "NOTHING is shipped as verified" in log


def test_self_verify_keeps_only_passing_files(tmp_path, monkeypatch):
    """JUnit XML is the source of truth: one failing file is dropped, one kept."""
    import subprocess

    ex = Exporter(tmp_path, "https://target.test")
    good, bad = "test_a.py", "test_b.py"
    (tmp_path / "exported_tests" / good).write_text("", encoding="utf-8")
    (tmp_path / "exported_tests" / bad).write_text("", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        xml = next(a.split("=", 1)[1] for a in cmd if str(a).startswith("--junitxml="))
        Path(xml).write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            f'<testsuites><testsuite name="pytest" tests="2">'
            f'<testcase classname="t" name="a" file="{good}"/>'
            f'<testcase classname="t" name="b" file="{bad}">'
            f'<failure message="boom">assert</failure></testcase>'
            f'</testsuite></testsuites>', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, stdout="1 failed, 1 passed", stderr="")

    monkeypatch.setattr("core.exporter.subprocess.run", fake_run)
    keep, log = ex.self_verify([good, bad])
    assert keep == [good]
    assert "kept 1/2" in log
    assert (tmp_path / "exported_tests" / (bad + ".dropped")).exists()


def test_evidence_pack_is_self_contained(tmp_path):
    """A downloaded pack must open without QArena installed: pure files."""
    import shutil

    run_dir = tmp_path / "run_x"
    (run_dir / "bug_reports").mkdir(parents=True)
    (run_dir / "bug_reports" / "QA-01.md").write_text(
        "# QA-01 \u2014 dead route\n\nGET /cart \u2192 404\n", encoding="utf-8")
    (run_dir / "exported_tests").mkdir()
    (run_dir / "exported_tests" / "test_a.py").write_text("def test_a():\n    pass\n",
                                                          encoding="utf-8")
    (run_dir / "bugs_filed.json").write_text(
        json.dumps([{"bug_id": "QA-01"}]), encoding="utf-8")

    archive = shutil.make_archive(str(run_dir), "zip", run_dir)
    assert Path(archive).exists()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "bug_reports/QA-01.md" in names
        assert "exported_tests/test_a.py" in names
        assert "\u2192" in zf.read("bug_reports/QA-01.md").decode("utf-8")
    _ = pytest  # keep the import meaningful for readers
