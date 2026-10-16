"""Regression tests for the Windows encoding failure (Stage 7a).

Original incident: a run against a live site got two suspects through all three
replay checks, then died with

    UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
    in position 75: character maps to <undefined>

and every verified finding vanished. Root cause: report files were written with
the platform-default text encoding. On Windows that is the ANSI code page
(cp1252/"charmap"), which can encode "—" (0x97) but not "→" (U+2192) — and the
judge builds network evidence as f"{method} {url} → {status}".

The tests below reproduce that condition platform-independently by running a
driver with locale coercion disabled, so the *default* encoding is never UTF-8.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARROW = "\u2192"          # the exact character from the traceback
EMDASH = "\u2014"

# A student-machine-like environment: no UTF-8 mode, no locale coercion, C locale.
# getpreferredencoding() then returns ANSI_X3.4-1968 (Windows: cp1252/"charmap"),
# which is precisely the condition that broke the original run.
NON_UTF8_ENV = {"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C"}

DRIVER = '''
import json, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from core.contracts import (ExpectedContract, PossibleBug, SignalHit,
                            StepEvidence, StepSpec)
from core.llm import LLMPolicy
from core.reporter import Reporter

run_dir = Path(sys.argv[1]); run_dir.mkdir(parents=True, exist_ok=True)
ev = StepEvidence(step=2, action="goto", target="/cart", status="failed",
                  url_before="https://target.test/",
                  url_after="https://target.test/cart",
                  http_failures=[{{"method": "GET",
                                   "url": "https://target.test/cart",
                                   "status": 404}}],
                  broken_items=[{{"kind": "link", "name": "Basket",
                                  "detail": "/cart {arrow} 404"}}],
                  screenshot="steps/TC_007_step2.png")
pb = PossibleBug(journey="Route '/cart' serves a real page", test_id="TC_007",
                 step_index=1, group="http:/cart", score=0.95,
                 signals=[SignalHit(signal=2, name="Network Failures", weight=0.5,
                                    evidence="GET https://target.test/cart {arrow} 404"),
                          SignalHit(signal=1, name="Console Errors", weight=0.45,
                                    evidence="Failed to load resource {{/cart}}")],
                 summary="S2 Network Failures: GET https://target.test/cart {arrow} 404",
                 repro_steps=[StepSpec(step=1, action="goto", target="/")],
                 oracle=[ExpectedContract().model_dump()], evidence=ev)
filed = Reporter(run_dir, LLMPolicy()).file_bugs([(pb, "3/3")])
print(json.dumps({{"filed": [b.bug_id for b in filed]}}))
'''


def _run_driver(tmp_path: Path, env_extra: dict) -> subprocess.CompletedProcess:
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(root=str(ROOT), arrow=ARROW), encoding="utf-8")
    env = {**os.environ, **env_extra}
    return subprocess.run([sys.executable, str(driver), str(tmp_path / "run")],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=120)


def test_reporter_survives_a_non_utf8_default_encoding(tmp_path):
    """The exact failure: default encoding is not UTF-8 (Windows cp1252)."""
    proc = _run_driver(tmp_path, NON_UTF8_ENV)
    assert proc.returncode == 0, (
        f"reporter crashed under a non-UTF-8 default encoding "
        f"(the original UnicodeEncodeError):\n{proc.stderr[-2000:]}")
    assert json.loads(proc.stdout.strip().splitlines()[-1])["filed"] == ["QA-01"]


def test_report_keeps_unicode_and_is_valid_utf8(tmp_path):
    """Legitimate evidence characters must survive — not be stripped."""
    proc = _run_driver(tmp_path, NON_UTF8_ENV)
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = tmp_path / "run"
    md = (run_dir / "bug_reports" / "QA-01.md").read_text(encoding="utf-8")
    assert ARROW in md or EMDASH in md, "expected the evidence characters to survive"
    # a report must never be silently mangled into replacements
    assert "\ufffd" not in md
    filed = json.loads((run_dir / "bugs_filed.json").read_text(encoding="utf-8"))
    assert [b["bug_id"] for b in filed] == ["QA-01"]
    assert filed[0]["reproduced_count"] == "3/3"


def test_verified_findings_are_persisted_with_the_report(tmp_path):
    """bugs_filed.json exists even though the report writing paths changed."""
    proc = _run_driver(tmp_path, NON_UTF8_ENV)
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = tmp_path / "run"
    assert (run_dir / "bugs_filed.json").exists()
    body = (run_dir / "bugs_filed.json").read_text(encoding="utf-8")
    assert "http:/cart" not in body  # sanity: it holds filings, not suspect groups
    assert "QA-01" in body


def test_every_text_file_write_declares_an_encoding():
    """Static guard for the whole bug class.

    Any text-mode open()/read_text()/write_text() in the application code that
    omits `encoding=` re-introduces the Windows failure. This test fails on the
    pre-fix code (core/reporter.py:108) and keeps it from coming back.
    """
    offenders: list[str] = []
    for path in sorted(list((ROOT / "core").rglob("*.py"))
                       + list((ROOT / "app").rglob("*.py"))
                       + list((ROOT / "scorer").rglob("*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            if name not in ("open", "read_text", "write_text"):
                continue
            if name == "open":
                # binary modes carry no encoding by design
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if "b" in mode:
                    continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                rel = path.relative_to(ROOT)
                offenders.append(f"{rel}:{node.lineno} -> {name}() without encoding=")
    assert not offenders, (
        "text I/O without an explicit encoding (breaks on Windows/cp1252):\n  "
        + "\n  ".join(offenders))


def test_exported_suite_runner_uses_this_interpreter():
    """`python3` does not exist on a stock Windows box; the exported suite must
    be executed by the interpreter running QArena (checked on the AST, so
    comments and docstrings cannot mask the real command)."""
    tree = ast.parse((ROOT / "core" / "exporter.py").read_text(encoding="utf-8"))
    commands: list[ast.List] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            elts = node.value.elts
            if any(isinstance(e, ast.Constant) and e.value == "-m" for e in elts):
                commands.append(node.value)
    assert commands, "could not find the pytest command construction"
    for cmd in commands:
        first = cmd.elts[0]
        is_sys_executable = (isinstance(first, ast.Attribute)
                             and first.attr == "executable"
                             and isinstance(first.value, ast.Name)
                             and first.value.id == "sys")
        assert is_sys_executable, (
            f"pytest is launched with {ast.unparse(first)!r}; "
            f"use sys.executable so it works on Windows too")
    # and subprocess output must be decoded explicitly (console code pages vary)
    src = (ROOT / "core" / "exporter.py").read_text(encoding="utf-8")
    assert 'encoding="utf-8"' in src, "subprocess output needs an explicit encoding"


@pytest.mark.parametrize("module", ["core/reporter.py", "core/orchestrator.py",
                                    "core/exporter.py", "core/explorer.py",
                                    "app/main.py"])
def test_no_bare_locale_reads_in_hot_files(module):
    """Files that read artifacts back must pin UTF-8 too (results.json and
    run.log carry '→' written by the judge/tester)."""
    src = (ROOT / module).read_text(encoding="utf-8")
    assert "read_text())" not in src.replace(" ", "")
