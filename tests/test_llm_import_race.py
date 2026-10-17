"""Regression test for LiteLLM's first-import race (upstream #46142).

LiteLLM 1.104.0-1.105.1 resolve ``rust_bridge`` imports lazily inside their
logging filter, which is attached to ``uvicorn.access`` at import time. If the
first ``import litellm`` happens on a run thread while another thread logs,
``litellm.rust_bridge.catalog`` is observed partially initialized and the
process's logging is left corrupted.

Importing ``core.llm`` must therefore warm-import LiteLLM on the importing
thread, before ``orchestrator`` starts a run thread. These tests run in a
subprocess against a stub ``litellm`` package: they need no real dependency, no
API key, and make no network calls. The only key involved is a dummy string and
the stub has no ``completion()`` at all.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_RECORD = '''\
import json, os, sys, threading


def record(name):
    core = sys.modules.get("core.llm")
    phase = ("module_import" if core is not None and not hasattr(core, "LLMPolicy")
             else "after_import")
    entry = {"module": name,
             "thread": threading.current_thread().name,
             "main_thread": threading.current_thread() is threading.main_thread(),
             "phase": phase}
    with open(os.environ["FAKE_LITELLM_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\\n")
'''

_STUB = {
    "litellm/_record.py": _RECORD,
    "litellm/__init__.py": 'from litellm._record import record\nrecord("litellm")\n',
    "litellm/utils.py": 'from litellm._record import record\nrecord("litellm.utils")\n',
    "litellm/rust_bridge/__init__.py": "",
    "litellm/rust_bridge/catalog.py":
        'from litellm._record import record\nrecord("litellm.rust_bridge.catalog")\n',
    "litellm/rust_bridge/configuration.py":
        'from litellm._record import record\nrecord("litellm.rust_bridge.configuration")\n',
    "litellm/types/__init__.py": "",
    "litellm/types/secret_managers/__init__.py": "",
    "litellm/types/secret_managers/main.py":
        'from litellm._record import record\nrecord("litellm.types.secret_managers.main")\n',
}

_REQUIRED = {
    "litellm.utils",
    "litellm.rust_bridge.catalog",
    "litellm.rust_bridge.configuration",
    "litellm.types.secret_managers.main",
}

_IMPORT_AND_FIRST_USE = '''\
import json, threading
import core.llm as llm

out = {}


def use():
    out["status"] = llm.LLMPolicy().status


t = threading.Thread(target=use, name="qarena-run")
t.start()
t.join()
print(json.dumps(out))
'''

_NO_LITELLM = '''\
import sys


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("litellm blocked for this test")
        return None


sys.meta_path.insert(0, Blocker())
import core.llm as llm
print(llm.LLMPolicy().status)
'''

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
             "ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")


def _write_stub(tmp_path: Path) -> Path:
    stub = tmp_path / "stub"
    for rel, text in _STUB.items():
        path = stub / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return stub


def _run(script: str, tmp_path: Path, stub: Path | None,
         **env_extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("QARENA_OFFLINE", None)
    for name in _KEY_VARS:
        env.pop(name, None)
    paths = ([str(stub)] if stub else []) + [str(ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                          env=env, capture_output=True, text=True, timeout=120)


def test_importing_core_llm_warms_litellm_on_the_importing_thread(tmp_path):
    stub = _write_stub(tmp_path)
    log = tmp_path / "imports.jsonl"
    proc = _run(_IMPORT_AND_FIRST_USE, tmp_path, stub,
                FAKE_LITELLM_LOG=str(log), GEMINI_API_KEY="test-key-not-real")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"status": "litellm"}
    assert log.exists(), "core.llm never imported litellm at all"

    records = [json.loads(line) for line in log.read_text().splitlines()]
    first = next(r for r in records if r["module"] == "litellm")
    assert first["phase"] == "module_import", (
        "litellm must be imported while core.llm itself is importing, not on "
        f"first use; got {first}")
    assert first["main_thread"] is True
    assert not [r for r in records if not r["main_thread"]], (
        "no litellm module may be first imported from a worker thread")
    assert _REQUIRED <= {r["module"] for r in records}


def test_heuristic_mode_survives_when_litellm_is_missing(tmp_path):
    proc = _run(_NO_LITELLM, tmp_path, None,
                GEMINI_API_KEY="test-key-not-real")
    assert proc.returncode == 0, proc.stderr
    assert "litellm is not installed" in proc.stdout
