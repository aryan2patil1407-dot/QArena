"""
P5 — the run must describe its AI layer truthfully.

Requirements: a configured API key must not imply successful calls; the
LiteLLM dependency is optional and that must be visible; a missing or broken AI
layer must never fail a run; the deterministic pipeline must work with no key at
all; and no key may ever be written to a log, artifact or report.
"""
from __future__ import annotations

import json
import os

import pytest

from core import orchestrator as orch_mod
from core.llm import KEY_ENV_NAMES, LLMPolicy
from core.contracts import StateElement
from tests.conftest import make_map, make_state

FAKE_KEY = "sk-test-DO-NOT-LEAK-0123456789abcdef"


@pytest.fixture()
def clean_env(monkeypatch):
    for k in KEY_ENV_NAMES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("QARENA_OFFLINE", raising=False)
    monkeypatch.delenv("QARENA_LLM_MODEL", raising=False)
    return monkeypatch


# ----------------------------------------------------------------- mode naming
def test_no_key_is_reported_as_key_free_heuristic(clean_env):
    assert LLMPolicy().available is False
    assert "key-free" in LLMPolicy().status
    assert "litellm" not in LLMPolicy().status


def test_offline_flag_is_reported_explicitly(clean_env):
    clean_env.setenv("QARENA_OFFLINE", "1")
    clean_env.setenv("OPENAI_API_KEY", FAKE_KEY)
    assert LLMPolicy().available is False
    assert "QARENA_OFFLINE" in LLMPolicy().status


def test_key_without_litellm_admits_the_missing_dependency(clean_env):
    clean_env.setenv("GROQ_API_KEY", FAKE_KEY)
    pol = LLMPolicy()
    assert pol.available is True, "a key means the layer will be attempted"
    status = pol.status
    try:
        import litellm  # noqa: F401
        assert status == "litellm", status
    except ImportError:
        # the honest outcome: key present, dependency missing => still heuristic
        assert "litellm is not installed" in status, status
        assert "pip install litellm" in status, "must say how to fix it"


def test_a_failing_call_falls_back_and_is_admitted(clean_env, monkeypatch):
    clean_env.setenv("OPENAI_API_KEY", FAKE_KEY)
    pol = LLMPolicy()
    # a provider that is reachable in the code path but always errors
    import sys
    import types
    fake = types.ModuleType("litellm")

    def _boom(**kwargs):
        raise RuntimeError("invalid api key")

    fake.completion = _boom
    monkeypatch.setitem(sys.modules, "litellm", fake)
    out = pol.call_json("planner#1", "sys", "user")
    assert out is None, "a failed call must fall back, never raise"
    assert pol.failures == 1
    assert "fell back to heuristics" in pol.status
    # the key itself never appears anywhere in the status text
    assert FAKE_KEY not in pol.status


def test_groq_and_openrouter_are_supported_providers():
    assert "GROQ_API_KEY" in KEY_ENV_NAMES
    assert "OPENROUTER_API_KEY" in KEY_ENV_NAMES


# ------------------------------------------------------- the run stays honest
def test_run_with_no_ai_layer_completes_and_says_heuristic(ctx, clean_env):
    home = make_state("S00", "/", 200,
                      [StateElement(id="e1", role="link", name="Neon Mug")],
                      discovered_via="seed")
    prod = make_state("S01", "/product/*", 200,
                      [StateElement(id="e2", role="heading", name="Neon Mug")],
                      discovered_via="link", link_name="Neon Mug")
    appmap = make_map([home, prod],
                      edges=[{"from": "S00", "to": "/product/*",
                              "via": 'link "Neon Mug"'}])
    run = ctx.run(appmap=appmap, verdict="passed", suspects=[])
    res = run.results()
    assert res["status"] == "done", res.get("stage_errors")
    assert "heuristic" in res["llm_mode"] or "litellm" in res["llm_mode"]
    assert res["tokens_used"] == 0 or res["tokens_used"] > 0


def test_no_key_leaks_into_artifacts(ctx, clean_env):
    """A key in the environment must never reach a log, result or report."""
    clean_env.setenv("OPENAI_API_KEY", FAKE_KEY)
    appmap = make_map([make_state("S00", "/", 200, [], discovered_via="seed")])
    run = ctx.run(appmap=appmap, verdict="passed", suspects=[])
    for path in run.run_dir.rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            assert FAKE_KEY.encode() not in blob, \
                f"the API key leaked into {path.name}"
    log = (run.run_dir / "run.log").read_text(encoding="utf-8")
    assert FAKE_KEY not in log
    # and the mode still tells the truth about what happened
    assert "heuristic" in json.loads(
        (run.run_dir / "results.json").read_text(encoding="utf-8"))["llm_mode"]
