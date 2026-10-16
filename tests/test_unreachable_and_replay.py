"""Regression tests for unreachable-target diagnostics and idempotent UI replay.

Both were exposed by a real run against a host this sandbox's egress rules
block: the hunt stopped correctly but said only "target never became
responsive", and the run page showed the same stage row twice (the SSE snapshot
replays history on every reconnect).
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

import httpx
import pytest

from core import sandbox as sandbox_mod
from core.contracts import RunResult
from core.orchestrator import Orchestrator, RunHandle
from core.sandbox import Sandbox


# ------------------------------------------------------------------ diagnosis
@pytest.fixture()
def fast_wait(monkeypatch):
    """Make the retry loop instant: no real sleeping, 1 attempt."""
    monkeypatch.setattr(sandbox_mod.time, "sleep", lambda _s: None)
    ticks = iter([0.0, 0.0, 10.0, 10.0, 20.0])
    monkeypatch.setattr(sandbox_mod.time, "time", lambda: next(ticks, 99.0))


def _raises(monkeypatch, exc: BaseException):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(sandbox_mod.httpx, "get", boom)


def test_tls_reset_is_reported_as_a_firewall_style_block(fast_wait, monkeypatch):
    _raises(monkeypatch, httpx.ConnectError(
        "TLS/SSL connection has been closed (EOF)"))
    with pytest.raises(RuntimeError) as err:
        Sandbox.wait_until_responsive("https://blocked.example/", timeout=1)
    msg = str(err.value)
    assert "TLS/HTTP handshake" in msg                      # classified
    assert "sandboxed/CI" in msg                            # actionable hint
    assert "ConnectError" in msg and "EOF" in msg           # raw cause kept
    assert "attempt(s) over 1s" in msg


def test_dns_failure_is_distinguished_from_a_blocked_connection(fast_wait, monkeypatch):
    _raises(monkeypatch, socket.gaierror("[Errno -2] Name or service not known"))
    with pytest.raises(RuntimeError) as err:
        Sandbox.wait_until_responsive("https://typo.invalid/", timeout=1)
    assert "does not resolve" in str(err.value)
    assert "TLS/HTTP handshake" not in str(err.value)


def test_connection_refused_says_nothing_is_listening(fast_wait, monkeypatch):
    _raises(monkeypatch, httpx.ConnectError("Connection refused"))
    with pytest.raises(RuntimeError) as err:
        Sandbox.wait_until_responsive("http://127.0.0.1:9/", timeout=1)
    assert "connection refused" in str(err.value).lower()


def test_timeout_is_reported_as_a_timeout(fast_wait, monkeypatch):
    _raises(monkeypatch, httpx.ConnectTimeout("timed out"))
    with pytest.raises(RuntimeError) as err:
        Sandbox.wait_until_responsive("http://10.255.255.1/", timeout=1)
    assert "timed out" in str(err.value)


def test_a_responding_server_still_counts_as_responsive(monkeypatch):
    """An error STATUS means the app is up — QArena must go on and report it."""
    monkeypatch.setattr(sandbox_mod.httpx, "get",
                        lambda *a, **k: type("R", (), {"status_code": 500})())
    assert Sandbox.wait_until_responsive("http://t.test/", timeout=5) is None


# ------------------------------------------------------------- idempotent replay
def _handle(tmp_path: Path) -> tuple[Orchestrator, RunHandle]:
    orch = Orchestrator()
    h = RunHandle(run_id="replaytest")
    h.run_dir = tmp_path
    h.result = RunResult(run_id="replaytest", target_url="http://t.test",
                         description="demo target")
    orch.runs[h.run_id] = h
    return orch, h


def test_every_streamed_event_carries_a_monotonic_seq(tmp_path):
    orch, h = _handle(tmp_path)
    emit = orch.emit_of(h.run_id)
    for i in range(3):
        emit({"type": "stage", "stage": 2, "name": "Sandbox"})
    seqs = [e["seq"] for e in h.history]
    assert seqs == [1, 2, 3]
    # the same objects went to the live queue, so replay and stream agree
    queued = [h.events.get_nowait() for _ in range(3)]
    assert [e["seq"] for e in queued] == seqs
    # and the persisted log carries them too (a page reloaded after a restart
    # still knows which events it already rendered)
    logged = [json.loads(ln) for ln in
              (tmp_path / "run.log").read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in logged] == seqs


def test_terminal_notices_also_get_a_seq_and_stay_out_of_history(tmp_path):
    orch, h = _handle(tmp_path)
    emit = orch.emit_of(h.run_id)
    emit({"type": "stage", "stage": 2, "name": "Sandbox"})
    orch._push(h, {"type": "error", "msg": "boom"})
    orch._push(h, {"type": "done", "status": "error"})
    queue_seqs = []
    while not h.events.empty():
        queue_seqs.append(h.events.get_nowait()["seq"])
    assert queue_seqs == [1, 2, 3]
    assert [e["type"] for e in h.history] == ["stage"]        # not replayed
