"""Regression tests for the optional external-browser override (core.sandbox).

The override only changes behaviour when QARENA_CHROMIUM_PATH is set; nothing
here launches a real browser (that would need a downloaded Chromium), it only
checks the launch kwargs the pipeline builds.
"""
from __future__ import annotations

import os

from core.sandbox import Sandbox


def _sandbox() -> Sandbox:
    return Sandbox("http://127.0.0.1:1", os.environ.get("TMPDIR") and __import__("pathlib").Path("/tmp"))


def test_override_is_off_by_default(monkeypatch):
    monkeypatch.delenv("QARENA_CHROMIUM_PATH", raising=False)
    assert Sandbox.browser_override() == {}


def test_override_blank_string_is_ignored(monkeypatch):
    monkeypatch.setenv("QARENA_CHROMIUM_PATH", "   ")
    assert Sandbox.browser_override() == {}


def test_override_sets_executable_path(monkeypatch):
    monkeypatch.setenv("QARENA_CHROMIUM_PATH", "/opt/chrome/chromium")
    assert Sandbox.browser_override() == {"executable_path": "/opt/chrome/chromium"}


def test_start_browser_uses_override_and_skips_channel(monkeypatch):
    """With the override set the launch must not pass a channel name."""
    monkeypatch.setenv("QARENA_CHROMIUM_PATH", "/opt/chrome/chromium")
    calls: dict = {}

    class _FakeChromium:
        @staticmethod
        def launch(**kwargs):
            calls.update(kwargs)
            return "fake-browser"

    class _FakePW:
        chromium = _FakeChromium()

    monkeypatch.setattr("core.sandbox.sync_playwright",
                        lambda: type("P", (), {"start": staticmethod(lambda: _FakePW())})())

    sb = _sandbox()
    assert sb.start_browser() == "fake-browser"
    assert calls["executable_path"] == "/opt/chrome/chromium"
    assert "channel" not in calls
    assert calls["headless"] is True
