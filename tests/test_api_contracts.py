"""HTTP contract tests for the dashboard surface.

The UI reads exactly these endpoints; a change here silently breaks the run page
(the original bug class was a mismatch between persisted artifacts and what the
dashboard displayed). No browser and no live server are involved.
"""
from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core import config
from core.contracts import FiledBug, FlakySuspect, RunResult
from core.orchestrator import RunHandle

TARGET = "https://sweetshop.test"


@pytest.fixture()
def main(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    import app.main as main_mod

    main_mod.orch.runs.clear()
    return main_mod


@pytest.fixture()
def client(main):
    return TestClient(main.app)


def craft_run(main, run_id: str = "apicontract1", status: str = "partial") -> Path:
    rd = config.ARTIFACTS / f"run_{run_id}"
    (rd / "bug_reports").mkdir(parents=True, exist_ok=True)
    bug = FiledBug(
        bug_id="QA-01", title="Basket: Network Failures — GET /cart → 404",
        severity="Minor", journey="Route '/cart' serves a real page", route="/cart",
        element_name="Basket", signals=[2, 1], score=0.95, reproduced_count="3/3",
        steps_to_reproduce=["1. Open `/`", "2. Open `/cart`"],
        expected_behavior="pages load without errors",
        actual_behavior="GET /cart → 404",
        evidence={"screenshot": "steps/TC_007_step2.png",
                  "trace": "trace_TC_007.zip", "url": TARGET + "/cart",
                  "test_id": "TC_007", "confidence": 0.95,
                  "http_failures": [f"GET {TARGET}/cart → 404"],
                  "console_errors": ["Failed to load resource: 404 {/cart}"],
                  "signals": [f"S2 Network Failures: GET {TARGET}/cart → 404"]})
    res = RunResult(
        run_id=run_id, target_url=TARGET, description="shop", status=status,
        states_found=8, tests_planned=2, steps_executed=4, filed_bugs=[bug],
        flaky=[FlakySuspect(journey="TC_009 [axe:color-contrast]",
                            summary="colour contrast", reproduced="1/3", signals=[5])],
        stage_errors=({"exporter": "stage 7 (Exporter): FileNotFoundError: pytest"}
                      if status == "partial" else {}),
        coverage={"route_coverage": 100.0, "routes_tested": 2, "routes_total": 2,
                  "element_coverage": 0.0, "elements_exercised": 0, "elements_total": 2})
    (rd / "results.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
    (rd / "bugs_filed.json").write_text(
        json.dumps([bug.model_dump()], indent=2), encoding="utf-8")
    (rd / "bug_reports" / "QA-01.md").write_text(
        "# QA-01 — Basket: Network Failures — GET /cart → 404\n", encoding="utf-8")
    (rd / "run.log").write_text(
        json.dumps({"type": "stage", "stage": 2, "name": "Sandbox"}) + "\n",
        encoding="utf-8")
    return rd


# ------------------------------------------------------------------ read paths
def test_result_endpoint_reports_partial_with_findings(client, main):
    craft_run(main)
    r = client.get("/api/runs/apicontract1/result")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "partial"
    assert len(body["filed_bugs"]) == 1
    assert len(body["flaky"]) == 1
    assert body["stage_errors"]["exporter"].startswith("stage 7")
    # the dashboard's counters come straight from these fields
    assert body["filed_bugs"][0]["reproduced_count"] == "3/3"


def test_bugs_filed_endpoint_matches_the_artifact(client, main):
    craft_run(main)
    r = client.get("/api/runs/apicontract1/bugs_filed")
    assert r.status_code == 200
    assert [b["bug_id"] for b in r.json()] == ["QA-01"]
    on_disk = json.loads((config.ARTIFACTS / "run_apicontract1" /
                          "bugs_filed.json").read_text(encoding="utf-8"))
    assert r.json() == on_disk, "endpoint and artifact must not disagree"


def test_run_page_renders_for_partial_and_unknown_ids(client, main):
    craft_run(main)
    page = client.get("/runs/apicontract1")
    assert page.status_code == 200
    html = page.text
    assert 'id="partial-card"' in html      # the warnings card exists
    assert "Finished with warnings" in html
    assert "apicontract1" in html
    assert client.get("/runs/doesnotexist").status_code == 404
    assert client.get("/api/runs/doesnotexist/result").status_code == 404


def test_artifacts_are_served_but_traversal_is_refused(client, main):
    craft_run(main)
    ok = client.get("/artifacts/apicontract1/bug_reports/QA-01.md")
    assert ok.status_code == 200
    assert "→" in ok.text                    # evidence survives the HTTP layer
    for evil in ["/artifacts/apicontract1/../../../../etc/passwd",
                 "/artifacts/apicontract1/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                 "/artifacts/apicontract1/....//....//etc/passwd"]:
        r = client.get(evil)
        assert r.status_code in (400, 404), evil
        assert "root:" not in r.text


def test_zip_endpoint_is_cached_for_a_terminal_run(client, main):
    craft_run(main)
    first = client.get("/api/runs/apicontract1/zip")
    assert first.status_code == 200
    archive = config.ARTIFACTS / "run_apicontract1.zip"
    assert archive.exists()
    stamp = archive.stat().st_mtime_ns
    again = client.get("/api/runs/apicontract1/zip")
    assert again.status_code == 200
    assert archive.stat().st_mtime_ns == stamp, "a finished run is not re-zipped"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "bug_reports/QA-01.md" in names
        assert "results.json" in names
        assert not any("__pycache__" in n for n in names)
        assert "→" in zf.read("bug_reports/QA-01.md").decode("utf-8")
    assert client.get("/api/runs/doesnotexist/zip").status_code == 404


# --------------------------------------------------------------- write path
def test_run_intake_validation(client, main):
    for payload, why in [
            ({}, "missing everything"),
            ({"url": "ftp://x.test", "description": "d"}, "non-http scheme"),
            ({"url": "https://x.test", "description": ""}, "empty description"),
            ({"url": "https://user:pw@x.test", "description": "d"}, "embedded credentials"),
            ({"url": "https://x.test", "description": "d" * 4001}, "description too long")]:
        r = client.post("/api/runs", json=payload)
        assert r.status_code == 400, f"{why}: {r.status_code} {r.text[:120]}"


def test_live_run_blocks_a_second_expedition(client, main):
    """Concurrency guard: a genuinely running pipeline still returns 409."""
    release = threading.Event()
    h = RunHandle(run_id="liverun001")
    h.run_dir = config.ARTIFACTS / "run_liverun001"
    h.run_dir.mkdir(parents=True, exist_ok=True)
    h.result = RunResult(run_id="liverun001", target_url=TARGET, description="d",
                         status="running")
    h.thread = threading.Thread(target=release.wait, daemon=True)
    h.thread.start()
    main.orch.runs["liverun001"] = h
    try:
        r = client.post("/api/runs", json={"url": "https://x.test", "description": "d"})
        assert r.status_code == 409
        assert "already running" in r.json()["detail"]
    finally:
        release.set()
        h.thread.join(timeout=5)


def test_orphaned_run_is_not_treated_as_live(client, main):
    """The stuck-409 regression: a run whose process died must not block intake."""
    rd = config.ARTIFACTS / "run_orphan0001"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "results.json").write_text(
        RunResult(run_id="orphan0001", target_url=TARGET, description="d",
                  status="running").model_dump_json(indent=2), encoding="utf-8")
    h = main.load_handle("orphan0001")
    assert h is not None
    assert h.result.status == "error"
    assert "interrupted" in h.result.error
    assert not any(x.thread and x.thread.is_alive() for x in main.orch.runs.values())
    persisted = json.loads((rd / "results.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "error", "the correction is durable, not in-memory only"
