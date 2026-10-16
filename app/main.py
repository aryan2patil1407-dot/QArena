"""
QArena web application — FastAPI.
Replaces the spec's Streamlit with a hand-crafted field-guide UI:
  /            Stage 1 input form (expedition setup)
  /runs/{id}   live expedition view (SSE) -> findings board
  /api/...     run control, artifacts, benchmark, zip download
"""
from __future__ import annotations

import asyncio
import json
import queue
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core import config
from core.orchestrator import Orchestrator

ROOT = config.ROOT

# Console safety (Windows PowerShell / any non-UTF-8 code page): event payloads
# and exception messages contain evidence characters such as "→". A console that
# cannot encode them would raise *while reporting an error*, hiding the original
# failure. Files are always written as UTF-8; the console is best-effort only.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
app = FastAPI(title="QArena — Autonomous AI QA Engineer", docs_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "frontend" / "static"),
          name="static")
templates = Jinja2Templates(directory=ROOT / "frontend" / "templates")
orch = Orchestrator()


# -------------------------------------------------- restart-resilient runs
def load_handle(run_id: str):
    """Return the live handle, or resurrect the run from its artifact dir so
    pages/events/zip keep working after a UI server restart."""
    h = orch.runs.get(run_id)
    if h:
        return h
    from core.orchestrator import RunHandle
    from core.contracts import RunResult
    run_dir = config.ARTIFACTS / f"run_{run_id}"
    results = run_dir / "results.json"
    if not results.exists():
        return None
    h = RunHandle(run_id=run_id)
    h.run_dir = run_dir
    # artifacts are always written as UTF-8 (evidence carries "→", "—", …)
    h.result = RunResult(**json.loads(results.read_text(encoding="utf-8")))
    if h.result.status == "running":
        # Nothing is executing this run any more — the process that owned it
        # restarted. Stop advertising it as live (this also unblocks the
        # concurrent-run guard, which would otherwise refuse every new
        # expedition until the UI server was restarted).
        h.result.status = "error"
        h.result.error = (h.result.error or
                          "run was interrupted (the UI server restarted before "
                          "the pipeline finished) — artifacts up to the last "
                          "completed step are preserved")
        try:
            (run_dir / "results.json").write_text(
                h.result.model_dump_json(indent=2), encoding="utf-8")
        except Exception:
            pass
    log = run_dir / "run.log"
    if log.exists():
        h.history = [json.loads(ln) for ln in
                     log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    orch.runs[run_id] = h
    return h


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Product homepage — what QArena is and how a hunt works."""
    return templates.TemplateResponse(request, "landing.html")


@app.get("/expedition", response_class=HTMLResponse)
async def expedition_form(request: Request):
    # never prefill target or credentials — blank fields, honest placeholders
    return templates.TemplateResponse(request, "index.html",
                                      {"default_url": "", "default_desc": ""})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_page(request: Request, run_id: str):
    h = load_handle(run_id)
    if not h:
        raise HTTPException(404, "unknown run")
    return templates.TemplateResponse(request, "run.html", {
        "run_id": run_id, "target": h.result.target_url if h.result else "",
        "history": json.dumps(h.history),
    })


# ---------------------------------------------------------------- API
@app.post("/api/runs")
async def api_start(request: Request):
    body = await request.json()
    url = (body.get("url") or "").strip().rstrip("/")
    desc = (body.get("description") or "").strip()
    if not url.startswith("http") or not desc:
        raise HTTPException(400, "target url and description are required")
    # ---- input hygiene ------------------------------------------------------
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise HTTPException(400, "target must be a valid http(s) URL, "
                                 "e.g. https://your-app.com")
    if u.username or u.password:
        raise HTTPException(400, "do not embed credentials in the URL — "
                                 "use the optional login fields instead")
    if len(desc) > 4000:
        raise HTTPException(400, "description is too long (keep it under 4000 "
                                 "characters — business rules, not essays)")
    if len(body.get("email") or "") > 320 or len(body.get("password") or "") > 320:
        raise HTTPException(400, "credential fields are too long")
    # a live *thread* is the only real evidence of a concurrent run: a handle
    # resurrected from disk has no thread and must never block new expeditions
    if any(h.thread and h.thread.is_alive() for h in orch.runs.values()):
        raise HTTPException(409, "an expedition is already running — start the "
                                 "next one when it finishes")
    run_id = orch.start_run(
        url, desc,
        email=(body.get("email") or "").strip(),
        password=(body.get("password") or "").strip(),
        whitebox=body.get("whitebox") or None)
    return {"run_id": run_id, "watch": f"/runs/{run_id}"}


@app.get("/api/runs/{run_id}/events")
async def api_events(run_id: str):
    h = load_handle(run_id)
    if not h:
        raise HTTPException(404, "unknown run")

    async def stream():
        yield f"data: {json.dumps({'type':'snapshot','history':h.history,'result':h.result.model_dump() if h.result else {}})}\n\n"
        while True:
            try:
                ev = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: h.events.get(timeout=20))
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") == "done":
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
            except Exception:
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/result")
async def api_result(run_id: str):
    h = load_handle(run_id)
    if not h or not h.result:
        raise HTTPException(404, "unknown run")
    return JSONResponse(h.result.model_dump())


@app.get("/api/runs/{run_id}/bugs_filed")
async def api_bugs(run_id: str):
    p = config.ARTIFACTS / f"run_{run_id}" / "bugs_filed.json"
    if not p.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/artifacts/{run_id}/{path:path}")
async def api_artifact(run_id: str, path: str):
    base = (config.ARTIFACTS / f"run_{run_id}").resolve()
    p = (base / path).resolve()
    if not str(p).startswith(str(base)) or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/api/runs/{run_id}/zip")
async def api_zip(run_id: str):
    import asyncio
    h = load_handle(run_id)
    if not h or not h.run_dir:
        raise HTTPException(404, "unknown run")
    archive = Path(str(h.run_dir) + ".zip")
    # terminal states: a "partial" run is finished too (a late stage failed,
    # the evidence pack is still complete for what did run)
    done = h.result is not None and h.result.status in ("done", "partial", "error")
    # finished runs: build the evidence pack ONCE, then serve the cached file;
    # running runs: always re-zip (fresh data) — never block the event loop
    if done and archive.exists():
        return FileResponse(archive, filename=f"qarena_run_{run_id}.zip")
    if archive.exists():
        archive.unlink()
    made = await asyncio.to_thread(orch.zip_run, run_id)
    if not made or not archive.exists():
        raise HTTPException(404, "run has no artifacts yet")
    return FileResponse(archive, filename=f"qarena_run_{run_id}.zip")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=config.UI_PORT)
