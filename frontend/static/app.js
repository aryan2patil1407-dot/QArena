/* QArena live expedition view — SSE-driven, no frameworks.
   Design rule: every line must make sense to a first-time user. */
(function () {
  const runId = location.pathname.split("/").pop();
  const $ = (id) => document.getElementById(id);
  const state = {
    suspects: [], flaky: [], rejected: [], result: null,
    suppressed: 0, steps: 0, done: false,
  };
  const t0 = Date.now();
  const clockTimer = setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    $("clock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);

  // ------------------------------------------------ translator: raw → plain English
  const SIG_NAMES = { 1: "console errors", 2: "broken requests", 3: "broken page content",
                      4: "form validation", 5: "accessibility", 6: "business-rule break" };

  function humanSource(str) {
    // suspect groups/summaries → a sentence a user understands
    let s = String(str || "");
    let m;
    if ((m = s.match(/^TC_\d+\s+(.*)$/))) s = m[1];                  // strip journey prefix
    if ((m = s.match(/^S(\d)\s+([^:]+):\s*(.*)$/))) s = m[3];        // strip "S5 Accessibility:"
    s = s.split(" | ")[0].trim();                                    // keep the primary signal only
    if ((m = s.match(/^GET\s+(https?:\S+)\s*→\s*(\d+)/))) return `broken page load — ${m[1]} answers ${m[2]}`;
    if ((m = s.match(/^POST\s+(https?:\S+)\s*→\s*(\d+)/))) return `broken form submit — ${m[1]} answers ${m[2]}`;
    if ((m = s.match(/^http:(\S+)/))) return `broken request to ${m[1]}`;
    if ((m = s.match(/^console:(.*)/))) return `JavaScript error: ${m[1].trim()}`;
    if ((m = s.match(/^axe:(\S+)\s*(.*)/)))
      return `accessibility violation (${m[1].replace("a11y-", "").replace(/-/g, " ")})${m[2] ? " — " + m[2] : ""}`;
    if (s.startsWith("form:money")) return "money amount went negative on the page";
    if (s.startsWith("form:invalid input was silently"))
      return "invalid input was accepted with no error message";
    if ((m = s.match(/^form:forbidden text present:\s*(.+)/))) return `page shows text it never should: ${m[1]}`;
    if ((m = s.match(/^form:reached disallowed URL\s*(.+)/))) return `user reached a page the app should have blocked (${m[1]})`;
    if ((m = s.match(/^intent:(.*)/))) return `business-rule break — ${m[1].trim()}`;
    if ((m = s.match(/^flow:badge (.*)/))) return `cart badge is wrong — ${m[1].trim()}`;
    if ((m = s.match(/^expect:(.*)/))) return m[1].trim();
    return s.replace(/[|]/g, "·").trim().slice(0, 160);
  }
  function journeyOf(str) {
    const m = String(str || "").match(/^(TC_\d+)/);
    return m ? m[1] : "";
  }

  // ---------------------------------------------------------------- helpers
  function log(msg, cls) {
    const li = document.createElement("li");
    if (cls) li.className = cls;
    li.textContent = msg;
    $("log").appendChild(li);
    $("log").scrollTop = $("log").scrollHeight;
  }
  function setStage(n, name) {
    document.querySelectorAll("#stepper li").forEach((li) => {
      const s = +li.dataset.stage;
      li.classList.toggle("done", s < n);
      li.classList.toggle("active", s === n);
    });
    log(`▶ ${name}`, "stage-ev");
  }
  function esc(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // ---------------------------------------------------------------- events
  const seenIds = new Set();
  function deliver(ev) {
    // The server stamps every event with a monotonic seq. A live client already
    // rendered the streamed events, then receives the whole history again as a
    // snapshot on every SSE reconnect — without this guard the same event was
    // shown twice (two identical stage rows, duplicated field-log lines).
    if (typeof ev.seq === "number") {
      if (ev.seq <= (state.lastSeq || 0)) return;
      state.lastSeq = ev.seq;
      handle(ev);
      return;
    }
    // runs recorded before seq existed: an event's identity is its timestamp +
    // shape, and a replay carries the same values, so this stays exact
    const key = [ev.type, ev.ts, ev.stage, ev.name, ev.msg].join("|");
    if (seenIds.has(key)) return;
    seenIds.add(key);
    handle(ev);
  }
  function handle(ev) {
    switch (ev.type) {
      case "stage":
        setStage(ev.stage, ev.name);
        break;
      case "log": {
        let msg = ev.msg;
        const dm = String(msg).match(/^(\d+) unique suspects after dedupe$/);
        if (dm) msg = `${dm[1]} unique possible bugs to double-check (duplicates removed)`;
        else if (msg === "target responsive; seed=yes") msg = "Target answered — it's reachable. Crawling is safe to begin";
        else if (msg === "login-once: ok") msg = "Signed in once with the test credentials";
        log("· " + msg); break;
      }
      case "state":
        bump("st-states");
        log(`Found page ${ev.route} [${ev.status}] — ${ev.elements} interactive controls`);
        break;
      case "map":
        log(`Site map complete: ${ev.states} pages discovered in ${ev.seconds}s`, "ok");
        $("map-card").hidden = false;
        $("map-frame").src = `/artifacts/${runId}/map.html`;
        break;
      case "plan":
        $("st-tests").textContent = ev.tests;
        log(`Test plan ready: ${ev.tests} journeys. (${ev.rejected} ungrounded ideas auto-rejected at the guardrail)`, "ok");
        break;
      case "suspect": {
        state.suspects.push(ev);
        $("st-suspects").textContent = state.suspects.length;
        $("suspect-tray").hidden = false;
        const what = humanSource(ev.summary);
        const where = journeyOf(ev.journey) ? `spotted during journey ${journeyOf(ev.journey)}` : "spotted while exploring";
        renderTray("suspect-list", state.suspects,
          (s) => `<b>${esc(humanSource(s.summary))}</b><br>
                  <span class="mini">${esc(journeyOf(s.journey) ? "during journey " + journeyOf(s.journey) : "while exploring")} · confidence ${s.score} · being double-checked…</span>`);
        log(`⚑ Possible bug: ${what} (${where}, confidence ${ev.score}) — will prove it before reporting`, "bug");
        break;
      }
      case "confirmed":
        state.confirmed = (state.confirmed || 0) + 1;
        $("st-confirmed").textContent = state.confirmed;
        log(`✓ PROVEN bug: ${humanSource(ev.journey)} — reproduced ${ev.count} in fresh browsers`, "bug");
        break;
      case "flaky":
        state.flaky.push(ev);
        $("st-flaky").textContent = state.flaky.length;
        $("flaky-tray").hidden = false;
        renderTray("flaky-list", state.flaky, flakyLine);
        log(flakyLog(ev), "");
        break;
      case "suppressed":
        state.suppressed = (state.suppressed || 0) + 1;
        $("suppressed-tray").hidden = false;
        renderTray("suppressed-list", state.suppressed > 1 ? [] : [ev],
          (s) => `<b>${esc(humanSource(s.journey))}</b> — withheld: ${esc(s.reason)}`);
        if (state.suppressed === 1) {
          $("suppressed-list").innerHTML =
            `<li><b>${esc(humanSource(ev.journey))}</b> — withheld: ${esc(ev.reason)}</li>`;
        } else {
          $("suppressed-list").insertAdjacentHTML("beforeend",
            `<li><b>${esc(humanSource(ev.journey))}</b> — withheld: ${esc(ev.reason)}</li>`);
        }
        break;
      case "step":
        state.steps += 1;
        $("st-steps").textContent = state.steps;
        break;
      case "replay":
        log(`  double-check ${ev.attempt}/3 on “${humanSource(ev.journey)}” — ${ev.matched ? "bug happened again ✓" : "did not reappear"}`);
        break;
      case "verdict":
        log(ev.verdict === "passed"
          ? `Journey ${ev.test}: behaved exactly as the description says ✓`
          : `Journey ${ev.test}: broke a promised rule — looking for solid proof…`,
          ev.verdict === "passed" ? "ok" : "");
        break;
      case "warning":
        log(`⚠ ${ev.msg}`, "warn-ev");
        break;
      case "error":
        log(`STOPPED: ${ev.msg}`, "bug");
        break;
      case "done":
        onDone(ev.result);
        es.close();
        break;
    }
  }
  function bump(id) { $(id).textContent = (+$(id).textContent || 0) + 1; }
  function renderTray(listId, items, tpl) {
    $(listId).innerHTML = items.map((i) => `<li>${tpl(i)}</li>`).join("");
  }

  // A dropped suspect is only "flaky" when the replay actually ran. An
  // unreplayable suspect is a statement about the test environment, not about
  // the app, and is labelled as such instead of being quietly blamed on
  // "flakiness".
  function flakyLine(s) {
    const kind = s.kind || "flaky";
    const label = `<b>${esc(humanSource(s.summary))}</b>`;
    if (kind === "infrastructure_error")
      return `${label} — <em>could not be re-tested</em>: ${esc(s.detail || "the replay never ran")}. This says nothing about your app.`;
    if (kind === "inconclusive")
      return `${label} — <em>inconclusive</em>: ${esc(s.detail || "a replay step could not complete")}. Not evidence of a defect either way.`;
    if (kind === "not_reproduced")
      return `${label} — <em>did not recur</em>: ${esc(s.detail || "clean replays did not reproduce it")}.`;
    return `${label} — <em>flaky</em>: ${esc(s.detail || `reappeared in only ${s.reproduced || s.count}`)}. Below the 2-of-3 bar, so not reported.`;
  }

  function flakyLog(ev) {
    const kind = ev.kind || "flaky";
    if (kind === "infrastructure_error")
      return `⚠ Not re-tested: ${humanSource(ev.journey)} — the replay could not run (${ev.detail || "infrastructure"}). No conclusion about the app.`;
    if (kind === "inconclusive")
      return `⚠ Inconclusive: ${humanSource(ev.journey)} — a replay step could not complete, so nothing is claimed.`;
    if (kind === "not_reproduced")
      return `— Dropped: ${humanSource(ev.journey)} — the clean replays did not reproduce it.`;
    return `✖ Dropped: ${humanSource(ev.journey)} — reappeared only ${ev.count} of 3 replays, not solid enough to report`;
  }

  // ---------------------------------------------------------------- done
  async function onDone(result) {
    if (state.done) return;
    state.done = true;
    clearInterval(clockTimer);
    document.querySelectorAll("#stepper li").forEach((li) => {
      li.classList.remove("active");
      li.classList.add("done");
    });
    const pill = $("status-pill");
    const status = result && result.status;
    // "partial" is a real outcome: the hunt finished and proved findings, but a
    // later stage failed. Say that, rather than "COMPLETE" or "STOPPED EARLY".
    pill.textContent = status === "error" ? "STOPPED EARLY"
                     : status === "partial" ? "DONE — WITH WARNINGS"
                     : "COMPLETE";
    pill.className = "pill mono " + (status === "error" ? "error"
                                   : status === "partial" ? "partial" : "done");
    $("zip-btn").hidden = false;
    $("findings").hidden = false;
    const res = result || (await fetch(`/api/runs/${runId}/result`).then((r) => r.json()));
    // pin KPI tiles to results.json — history replay is a retelling, not accounting
    $("st-confirmed").textContent = (res.filed_bugs || []).length;
    $("st-flaky").textContent = (res.flaky || []).length;
    if (res.steps_executed != null) $("st-steps").textContent = res.steps_executed;
    if (res.states_found != null) $("st-states").textContent = res.states_found;
    if (res.tests_planned != null) $("st-tests").textContent = res.tests_planned;
    renderFindings(res);
  }

  function renderFindings(res) {
    $("tokens").textContent = res.tokens_used || 0;
    log(`expedition ${res.status === "error" ? "stopped early" : "complete"} — mode: ${res.llm_mode}`);

    if (res.status === "error" && res.error) {
      $("error-card").hidden = false;
      $("error-msg").textContent = res.error;
    }
    const stageErrors = res.stage_errors || {};
    const failedStages = Object.keys(stageErrors);
    if (failedStages.length) {
      $("partial-card").hidden = false;
      $("partial-list").innerHTML = failedStages
        .map((k) => `<li><b>${esc(k)}</b> stage failed — ${esc(stageErrors[k])}</li>`)
        .join("");
    }
    if (res.benchmark && res.benchmark.available) {
      const b = res.benchmark;
      $("bench-card").hidden = false;
      $("bench-grid").innerHTML = [
        [b.precision, "precision — of the bugs we filed, how many were real"],
        [b.recall, "recall — of the planted bugs, how many we found"],
        [b.f1, "F1 — combined score"],
        [`${b.reproducibility_pct}%`, "of reports reproducible"],
        [`${b.true_positives}/${b.planted_bugs}`, "planted bugs found"],
      ].map(([v, l]) => `<div class="bstat" title="${esc(l)}"><b>${v}</b><span>${esc(l.split("—")[0].trim())}</span></div>`).join("");
      $("bench-missed").textContent = b.missed && b.missed.length
        ? `honest admission — planted bugs we did not find: ${b.missed.join(", ")}`
        : "no planted bug escaped — every one found and proven 🎉";
    }
    // withheld findings survive a restart: replayed runs show the same tray
    if ((res.suppressed || []).length) {
      $("suppressed-tray").hidden = false;
      $("suppressed-list").innerHTML = res.suppressed
        .map((s) => `<li><b>${esc(humanSource(s.journey))}</b> — withheld: ${esc(s.reason)}</li>`)
        .join("");
    }
    if (res.coverage && (res.coverage.routes_total || res.coverage.journeys_planned)) {
      const c = res.coverage;
      const pct = (v) => (v === null || v === undefined) ? "n/a" : `${v}%`;
      $("cov-row").textContent =
        `Coverage — measured from browser execution, never from the plan: ` +
        `${pct(c.route_coverage)} of testable pages visited (${c.routes_executed ?? c.routes_tested}/${c.routes_total}) · ` +
        `${pct(c.element_coverage)} of controls exercised (${c.elements_exercised}/${c.elements_total}) · ` +
        `${c.journeys_planned || 0} journeys planned, ${c.journeys_executed || 0} executed` +
        (c.journeys_unexecuted ? `, ${c.journeys_unexecuted} never ran` : "") +
        (c.probe_only_routes && c.probe_only_routes.length
          ? ` · ${c.probe_only_routes.length} guessed URL(s) excluded from testing (${c.probe_only_routes.join(", ")})` : "") +
        ` · ${(res.exported_tests || []).length} reusable regression tests exported`;
      $("cov-basis").textContent = c.measured_from || "";
      $("cov-basis").hidden = !c.measured_from;
    }
    const bugs = res.filed_bugs || [];
    if (bugs.length) {
      $("specimens").insertAdjacentHTML("beforebegin",
        `<div class="sev-legend mono">reading the stamps:
         <b class="sev Critical">CRITICAL</b> breaks a core flow ·
         <b class="sev Major">MAJOR</b> blocks or misleads users ·
         <b class="sev Minor">MINOR</b> fix when convenient</div>`);
    }
    $("specimens").innerHTML = bugs.length
      ? bugs.map(specCard).join("")
      : `<div class="card clean-card"><h3>No proven bugs 🎉</h3>
         <p class="explain">QArena hunted and found nothing it could prove. That is a
         <strong>result</strong>, not a failure: anything uncertain sits in the
         “dropped” tray instead of wasting your time. If you expected bugs, add more
         detail to the description and run again — vague descriptions blunt the
         business-rule judge.</p></div>`;
    if (res.exported_tests && res.exported_tests.length) {
      $("tests-card").hidden = false;
      $("tests-list").innerHTML = res.exported_tests
        .map((f) => `<li><span>${esc(f)}</span>
           <a class="chip" href="/artifacts/${runId}/exported_tests/${esc(f)}">view</a></li>`)
        .join("") + `<li><span>proof they pass — pytest self-run log</span>
           <a class="chip" href="/artifacts/${runId}/pytest_self_run.log">view</a></li>`;
    }
    if (res.rejected_steps && res.rejected_steps.length) {
      $("rejected-tray").hidden = false;
      $("rejected-list").innerHTML = res.rejected_steps
        .map((r) => `<li><b>${esc(r.test_id)}</b> ${esc(r.title)} — refused: ${(r.invented_steps || []).map(esc).join(", ")}</li>`)
        .join("");
    }
  }

  // turn machine titles into one sentence a non-engineer understands
  function headlineFor(b) {
    const t = `${b.title} ${b.actual_behavior}`.toLowerCase();
    const page = b.route && b.route !== "/" ? `the ${b.route} page` : "the page";
    if (/404/.test(t) && (b.signals || []).includes(2))
      return `Dead route — ${page} doesn't exist (404 error)`;
    if (/500/.test(t) && (b.signals || []).includes(2))
      return `Server crash — this action returns HTTP 500`;
    if (/price mismatch/.test(t)) return `Wrong price shown — doesn't match the catalog`;
    if (/badge expected/.test(t)) return `Cart counter shows the wrong number`;
    if (/negative money|money went negative|-\$/.test(t)) return `Money allowed below zero — totals can go negative`;
    if (/silently accepted|no validation message/.test(t)) return `Bad input accepted silently — no error message shown`;
    if (/disallowed url/.test(t)) return `Users can reach a page the app must block`;
    if (/#\$0\.00 order confirmed|\$0\.00 order/.test(t)) return `Checkout pays an empty cart — $0.00 “Order confirmed”`;
    if (/order confirmed/.test(t) && b.signals.includes(4)) return `Order goes through despite missing/invalid details`;
    if (/color-contrast|colour-contrast/.test(t)) return `Text is hard to read — fails color-contrast accessibility rule`;
    if (/placeholder-only-label/.test(t)) return `Form field has no real label — screen-reader users get lost`;
    if (/a11y|axe|accessib/.test(t)) return `Accessibility violation found by axe`;
    if (/forbidden text present/.test(t)) return `Page shows an error message instead of working`;
    if (/is not defined|uncaught|console/.test(t)) return `JavaScript error breaks part of the page`;
    if (/forbidden|expect:/.test(t)) return `Page shows the wrong content for this action`;
    return b.title;
  }

  function specCard(b) {
    const chips = (b.signals || [])
      .map((s) => `<span class="chip">${esc(SIG_NAMES[s] || `signal ${s}`)}</span>`).join("");
    const steps = (b.steps_to_reproduce || []).slice(0, 6)
      .map((s) => `<li>${esc(s.replace(/^\d+\.\s*/, ""))}</li>`).join("");
    const shot = b.evidence.screenshot
      ? `<img src="/artifacts/${runId}/${esc(b.evidence.screenshot)}" alt="screenshot at the moment the bug happened">` : "";
    const where = b.route ? `page ${b.route}` : "on this site";
    // evidence strings carry signal tags — present them in plain words
    const untag = (s) => (s || "").replace(/S\d+ [A-Za-z ]+:/g, "").replace(/\s*\|\s*/g, " — ").trim();
    const wrong = untag(b.actual_behavior).slice(0, 240);
    const wanted = (b.expected_behavior || "").trim().slice(0, 240);
    return `<article class="spec">
      <span class="sev ${esc(b.severity)}">${esc(b.severity)}</span>
      <span class="sid">${esc(b.bug_id)}</span>
      <h3>${esc(headlineFor(b))}</h3>
      <div class="tech-title mono">${esc(b.title)}</div>
      <div class="loc">📍 Found on <b>${esc(where)}</b>, while testing “${esc(b.journey)}”</div>
      ${wrong ? `<div class="qa"><b>What goes wrong:</b> ${esc(wrong)}</div>` : ""}
      ${wanted ? `<div class="qa"><b>What should happen:</b> ${esc(wanted)}</div>` : ""}
      <div class="chipset">${chips}</div>
      <p class="repro">✔ proof: reproduced ${esc(b.reproduced_count)} in fresh browsers</p>
      ${steps ? `<p class="mini"><b>Try it yourself:</b></p><ol>${steps}</ol>` : ""}
      ${shot}
      <div class="actions">
        <a href="/artifacts/${runId}/bug_reports/${esc(b.bug_id)}.md">full report (.md)</a>
        ${b.evidence.screenshot ? `<a href="/artifacts/${runId}/${esc(b.evidence.screenshot)}">screenshot</a>` : ""}
        <a href="/artifacts/${runId}/${esc(b.evidence.trace)}">browser trace (.zip)</a>
      </div>
    </article>`;
  }

  // ------------------------------------------------------------- bootstrap
  const es = new EventSource(`/api/runs/${runId}/events`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.type === "snapshot") {
      if (state.done) { es.close(); return; }   // EventSource auto-reconnects
      (ev.history || []).forEach(deliver);      // — replay must stay idempotent
      if (ev.result && ev.result.status === "done") {
        onDone(ev.result);
        es.close();                             // a finished run never streams again
      }
      if (ev.result && ev.result.status === "error") {
        onDone(ev.result);
        es.close();
      }
      return;
    }
    if (ev.type === "ping") return;
    deliver(ev);
  };
  es.onerror = () => { if (!state.done) log("…reconnecting to the field unit"); };
})();
