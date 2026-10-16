"""
Targeted AI Reasoning Layer — architecture section 1.
LiteLLM is used for AT MOST 4 call sites (planner, intent check, repo rank,
report prose). With no API key, QArena falls back to its deterministic
heuristic engines and still runs 100% of the pipeline — demo-survival mode.
"""
from __future__ import annotations

import json
import os
import re

from . import config

# .env support: standard env file at project root is loaded once at import —
# real environment variables always win over .env values.
try:
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
except ImportError:
    pass

KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


class LLMPolicy:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.tokens_prompt = 0
        self.tokens_completion = 0
        self.failures = 0          # call sites that fell back to heuristics
        self.model = os.environ.get("QARENA_LLM_MODEL", config.LLM_MODEL_ENV_DEFAULT)

    @property
    def available(self) -> bool:
        if os.environ.get("QARENA_OFFLINE", "").lower() in ("1", "true"):
            return False
        return any(os.environ.get(k) for k in KEY_ENV_NAMES)

    @property
    def status(self) -> str:
        """Human-readable truth about the AI layer, used for the run's llm_mode.

        A key being present is not the same as calls succeeding — a missing
        ``litellm`` or a provider error silently falls back to heuristics, and
        the run must say so instead of advertising "litellm".
        """
        if os.environ.get("QARENA_OFFLINE", "").lower() in ("1", "true"):
            return "heuristic (key-free — QARENA_OFFLINE=1)"
        if not any(os.environ.get(k) for k in KEY_ENV_NAMES):
            return "heuristic (key-free)"
        try:
            import litellm  # noqa: F401
        except ImportError:
            return (f"heuristic (key present but litellm is not installed — "
                    f"pip install litellm to enable hybrid mode)")
        if self.failures:
            return f"litellm ({self.failures} call site(s) fell back to heuristics)"
        return "litellm"

    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    def call_json(self, site: str, system: str, user: str) -> dict | None:
        """One JSON-mode LLM call. Returns parsed dict or None (→ caller falls back)."""
        self.calls.append(site)
        if not self.available:
            return None
        try:
            import litellm
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2, max_tokens=config.LLM_MAX_TOKENS,
                timeout=config.LLM_TIMEOUT_S,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                self.tokens_prompt += getattr(usage, "prompt_tokens", 0) or 0
                self.tokens_completion += getattr(usage, "completion_tokens", 0) or 0
            text = resp.choices[0].message.content or ""
            return parse_json_loose(text)
        except Exception as exc:  # provider down / quota / bad key → fallback, never crash
            self.failures += 1
            # ASCII-only: this may hit a console on a non-UTF-8 code page
            print(f"[llm] {site} failed ({exc.__class__.__name__}) - using heuristics")
            return None


def parse_json_loose(text: str) -> dict | None:
    """Extract the first {...} block and repair common LLM JSON noise."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    blob = m.group(0)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)  # trailing commas
    try:
        return json.loads(blob)
    except Exception:
        return None
