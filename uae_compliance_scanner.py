#!/usr/bin/env python3
"""
UAE Compliance Scanner — Production-Ready
Single-file Flask app — scans GitHub repos for UAE regulatory compliance violations.
Uses OpenRouter (https://openrouter.ai/api/v1) to access Perplexity, Grok, and Claude.

Demo mode  : openrouter/auto:free — $0.00 billed, OpenRouter auto-selects best free model
Full mode  : Perplexity + Grok + Claude (paid BYOK)
Fallback   : paid chain → claude-sonnet → openrouter/auto:free as last resort

Requirements:
    pip install flask requests gitpython
"""

import concurrent.futures
import json
import os
import re
import shutil
import tempfile
import time
import threading
import traceback
from html import escape
from pathlib import Path

from flask import Flask, request, Response, stream_with_context
import requests as http_requests

try:
    import git as gitpython
except ImportError:
    gitpython = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

# ─── Constants ─────────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MAX_FILES       = 50
MAX_DEMO_FILES  = 5
MAX_FILE_CHARS  = 18_000
BATCH_SIZE      = 3
DEMO_BATCH_SIZE = 2

TARGET_EXTENSIONS = {".py", ".js", ".ts", ".sol"}
SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "env", "dist",
    "build", ".git", ".tox", ".mypy_cache", "coverage", ".next", ".nuxt",
}

DEMO_REPO_URL = os.environ.get(
    "DEMO_REPO_URL",
    "https://github.com/firmai/financial-machine-learning",
)

# ─── Model Definitions ─────────────────────────────────────────────────────────

# Full BYOK paid scan
MODELS = {
    "regulations": "perplexity/sonar-deep-research",
    "enforcement": "x-ai/grok-4.20-multi-agent-beta",
    "audit":       "anthropic/claude-sonnet-4-6",
}

MODEL_PRICING = {
    "perplexity/sonar-deep-research":  {"input": 3.00, "output": 15.00},
    "x-ai/grok-4.20-multi-agent-beta": {"input": 3.00, "output": 15.00},
    "anthropic/claude-sonnet-4-6":     {"input": 3.00, "output": 15.00},
}

# Demo free-tier scan — openrouter/auto:free for all roles, $0 billed to user.
# OpenRouter auto-selects the best available free model per request.
DEMO_MODELS = {
    "regulations": "openrouter/auto:free",
    "enforcement": "openrouter/auto:free",
    "audit":       "openrouter/auto:free",
}

DEMO_MODEL_PRICING = {
    "openrouter/auto:free": {"input": 0.0, "output": 0.0},
}

# Fallback chains:
#   Demo  — openrouter/auto:free handles everything; no further fallback needed
#           (OpenRouter itself load-balances across all available free models)
#   Full  — paid chain walks to claude-sonnet, then drops to free auto-router
#           as an absolute last resort so the scan never returns nothing
FALLBACK_CHAINS: dict[str, list[str]] = {
    # Demo
    "openrouter/auto:free": [],

    # Paid fallbacks
    "perplexity/sonar-deep-research":  ["anthropic/claude-sonnet-4-6", "openrouter/auto:free"],
    "x-ai/grok-4.20-multi-agent-beta": ["anthropic/claude-sonnet-4-6", "openrouter/auto:free"],
    "anthropic/claude-sonnet-4-6":     ["openrouter/auto:free"],
}

UAE_FRAMEWORKS = [
    ("1",  "PDPL Federal Decree-Law 45/2021",    "Personal Data Protection — consent, data minimisation, cross-border transfers, subject rights"),
    ("2",  "AML Law 20/2018",                     "Anti-Money Laundering — KYC, transaction monitoring, suspicious activity reporting, record keeping"),
    ("3",  "CBUAE Decree-Law 6/2025",             "Central Bank — payment services licensing, stored value, open banking, consumer protection"),
    ("4",  "VARA Virtual Asset Regulations 2023", "Virtual assets — exchange, custody, issuance, wallet services, travel rule"),
    ("5",  "DIFC DPL 5/2020",                     "DIFC Data Protection Law — lawful basis, data transfers, controller obligations"),
    ("6",  "NESA IA-7",                           "National Electronic Security Authority — cybersecurity controls, encryption, access management, logging"),
    ("7",  "UAE Corporate Tax Law 47/2022",        "Corporate tax compliance in financial logic — revenue recognition, transfer pricing, reporting"),
    ("8",  "Cabinet Resolution 58/2020",          "AML/CFT — beneficial ownership registers, enhanced due diligence, cross-border wire rules"),
    ("9",  "SCA Board Decision 23/2020",          "Securities and Commodities Authority — fintech, robo-advisory, crowdfunding, investment platforms"),
    ("10", "UAE Consumer Protection Law 15/2020", "Data handling, pricing disclosure, complaint mechanisms, unfair terms in consumer-facing code"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Thread-safe global rate limiter.
    Enforces a minimum interval between consecutive API calls.
    Default: 1.1 s gap (slight buffer above OpenRouter's 1-req/s free-tier limit).
    """

    def __init__(self, min_interval: float = 1.1):
        self._lock         = threading.Lock()
        self._last_call_at = 0.0
        self._min_interval = min_interval

    def wait(self) -> None:
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_call_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_at = time.monotonic()


_rate_limiter = RateLimiter(min_interval=1.1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — OpenRouter Client (retry + fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class OpenRouterError(Exception):
    """Base exception for all OpenRouter API failures."""

class RateLimitError(OpenRouterError):
    """Raised on HTTP 429 Too Many Requests."""


def _single_chat_call(
    api_key:  str,
    model:    str,
    messages: list[dict],
    timeout:  int = 180,
) -> tuple[str, int, int]:
    """
    Execute one (non-retried) OpenRouter chat completion call.
    Applies the global rate limiter before the request.

    Returns: (content, input_tokens, output_tokens)
    Raises:  RateLimitError on 429, OpenRouterError on all other failures.
    """
    _rate_limiter.wait()

    headers = {
        "Authorization":      f"Bearer {api_key}",
        "Content-Type":       "application/json",
        "HTTP-Referer":       "https://uae-compliance-scanner.local",
        "X-OpenRouter-Title": "UAE Compliance Scanner",
    }
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": 0.1,
        "max_tokens":  4096,
    }

    try:
        resp = http_requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=headers,
            data=json.dumps(body),
            timeout=(10, timeout),   # (connect_timeout, read_timeout)
        )
    except http_requests.exceptions.Timeout as exc:
        raise OpenRouterError(f"Request timed out after {timeout} s: {exc}") from exc
    except http_requests.exceptions.ConnectionError as exc:
        raise OpenRouterError(f"Connection error: {exc}") from exc

    if resp.status_code == 429:
        raise RateLimitError(f"HTTP 429 — {resp.text[:300]}")

    try:
        resp.raise_for_status()
    except http_requests.exceptions.HTTPError as exc:
        raise OpenRouterError(f"HTTP {resp.status_code}: {exc}") from exc

    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", str(data["error"]))
        raise OpenRouterError(f"API error: {msg}")

    usage   = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def openrouter_chat(
    api_key:       str,
    model:         str,
    messages:      list[dict],
    pricing_table: dict | None = None,
    max_retries:   int = 5,
    timeout:       int = 180,
) -> tuple[str, int, int, str]:
    """
    Retrying OpenRouter call with exponential backoff and automatic model fallback.

    Retry schedule per model: 1 s → 2 s → 4 s → 8 s → 16 s
    If all retries for the primary model fail, the next model in
    FALLBACK_CHAINS is tried from scratch (same retry schedule).

    Returns: (content, input_tokens, output_tokens, model_actually_used)
    Raises:  OpenRouterError when every model in the chain is exhausted.
    """
    if pricing_table is None:
        pricing_table = MODEL_PRICING

    candidates = [model] + FALLBACK_CHAINS.get(model, [])

    last_exc: Exception = OpenRouterError("No models attempted.")

    for candidate in candidates:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                content, inp, out = _single_chat_call(
                    api_key, candidate, messages, timeout=timeout
                )
                return content, inp, out, candidate

            except RateLimitError as exc:
                last_exc = exc
                if attempt == max_retries:
                    break
                sleep_for = delay + (delay * 0.15)
                time.sleep(sleep_for)
                delay = min(delay * 2, 30.0)

            except OpenRouterError as exc:
                last_exc = exc
                if attempt == max_retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    raise OpenRouterError(
        f"All retries and fallbacks exhausted. "
        f"Tried: {candidates}. Last error: {last_exc}"
    ) from last_exc


# ═══════════════════════════════════════════════════════════════════════════════
def _chat_with_hard_timeout(
    api_key:       str,
    model:         str,
    messages:      list[dict],
    pricing_table: dict | None,
    max_retries:   int,
    timeout:       int,
    hard_timeout:  int,
) -> tuple[str, int, int, str]:
    """
    Run openrouter_chat() in a background thread with a hard wall-clock deadline.

    Catches the case where a free model accepts the TCP connection but stalls
    mid-response so the requests read-timeout never fires. The thread is
    submitted to a ThreadPoolExecutor; if it doesn't finish within hard_timeout
    seconds the future is abandoned and OpenRouterError is raised immediately,
    letting the generator continue streaming.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            openrouter_chat,
            api_key, model, messages,
            pricing_table, max_retries, timeout,
        )
        try:
            return future.result(timeout=hard_timeout)
        except concurrent.futures.TimeoutError:
            raise OpenRouterError(
                f"Hard timeout: no response within {hard_timeout}s wall-clock. "
                "Skipping batch."
            )


# SECTION 3 — Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def extract_json(text: str):
    """
    Robustly extract a JSON array or object from an LLM response.
    Handles raw JSON, markdown code fences, and embedded JSON.
    Returns the parsed Python object, or None if nothing parseable found.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    for pat in [r"```json\s*\n([\s\S]*?)\n\s*```", r"```\s*\n([\s\S]*?)\n\s*```"]:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                continue

    for pat in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                continue

    return None


def calc_cost(model: str, inp: int, out: int, pricing_table: dict | None = None) -> float:
    """Calculate USD cost for a single API call based on token counts."""
    if pricing_table is None:
        pricing_table = MODEL_PRICING
    p = pricing_table.get(model, {"input": 3.0, "output": 15.0})
    return (inp * p["input"] + out * p["output"]) / 1_000_000


def find_source_files(repo_dir: str, limit: int = MAX_FILES) -> list[tuple[str, str]]:
    """Walk repo_dir and return up to `limit` (full_path, relative_path) tuples."""
    results: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if Path(fname).suffix.lower() in TARGET_EXTENSIONS:
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, repo_dir)
                results.append((full, rel))
                if len(results) >= limit:
                    return results
    return results


def read_file_safe(path: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """Read a source file safely, truncating to max_chars."""
    try:
        content = open(path, "r", errors="ignore").read()
    except OSError as exc:
        return f"# Could not read file: {exc}"
    if len(content) > max_chars:
        return content[:max_chars] + f"\n# ... [truncated at {max_chars} chars]"
    return content


def make_batches(
    files: list[tuple[str, str]],
    batch_size: int,
) -> list[list[tuple[str, str]]]:
    """Split a flat file list into sub-lists of at most batch_size entries."""
    return [files[i : i + batch_size] for i in range(0, len(files), batch_size)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Scan Health Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class ScanStats:
    """Accumulates audit health data across the full scan run."""

    def __init__(self) -> None:
        self.successful_batches:   int = 0
        self.failed_batches:       int = 0
        self.rate_limit_hits:      int = 0
        self.fallback_activations: int = 0
        self.models_used: dict[str, int] = {}

    @property
    def is_incomplete(self) -> bool:
        return self.failed_batches > 0 or self.rate_limit_hits > 0

    @property
    def total_batches(self) -> int:
        return self.successful_batches + self.failed_batches

    def record_success(self, model_id: str) -> None:
        self.successful_batches += 1
        self.models_used[model_id] = self.models_used.get(model_id, 0) + 1

    def record_failure(self, *, rate_limited: bool = False) -> None:
        self.failed_batches += 1
        if rate_limited:
            self.rate_limit_hits += 1

    def record_fallback(self) -> None:
        self.fallback_activations += 1

    def summary(self) -> str:
        parts = [f"✅ {self.successful_batches}/{self.total_batches} batch(es) succeeded"]
        if self.failed_batches:
            parts.append(f"❌ {self.failed_batches} failed")
        if self.rate_limit_hits:
            parts.append(f"⚠ {self.rate_limit_hits} rate-limit hit(s)")
        if self.fallback_activations:
            parts.append(f"🔀 {self.fallback_activations} fallback(s) activated")
        return " · ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Audit Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def build_audit_system_prompt(fw_list: str, additional_regs: str) -> str:
    return f"""You are a senior UAE financial compliance auditor and code security expert.

Audit the provided source code files against ALL of the following 10 UAE regulatory frameworks:
{fw_list}

ALSO audit against these ADDITIONAL live regulatory updates and trending enforcement patterns:
{additional_regs}

For every violation found, return a JSON object with EXACTLY these keys:
- "file": the filename (string)
- "function_name": the exact function or method name containing the violation (string)
- "line_start": integer line number where the violation begins
- "line_end": integer line number where the violation ends
- "regulation": the specific law name and article number violated
- "violation_description": clear explanation of what the code does wrong and why it violates the regulation
- "severity": one of "critical", "high", or "medium"

Return a JSON ARRAY of all violations across ALL provided files.
If no violations are found, return an empty array: []
Do NOT include any text outside the JSON array.
Do NOT wrap the array in markdown fences.
"""


def audit_file_batch(
    api_key:       str,
    model:         str,
    batch:         list[tuple[str, str]],
    system_prompt: str,
    pricing_table: dict,
    stats:         ScanStats,
    timeout:       int = 180,
) -> tuple[list[dict], int, int, str]:
    """
    Audit a batch of files in a single API call.
    Returns: (violations, input_tokens, output_tokens, model_used)
    On failure returns: ([], 0, 0, model) and updates stats accordingly.
    """
    file_blocks: list[str] = []
    for full_path, rel_path in batch:
        source = read_file_safe(full_path)
        if source.strip():
            file_blocks.append(f"=== FILE: {rel_path} ===\n```\n{source}\n```")

    if not file_blocks:
        return [], 0, 0, model

    combined_prompt = (
        "\n\n".join(file_blocks)
        + "\n\nReturn a single JSON array of ALL violations across ALL files above."
    )

    try:
        hard_limit = timeout + 15   # wall-clock hard kill: read-timeout + 15 s buffer
        content, inp, out, model_used = _chat_with_hard_timeout(
            api_key,
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": combined_prompt},
            ],
            pricing_table=pricing_table,
            max_retries=5,
            timeout=timeout,
            hard_timeout=hard_limit,
        )

        if model_used != model:
            stats.record_fallback()

        violations: list[dict] = []
        parsed = extract_json(content)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    violations.append(item)
        elif isinstance(parsed, dict) and parsed:
            violations.append(parsed)

        stats.record_success(model_used)
        return violations, inp, out, model_used

    except RateLimitError:
        stats.record_failure(rate_limited=True)
        return [], 0, 0, model

    except OpenRouterError:
        stats.record_failure(rate_limited=False)
        return [], 0, 0, model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CSS
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0a0e17;color:#c9d1d9;line-height:1.6}
.container{max-width:980px;margin:0 auto;padding:24px 16px}
.site-header{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.site-header h1{color:#58a6ff;font-size:1.9rem;letter-spacing:-.5px}
.tagline{color:#8b949e;font-size:.92rem;margin-bottom:28px}
.form-card{background:#161b22;border:1px solid #30363d;border-radius:10px;
           padding:28px;margin-bottom:20px}
.form-card h2{color:#c9d1d9;font-size:1.05rem;margin-bottom:20px;
              border-bottom:1px solid #21262d;padding-bottom:10px}
label{display:block;color:#c9d1d9;font-weight:600;margin-bottom:5px;font-size:.88rem;
      letter-spacing:.3px;text-transform:uppercase}
input[type=text],input[type=password],input[type=url]{
    width:100%;padding:10px 13px;background:#0d1117;border:1px solid #30363d;
    border-radius:7px;color:#c9d1d9;font-size:.95rem;margin-bottom:6px;
    transition:border .15s,box-shadow .15s}
input:focus{outline:none;border-color:#58a6ff;
            box-shadow:0 0 0 3px rgba(88,166,255,.15)}
.hint{color:#8b949e;font-size:.77rem;margin-bottom:16px}
.hint a{color:#58a6ff;text-decoration:none}
.hint a:hover{text-decoration:underline}
.btn-scan{background:linear-gradient(135deg,#238636,#2ea043);color:#fff;
          border:none;padding:13px 24px;border-radius:7px;font-size:1rem;
          font-weight:700;cursor:pointer;width:100%;letter-spacing:.3px;
          transition:opacity .15s;margin-top:4px}
.btn-scan:hover{opacity:.88}
.btn-scan:disabled{background:#21262d;color:#484f58;cursor:not-allowed;opacity:1}
.btn-demo{background:linear-gradient(135deg,#1a3a5c,#1f4e79);color:#58a6ff;
          border:2px solid #58a6ff55;padding:13px 24px;border-radius:7px;
          font-size:1rem;font-weight:700;cursor:pointer;width:100%;letter-spacing:.3px;
          transition:all .15s;margin-top:0}
.btn-demo:hover{background:linear-gradient(135deg,#1f4e79,#2563a8);
                border-color:#58a6ff99}
.btn-demo:disabled{background:#21262d;color:#484f58;cursor:not-allowed;
                   border-color:#30363d}
.demo-card{background:#0d1520;border:2px solid #58a6ff40;border-radius:10px;
           padding:28px;margin-bottom:20px;position:relative;overflow:hidden}
.demo-card::before{content:'DEMO';position:absolute;top:12px;right:14px;
                   color:#58a6ff;font-size:.68rem;font-weight:700;letter-spacing:1.5px;
                   background:#58a6ff18;border:1px solid #58a6ff40;border-radius:12px;
                   padding:2px 10px}
.demo-card h2{color:#58a6ff;font-size:1.05rem;margin-bottom:8px;
              border-bottom:1px solid #1a3a5c;padding-bottom:10px}
.demo-badge-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.demo-badge{background:#58a6ff12;border:1px solid #58a6ff40;border-radius:16px;
            padding:4px 12px;font-size:.74rem;color:#58a6ff}
.demo-warning{background:#d2992215;border:1px solid #d2992240;border-radius:7px;
              padding:10px 14px;margin-bottom:16px;font-size:.82rem;color:#d29922}
.demo-repo-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.demo-repo-label{color:#8b949e;font-size:.8rem;white-space:nowrap}
.demo-repo-url{color:#58a6ff;font-family:monospace;font-size:.8rem;
               word-break:break-all;text-decoration:none}
.demo-repo-url:hover{text-decoration:underline}
.demo-scan-banner{background:#0d1520;border:2px solid #58a6ff40;border-radius:8px;
                  padding:12px 18px;margin-bottom:16px;display:flex;
                  align-items:center;gap:10px;font-size:.85rem}
.demo-scan-banner strong{color:#58a6ff}
.incomplete-banner{background:#f8514912;border:2px solid #f8514960;border-radius:8px;
                   padding:12px 18px;margin-bottom:16px;display:flex;
                   align-items:center;gap:10px;font-size:.85rem;color:#f85149}
.incomplete-banner strong{color:#f85149}
.model-pill{display:inline-block;background:#161b22;border:1px solid #30363d;
            border-radius:6px;padding:2px 8px;font-family:monospace;font-size:.72rem;
            color:#8b949e;margin-left:4px}
.model-pill-free{border-color:#3fb95060;color:#3fb950;background:#3fb95010}
.model-pill-fallback{border-color:#d2992260;color:#d29922;background:#d2992210}
.note{text-align:center;color:#484f58;font-size:.78rem;margin-top:10px}
.fw-grid{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 24px}
.fw-pill{background:#161b22;border:1px solid #30363d;border-radius:20px;
         padding:5px 12px;font-size:.75rem;color:#8b949e;cursor:default}
.fw-pill:hover{border-color:#58a6ff;color:#58a6ff}
.or-divider{text-align:center;color:#484f58;font-size:.82rem;margin:6px 0 16px;
            position:relative}
.or-divider::before,.or-divider::after{content:'';position:absolute;top:50%;
    width:40%;height:1px;background:#21262d}
.or-divider::before{left:0}
.or-divider::after{right:0}
.progress-box{background:#0d1117;border:1px solid #30363d;border-radius:8px;
              padding:16px;margin-bottom:20px;max-height:420px;overflow-y:auto;
              font-family:'SFMono-Regular',Consolas,monospace}
.pline{font-size:.82rem;padding:2px 0;color:#8b949e}
.pline-ok{color:#3fb950}
.pline-err{color:#f85149}
.pline-work{color:#58a6ff}
.pline-info{color:#d29922}
.pline-warn{color:#d29922;font-weight:700}
h2.section{color:#58a6ff;font-size:1.15rem;margin:32px 0 14px;
           border-bottom:1px solid #21262d;padding-bottom:8px;
           display:flex;align-items:center;gap:8px}
.stats-row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}
.stat-box{flex:1;min-width:120px;background:#0d1117;border:1px solid #30363d;
          border-radius:8px;padding:14px;text-align:center}
.stat-num{font-size:1.7rem;font-weight:700}
.stat-lbl{font-size:.74rem;color:#8b949e;margin-top:3px;text-transform:uppercase;
          letter-spacing:.5px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;
      padding:16px;margin-bottom:10px;transition:border-color .15s}
.card:hover{border-color:#444c56}
.card-critical{border-left:4px solid #f85149}
.card-high{border-left:4px solid #d29922}
.card-medium{border-left:4px solid #58a6ff}
.card-enf{border-left:4px solid #d29922}
.card-reg{border-left:4px solid #3fb950}
.v-header{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:8px}
.v-file{color:#58a6ff;font-family:monospace;font-size:.82rem}
.v-func{color:#d2a8ff;font-family:monospace;font-size:.82rem}
.v-lines{color:#8b949e;font-family:monospace;font-size:.78rem}
.v-reg{color:#3fb950;font-weight:700;font-size:.82rem;margin-bottom:4px}
.v-desc{color:#c9d1d9;font-size:.9rem}
.badge{display:inline-flex;align-items:center;padding:2px 9px;border-radius:12px;
       font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.badge-critical{background:#f8514922;color:#f85149;border:1px solid #f8514966}
.badge-high{background:#d2992222;color:#d29922;border:1px solid #d2992266}
.badge-medium{background:#58a6ff22;color:#58a6ff;border:1px solid #58a6ff66}
.enf-header{display:flex;justify-content:space-between;flex-wrap:wrap;
            align-items:center;margin-bottom:6px}
.enf-company{color:#d29922;font-weight:700;font-size:1rem}
.enf-fine{color:#f85149;font-weight:700;font-size:1.1rem}
.enf-meta{color:#8b949e;font-size:.78rem;margin-bottom:6px}
.enf-match{margin-top:8px;padding:7px 10px;background:#f8514912;border:1px solid #f8514940;
           border-radius:5px;font-size:.83rem;color:#f85149}
.reg-header{display:flex;justify-content:space-between;flex-wrap:wrap;
            align-items:center;margin-bottom:4px}
.reg-name{color:#3fb950;font-weight:700}
.reg-auth{color:#8b949e;font-size:.78rem}
.reg-date{color:#8b949e;font-size:.78rem;margin-bottom:4px}
.reg-impact{margin-top:6px;padding:6px 10px;background:#3fb95012;
            border-left:3px solid #3fb950;border-radius:0 4px 4px 0;
            font-size:.83rem;color:#3fb950}
.roi-wrap{background:#0d1117;border:2px solid #238636;border-radius:10px;
          padding:28px;margin:8px 0 24px;text-align:center}
.roi-title{color:#3fb950;font-size:1.1rem;font-weight:700;margin-bottom:18px;
           text-transform:uppercase;letter-spacing:.8px}
.roi-grid{display:flex;flex-wrap:wrap;gap:16px;justify-content:center;margin-bottom:18px}
.roi-cell{background:#161b22;border:1px solid #30363d;border-radius:8px;
          padding:14px 20px;min-width:150px;flex:1}
.roi-cell-label{font-size:.74rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:.5px;margin-bottom:4px}
.roi-cell-val{font-size:1.35rem;font-weight:700}
.roi-green{color:#3fb950}
.roi-red{color:#f85149}
.roi-blue{color:#58a6ff}
.roi-big{font-size:2.2rem;font-weight:700;color:#58a6ff;margin:8px 0}
.roi-sub{color:#8b949e;font-size:.85rem;margin-top:4px}
.token-table{width:100%;border-collapse:collapse;font-size:.83rem;margin-top:10px;text-align:left}
.token-table th{color:#8b949e;font-weight:600;padding:5px 10px;border-bottom:1px solid #21262d;
                text-transform:uppercase;font-size:.72rem;letter-spacing:.4px}
.token-table td{padding:5px 10px;border-bottom:1px solid #21262d12;color:#c9d1d9}
.token-table tr:last-child td{border-bottom:none;color:#3fb950;font-weight:600}
.err-box{background:#f8514912;border:1px solid #f8514960;border-radius:7px;
         padding:10px 14px;margin:6px 0;color:#f85149;font-size:.87rem}
.site-footer{text-align:center;color:#8b949e;font-size:.85rem;margin-top:48px;
             padding:24px 0;border-top:1px solid #21262d}
.site-footer a{color:#58a6ff;text-decoration:none;font-weight:600}
.contact-box{display:inline-block;background:#161b22;border:1px solid #58a6ff40;
             border-radius:8px;padding:14px 28px;margin-top:10px}
.contact-box p{color:#c9d1d9;margin-bottom:4px}
@media(max-width:600px){
  .site-header h1{font-size:1.4rem}
  .roi-big{font-size:1.6rem}
  .stats-row{gap:6px}
  .stat-box{min-width:90px}
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HTML Helpers
# ═══════════════════════════════════════════════════════════════════════════════

FLUSH_PAD = "<!-- " + ("x" * 1024) + " -->\n"
_LINE_FLUSH = "<!-- " + ("f" * 512) + " -->\n"

FRAMEWORKS_HTML = "".join(
    f'<span class="fw-pill" title="{escape(desc)}">#{n} {escape(name)}</span>'
    for n, name, desc in UAE_FRAMEWORKS
)


def _p(cls: str, msg: str) -> str:
    return f'<div class="pline {cls}">{msg}</div>\n{_LINE_FLUSH}'


def _badge(severity: str) -> str:
    s = str(severity).lower()
    if s == "critical":
        return '<span class="badge badge-critical">⬤ Critical</span>'
    if s == "high":
        return '<span class="badge badge-high">⬤ High</span>'
    return '<span class="badge badge-medium">⬤ Medium</span>'


def _card_cls(severity: str) -> str:
    s = str(severity).lower()
    if s == "critical": return "card card-critical"
    if s == "high":     return "card card-high"
    return "card card-medium"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Landing Page
# ═══════════════════════════════════════════════════════════════════════════════

def index_html() -> str:
    demo_section = f"""
<div class="demo-card">
  <h2>⚡ Free-Model Demo — Powered by OpenRouter Auto-Router</h2>
  <div class="demo-badge-row">
    <span class="demo-badge">🆓 Free-tier models ($0 cost)</span>
    <span class="demo-badge">🔒 Public repos only</span>
    <span class="demo-badge">📄 First {MAX_DEMO_FILES} files scanned</span>
    <span class="demo-badge">🔑 Your key required</span>
    <span class="demo-badge">🤖 Auto best-model selection</span>
  </div>
  <div class="demo-warning">
    ⚠ <strong>Rate limits:</strong> Free-tier models on OpenRouter allow
    20 requests/min · 200 requests/day per key. The scanner automatically
    retries with exponential backoff. OpenRouter's auto-router picks the
    least-loaded free model on every request.
    <strong>You will NOT be charged anything.</strong>
    <a href="https://openrouter.ai/keys" target="_blank" style="color:#58a6ff">
      Get a free key →
    </a>
  </div>
  <div class="demo-repo-row">
    <span class="demo-repo-label">Default repo:</span>
    <a href="{escape(DEMO_REPO_URL)}" class="demo-repo-url"
       target="_blank" rel="noopener">{escape(DEMO_REPO_URL)}</a>
  </div>
  <form method="POST" action="/demo" id="df">
    <label for="demo_repo_url">Public GitHub Repo URL
      <span style="color:#484f58;font-weight:400;text-transform:none">
        (optional — defaults to sample repo above)</span>
    </label>
    <input type="url" id="demo_repo_url" name="repo_url"
           placeholder="{escape(DEMO_REPO_URL)}">
    <p class="hint">Must be a <strong>public</strong> repository.</p>
    <label for="demo_api_key">OpenRouter API Key
      <span style="color:#f85149;font-weight:700;text-transform:none">(REQUIRED)</span>
    </label>
    <input type="password" id="demo_api_key" name="demo_api_key"
           placeholder="sk-or-v1-xxxxxxxxxxxxxxxxxxxx"
           autocomplete="off" required>
    <p class="hint">
      <strong>Free models require your own OpenRouter key.</strong>
      <a href="https://openrouter.ai/keys" target="_blank">Get one free (30 seconds) →</a><br>
      <strong style="color:#3fb950">$0.00 billed</strong> — all calls use
      <code>openrouter/auto:free</code>, OpenRouter picks the best available free model per request.<br>
    </p>
    <div style="background:#0d1117;border:1px solid #30363d;border-radius:7px;
                padding:10px 14px;margin-bottom:16px;font-size:.78rem;color:#8b949e">
      <strong style="color:#c9d1d9">Models (all :free — $0.00 billed):</strong><br>
      Regulations: <span class="model-pill model-pill-free">openrouter/auto:free</span>
      &nbsp; Enforcement: <span class="model-pill model-pill-free">openrouter/auto:free</span>
      &nbsp; Audit: <span class="model-pill model-pill-free">openrouter/auto:free</span><br>
      <span style="color:#484f58;font-size:.72rem">
        OpenRouter auto-selects the best available free model per request
      </span>
    </div>
    <button type="submit" class="btn-demo" id="dbtn">
      ⚡ Run Free Demo Scan
    </button>
  </form>
</div>
<div class="or-divider">— or use paid models for a full scan —</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UAE Compliance Scanner</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<div class="site-header">
  <span style="font-size:2.2rem">🇦🇪</span>
  <h1>UAE Compliance Scanner</h1>
</div>
<p class="tagline">
  Scan any GitHub repository against <strong>10 UAE regulatory frameworks</strong>
  + live enforcement actions + regulatory updates.<br>
  Powered by <strong>Perplexity · Grok · Claude</strong> via OpenRouter.
  Auto-retry · Rate-limit recovery · Free fallback safety net · Never stores credentials.
</p>
<div style="margin-bottom:28px">
  <p style="color:#8b949e;font-size:.8rem;margin-bottom:8px;text-transform:uppercase;
            letter-spacing:.5px;font-weight:600">Regulatory Frameworks Checked</p>
  <div class="fw-grid">{FRAMEWORKS_HTML}</div>
</div>
{demo_section}
<div class="form-card">
  <h2>🔍 Full Scan — Bring Your Own Key</h2>
  <form method="POST" action="/scan" id="sf">
    <label for="repo_url">GitHub Repository URL</label>
    <input type="url" id="repo_url" name="repo_url"
           placeholder="https://github.com/org/repo" required>
    <p class="hint">Public or private HTTPS URL</p>
    <label for="pat">GitHub Personal Access Token
      <span style="color:#484f58;font-weight:400">(optional for public repos)</span>
    </label>
    <input type="password" id="pat" name="pat"
           placeholder="ghp_xxxxxxxxxxxxxxxxxxxx" autocomplete="off">
    <p class="hint">
      Fine-grained token, read-only, scoped to this repo only.
      <a href="https://github.com/settings/tokens?type=beta" target="_blank">Generate →</a>
    </p>
    <label for="api_key">OpenRouter API Key</label>
    <input type="password" id="api_key" name="api_key"
           placeholder="sk-or-v1-xxxxxxxxxxxxxxxxxxxx" required autocomplete="off">
    <p class="hint">
      Routes to Perplexity + Grok + Claude automatically.
      Falls back to <code>openrouter/auto:free</code> if paid models fail.
      <a href="https://openrouter.ai/keys" target="_blank">Get your key →</a>
    </p>
    <button type="submit" class="btn-scan" id="btn">🔍 Start Full Compliance Scan</button>
  </form>
</div>
<p class="note">🔒 PAT and API key are used only for this scan session and are never stored or logged.</p>
<script>
document.getElementById('sf').addEventListener('submit',function(){{
  var b=document.getElementById('btn');
  b.disabled=true;b.textContent='⏳ Scanning — this may take several minutes…';
}});
document.getElementById('df').addEventListener('submit',function(){{
  var b=document.getElementById('dbtn');
  b.disabled=true;b.textContent='⏳ Running demo — please wait…';
}});
</script>
</div></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Report Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_report(
    violations:     list[dict],
    enforcements:   list[dict],
    regulations:    list[dict],
    total_in:       int,
    total_out:      int,
    total_cost:     float,
    cost_by_model:  dict,
    num_files:      int,
    errors:         list[str],
    stats:          ScanStats,
    is_demo:        bool = False,
) -> str:
    h: list[str] = []

    # ── Demo banner ──────────────────────────────────────────────────────────
    if is_demo:
        h.append(f"""
<div class="demo-scan-banner">
  <span style="font-size:1.4rem">⚡</span>
  <div>
    <strong>Demo Scan</strong> — openrouter/auto:free · auto best-model selection ·
    first {MAX_DEMO_FILES} files · <strong>$0.00 billed</strong><br>
    <span style="color:#8b949e">{stats.summary()}</span>
  </div>
</div>
""")

    # ── Audit incomplete banner ───────────────────────────────────────────────
    if stats.is_incomplete:
        h.append(f"""
<div class="incomplete-banner">
  <span style="font-size:1.4rem">⚠</span>
  <div>
    <strong>Audit incomplete due to rate limits</strong> —
    {stats.failed_batches} batch(es) could not be audited after all retries exhausted.<br>
    Violations below reflect only the {stats.successful_batches} successfully audited batch(es).
    Re-run the scan or wait a few minutes and try again.
  </div>
</div>
""")

    # ── Summary stats ──────────────────────────────────────────────────────────
    crit = sum(1 for v in violations if str(v.get("severity","")).lower() == "critical")
    high = sum(1 for v in violations if str(v.get("severity","")).lower() == "high")
    med  = sum(1 for v in violations if str(v.get("severity","")).lower() == "medium")

    h.append('<h2 class="section">📊 Scan Summary</h2>')
    h.append('<div class="stats-row">')
    for val, lbl, col in [
        (num_files,        "Files Scanned",  "#58a6ff"),
        (len(violations),  "Violations",     "#f85149"),
        (crit,             "Critical",       "#f85149"),
        (high,             "High",           "#d29922"),
        (med,              "Medium",         "#58a6ff"),
        (len(regulations), "New Regs",       "#3fb950"),
        (len(enforcements),"Enfrc. Cases",   "#d29922"),
    ]:
        h.append(
            f'<div class="stat-box">'
            f'<div class="stat-num" style="color:{col}">{val}</div>'
            f'<div class="stat-lbl">{lbl}</div>'
            f'</div>'
        )
    h.append('</div>')

    # ── Violations ─────────────────────────────────────────────────────────────
    h.append('<h2 class="section">🚨 Code Violations</h2>')
    if violations:
        order = {"critical": 0, "high": 1, "medium": 2}
        violations.sort(key=lambda v: order.get(str(v.get("severity","")).lower(), 3))
        for v in violations:
            sev = str(v.get("severity", "medium")).lower()
            h.append(f'<div class="{_card_cls(sev)}">')
            h.append(
                f'<div class="v-header">'
                f'<span class="v-file">📄 {escape(str(v.get("file","?")))}</span>'
                f'<span style="color:#484f58">›</span>'
                f'<span class="v-func">⚙ {escape(str(v.get("function_name","N/A")))}</span>'
                f'<span class="v-lines">L{v.get("line_start","?")}–L{v.get("line_end","?")}</span>'
                f'{_badge(sev)}'
                f'</div>'
            )
            h.append(f'<div class="v-reg">⚖ {escape(str(v.get("regulation","N/A")))}</div>')
            h.append(f'<div class="v-desc">{escape(str(v.get("violation_description","")))}</div>')
            h.append('</div>')
    elif stats.is_incomplete:
        h.append(
            '<div class="card" style="border-left:4px solid #f85149">'
            '⚠ No violations found in the <strong>audited</strong> files, '
            'but some batches failed. Re-run for a complete picture.'
            '</div>'
        )
    else:
        h.append(
            '<div class="card" style="border-left:4px solid #3fb950">'
            '✅ No violations detected across all 10 UAE regulatory frameworks.'
            '</div>'
        )

    # ── Enforcement actions ────────────────────────────────────────────────────
    enf_label = "(via openrouter/auto:free — demo)" if is_demo else "(via Grok — trending)"
    h.append(
        f'<h2 class="section">⚖ Recent UAE Enforcement Actions '
        f'<span style="font-size:.75rem;color:#8b949e;font-weight:400">{enf_label}</span></h2>'
    )
    if enforcements:
        for e in enforcements:
            try:
                fine_usd = f"${int(float(e.get('fine_amount_usd', 0))):,}"
            except (ValueError, TypeError):
                fine_usd = str(e.get("fine_amount_usd", "N/A"))

            enf_text = str(e.get("violation","")).lower()
            kws = ["pdpl","aml","vara","cbuae","kyc","data protection","privacy",
                   "sanction","compliance","adgm","difc","nesa","sca","tax"]
            matched = [
                v for v in violations
                if any(
                    k in enf_text and (
                        k in str(v.get("violation_description","")).lower()
                        or k in str(v.get("regulation","")).lower()
                    )
                    for k in kws
                )
            ]
            h.append('<div class="card card-enf">')
            h.append(
                f'<div class="enf-header">'
                f'<span class="enf-company">🏢 {escape(str(e.get("company","Unknown")))}</span>'
                f'<span class="enf-fine">{fine_usd}</span>'
                f'</div>'
            )
            h.append(
                f'<div class="enf-meta">'
                f'{escape(str(e.get("authority","")))} &nbsp;·&nbsp; {escape(str(e.get("date","")))}'
                f'</div>'
            )
            h.append(f'<div class="v-desc">{escape(str(e.get("violation","")))}</div>')
            if e.get("details"):
                h.append(
                    f'<div style="color:#8b949e;font-size:.83rem;margin-top:4px">'
                    f'{escape(str(e.get("details",""))[:300])}</div>'
                )
            if matched:
                h.append(
                    f'<div class="enf-match">'
                    f'⚠ <strong>{len(matched)} violation(s) in your code match this enforcement type</strong>'
                    f'</div>'
                )
            h.append('</div>')
    else:
        h.append('<div class="card">No enforcement data retrieved.</div>')

    # ── Regulatory updates ─────────────────────────────────────────────────────
    reg_label = "(via openrouter/auto:free — demo)" if is_demo else "(via Perplexity)"
    h.append(
        f'<h2 class="section">📜 New UAE Regulatory Updates '
        f'<span style="font-size:.75rem;color:#8b949e;font-weight:400">{reg_label}</span></h2>'
    )
    if regulations:
        for r in regulations:
            h.append('<div class="card card-reg">')
            h.append(
                f'<div class="reg-header">'
                f'<span class="reg-name">📋 {escape(str(r.get("regulation","Unknown")))}</span>'
                f'<span class="reg-auth">{escape(str(r.get("authority","")))}</span>'
                f'</div>'
            )
            if r.get("date"):
                h.append(f'<div class="reg-date">🗓 {escape(str(r.get("date","")))}</div>')
            h.append(f'<div class="v-desc">{escape(str(r.get("summary","")))}</div>')
            if r.get("impact_on_code"):
                h.append(
                    f'<div class="reg-impact">'
                    f'💻 Code Impact: {escape(str(r.get("impact_on_code",""))[:300])}</div>'
                )
            h.append('</div>')
    else:
        h.append('<div class="card">No regulatory update data retrieved.</div>')

    # ── ROI Analysis ───────────────────────────────────────────────────────────
    h.append('<h2 class="section">💰 ROI Analysis</h2>')

    fines = []
    for e in enforcements:
        try:
            f = float(e.get("fine_amount_usd", 0))
            if f > 0:
                fines.append(f)
        except (ValueError, TypeError):
            pass

    lowest_fine  = min(fines) if fines else 0.0
    highest_fine = max(fines) if fines else 0.0
    roi_pct      = ((lowest_fine - total_cost) / total_cost * 100) if total_cost > 0 and lowest_fine > 0 else 0
    per_dollar   = (lowest_fine / total_cost) if total_cost > 0 and lowest_fine > 0 else 0

    h.append('<div class="roi-wrap">')
    h.append('<div class="roi-title">📈 Scan Cost vs. Regulatory Risk</div>')
    h.append('<div class="roi-grid">')

    scan_cost_display = '$0.0000' if is_demo else f'${total_cost:.4f}'
    h.append(
        f'<div class="roi-cell">'
        f'<div class="roi-cell-label">{"Scan Cost (Your Key)" if is_demo else "Scan Cost"}</div>'
        f'<div class="roi-cell-val roi-green">{scan_cost_display}</div>'
        f'</div>'
    )
    if lowest_fine > 0:
        h.append(
            f'<div class="roi-cell">'
            f'<div class="roi-cell-label">Lowest Recent Fine</div>'
            f'<div class="roi-cell-val roi-red">${lowest_fine:,.0f}</div>'
            f'</div>'
        )
        h.append(
            f'<div class="roi-cell">'
            f'<div class="roi-cell-label">Highest Recent Fine</div>'
            f'<div class="roi-cell-val roi-red">${highest_fine:,.0f}</div>'
            f'</div>'
        )
    h.append(
        f'<div class="roi-cell">'
        f'<div class="roi-cell-label">Total Tokens</div>'
        f'<div class="roi-cell-val roi-blue">{total_in + total_out:,}</div>'
        f'</div>'
    )
    h.append('</div>')  # roi-grid

    if is_demo and violations:
        h.append('<div class="roi-big" style="color:#3fb950">$0.00 ✓</div>')
        h.append(
            '<div class="roi-sub">Free-tier models via openrouter/auto:free — $0.00 billed. '
            'Run a <a href="/" style="color:#58a6ff">full BYOK scan</a> for complete coverage.</div>'
        )
    elif lowest_fine > 0 and violations:
        h.append(f'<div class="roi-big">{roi_pct:,.0f}% ROI</div>')
        h.append(
            f'<div class="roi-sub">For every <strong>$1</strong> spent on this scan, '
            f'you could avoid up to <strong>${per_dollar:,.0f}</strong> in regulatory fines.</div>'
        )
    elif not violations:
        h.append(
            '<div style="color:#3fb950;font-size:1.1rem;margin:12px 0">'
            '✅ No violations found — your codebase appears compliant! 🎉</div>'
        )
    else:
        h.append('<div class="roi-sub">No fine data available for ROI calculation.</div>')

    # Token / cost breakdown table
    h.append('<div style="margin-top:20px;overflow-x:auto"><table class="token-table">')
    h.append(
        '<tr><th>Model</th><th>Input Tokens</th><th>Output Tokens</th>'
        '<th>Total</th><th>Cost (USD)</th><th>Calls</th></tr>'
    )
    t_in = t_out = t_cost = 0
    for mdl, d in cost_by_model.items():
        mi, mo, mc = d["input"], d["output"], d["cost"]
        t_in   += mi
        t_out  += mo
        t_cost += mc
        calls   = stats.models_used.get(mdl, "-")
        pill    = ' <span class="model-pill model-pill-free">free</span>' if is_demo else ""
        cost_td = "$0.0000 ✓" if is_demo else f"${mc:.4f}"
        h.append(
            f'<tr><td>{escape(mdl)}{pill}</td><td>{mi:,}</td><td>{mo:,}</td>'
            f'<td>{mi+mo:,}</td><td>{cost_td}</td><td>{calls}</td></tr>'
        )
    total_cost_td = "$0.0000 ✓" if is_demo else f"${t_cost:.4f}"
    h.append(
        f'<tr><td>TOTAL</td><td>{t_in:,}</td><td>{t_out:,}</td>'
        f'<td>{t_in+t_out:,}</td><td>{total_cost_td}</td><td></td></tr>'
    )
    h.append('</table></div>')

    if stats.fallback_activations > 0:
        h.append(
            f'<div style="margin-top:10px;color:#d29922;font-size:.82rem">'
            f'🔀 {stats.fallback_activations} fallback model activation(s) during this scan.</div>'
        )

    h.append('</div>')  # roi-wrap

    # ── Errors ────────────────────────────────────────────────────────────────
    if errors:
        h.append('<h2 class="section">⚠ Warnings &amp; Errors</h2>')
        for err in errors:
            h.append(f'<div class="err-box">{escape(str(err)[:500])}</div>')

    # ── Footer ────────────────────────────────────────────────────────────────
    h.append("""
<div class="site-footer">
  <p>Scanned against 10 UAE regulatory frameworks · live enforcement trends · regulatory updates.</p>
  <div class="contact-box">
    <p>Found violations in your codebase? Need remediation?</p>
    <a href="mailto:virgil3692@proton.me">📩 virgil3692@proton.me</a>
  </div>
  <p style="margin-top:14px;font-size:.75rem">
    UAE Compliance Scanner &nbsp;·&nbsp; Powered by OpenRouter &nbsp;·&nbsp;
    Your credentials are never stored.
  </p>
</div>
""")

    return "\n".join(h)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Streaming Scan Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def stream_scan(repo_url: str, pat: str, api_key: str, is_demo: bool = False):
    """
    Core streaming scan generator.

    is_demo=False  → full BYOK scan (paid models; falls back to openrouter/auto:free)
    is_demo=True   → demo scan (openrouter/auto:free for all roles, $0 billed, 5-file cap)

    Yields HTML chunks that build the page progressively.
    """
    models      = DEMO_MODELS        if is_demo else MODELS
    pricing     = DEMO_MODEL_PRICING if is_demo else MODEL_PRICING
    file_limit  = MAX_DEMO_FILES     if is_demo else MAX_FILES
    batch_size  = DEMO_BATCH_SIZE    if is_demo else BATCH_SIZE
    stats       = ScanStats()

    page_title = "Demo Scan… — UAE Compliance Scanner" if is_demo else "Scanning… — UAE Compliance Scanner"
    scan_label = "Demo scan in progress…"              if is_demo else "Scan in progress…"

    # Demo mode: 45 s per-call timeout (free models can be slow).
    # Full BYOK scan: full 180 s.
    api_timeout = 90 if is_demo else 180

    # ── Initial HTML skeleton ─────────────────────────────────────────────────
    yield f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
{FLUSH_PAD}
<div class="site-header">
  <span style="font-size:2.2rem">🇦🇪</span>
  <h1>UAE Compliance Scanner</h1>
</div>
<p class="tagline">{scan_label}</p>
"""
    if is_demo:
        yield f"""<div class="demo-scan-banner">
  <span style="font-size:1.3rem">⚡</span>
  <div>
    <strong>Demo Mode</strong> — openrouter/auto:free · auto best-model selection ·
    first {MAX_DEMO_FILES} files · batched {batch_size} files/call ·
    auto-retry · <strong>$0.00 billed</strong>
  </div>
</div>
"""

    yield '<div class="progress-box" id="pb">\n'

    # ── Mutable state ─────────────────────────────────────────────────────────
    clone_dir:      str | None  = None
    total_in:       int         = 0
    total_out:      int         = 0
    total_cost:     float       = 0.0
    cost_by_model:  dict        = {}
    all_violations: list[dict]  = []
    regulations:    list[dict]  = []
    enforcements:   list[dict]  = []
    errors:         list[str]   = []
    num_files:      int         = 0

    fw_list = "\n".join(
        f"{n}. {name} — {desc}"
        for n, name, desc in UAE_FRAMEWORKS
    )

    def accumulate_cost(model_id: str, inp: int, out: int) -> None:
        nonlocal total_in, total_out, total_cost
        total_in  += inp
        total_out += out
        cost       = calc_cost(model_id, inp, out, pricing_table=pricing)
        total_cost += cost
        if model_id not in cost_by_model:
            cost_by_model[model_id] = {"input": 0, "output": 0, "cost": 0.0}
        cost_by_model[model_id]["input"]  += inp
        cost_by_model[model_id]["output"] += out
        cost_by_model[model_id]["cost"]   += cost

    try:
        # ── Step 1: Clone ──────────────────────────────────────────────────────
        yield _p("pline-work", "⏳ Step 1/5: Cloning repository…")
        if gitpython is None:
            raise RuntimeError("GitPython not installed. Run: pip install gitpython")

        clone_dir = tempfile.mkdtemp(prefix="uae_scan_")
        auth_url  = repo_url if is_demo else (
            repo_url.replace("https://", f"https://x-access-token:{pat}@")
            if pat else repo_url
        )
        try:
            gitpython.Repo.clone_from(auth_url, clone_dir, depth=1)
        except Exception as exc:
            raise RuntimeError(f"Clone failed: {exc}") from exc

        files     = find_source_files(clone_dir, limit=file_limit)
        num_files = len(files)

        cap_suffix = (
            f" (demo cap: {MAX_DEMO_FILES})" if is_demo and num_files >= MAX_DEMO_FILES
            else f" (capped at {MAX_FILES})"  if num_files >= MAX_FILES
            else ""
        )
        yield _p("pline-ok", f"✅ Cloned. {num_files} scannable file(s) found{cap_suffix}.")

        if num_files == 0:
            yield _p("pline-err", "⚠ No .py / .js / .ts / .sol files found. Scan complete.")

        # ── Step 2: Regulatory updates ─────────────────────────────────────────
        reg_model = models["regulations"]
        yield _p(
            "pline-work",
            f'⏳ Step 2/5: Fetching UAE regulatory updates '
            f'<span class="model-pill model-pill-free">{escape(reg_model)}</span>…'
        )
        try:
            reg_content, reg_inp, reg_out, reg_model_used = openrouter_chat(
                api_key,
                reg_model,
                [
                    {
                        "role": "system",
                        "content": "You are a UAE financial regulation researcher. Return ONLY valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "List the 5 most recent UAE regulatory updates relevant to fintech, "
                            "crypto, and banking software. Cover VARA, ADGM, CBUAE, PDPL, AML, "
                            "NESA, SCA, DIFC, and Consumer Protection regulations. "
                            "Return a JSON array where each object has exactly these keys: "
                            '"regulation", "authority", "date", "summary", "impact_on_code". '
                            "Return ONLY the JSON array, no markdown, no preamble."
                        ),
                    },
                ],
                pricing_table=pricing,
                timeout=api_timeout,
            )
            accumulate_cost(reg_model_used, reg_inp, reg_out)

            parsed = extract_json(reg_content)
            if isinstance(parsed, list):
                regulations = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        regulations = v
                        break

            if not regulations:
                errors.append(f"{reg_model_used}: no regulation list parsed (unexpected format).")
                yield _p("pline-info", "⚠ Regulations model returned unexpected format — continuing.")
            else:
                fallback_note = (
                    f' (routed to: <span class="model-pill model-pill-fallback">'
                    f'{escape(reg_model_used)}</span>)'
                    if reg_model_used != reg_model else ""
                )
                yield _p("pline-ok", f"✅ {len(regulations)} regulatory update(s) fetched.{fallback_note}")

        except OpenRouterError as exc:
            errors.append(f"Regulations model error: {exc}")
            yield _p("pline-err", f"⚠ Regulations model failed: {escape(str(exc)[:300])}")

        # ── Step 3: Enforcement actions ────────────────────────────────────────
        enf_model = models["enforcement"]
        yield _p(
            "pline-work",
            f'⏳ Step 3/5: Fetching UAE enforcement actions '
            f'<span class="model-pill model-pill-free">{escape(enf_model)}</span>…'
        )
        try:
            enf_content, enf_inp, enf_out, enf_model_used = openrouter_chat(
                api_key,
                enf_model,
                [
                    {
                        "role": "system",
                        "content": "You are a UAE enforcement action researcher. Return ONLY valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "List the 3 most recent real UAE enforcement actions against companies "
                            "for financial, fintech, crypto, or data protection violations. "
                            "Include VARA, ADGM, CBUAE, DIFC, NESA, or SCA actions. "
                            "Return a JSON array where each object has exactly these keys: "
                            '"company", "fine_amount_usd" (number only), "violation", '
                            '"authority", "date", "details". '
                            "Return ONLY the JSON array, no markdown, no preamble."
                        ),
                    },
                ],
                pricing_table=pricing,
                timeout=api_timeout,
            )
            accumulate_cost(enf_model_used, enf_inp, enf_out)

            parsed = extract_json(enf_content)
            if isinstance(parsed, list):
                enforcements = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        enforcements = v
                        break

            if not enforcements:
                errors.append(f"{enf_model_used}: no enforcement list parsed.")
                yield _p("pline-info", "⚠ Enforcement model returned unexpected format — continuing.")
            else:
                fallback_note = (
                    f' (routed to: <span class="model-pill model-pill-fallback">'
                    f'{escape(enf_model_used)}</span>)'
                    if enf_model_used != enf_model else ""
                )
                yield _p("pline-ok", f"✅ {len(enforcements)} enforcement action(s) fetched.{fallback_note}")

        except OpenRouterError as exc:
            errors.append(f"Enforcement model error: {exc}")
            yield _p("pline-err", f"⚠ Enforcement model failed after all retries: {escape(str(exc)[:300])}")

        # ── Build additional context for auditor ──────────────────────────────
        additional_regs_lines: list[str] = []
        for r in regulations:
            additional_regs_lines.append(
                f"- NEW REG: {r.get('regulation','')} ({r.get('authority','')}): "
                f"{r.get('summary','')} | Code impact: {r.get('impact_on_code','')}"
            )
        for e in enforcements:
            additional_regs_lines.append(
                f"- TRENDING ENFORCEMENT: {e.get('company','')} fined "
                f"${e.get('fine_amount_usd',0):,} by {e.get('authority','')} for: "
                f"{e.get('violation','')}. Flag similar patterns carefully."
            )
        additional_regs = (
            "\n".join(additional_regs_lines)
            if additional_regs_lines
            else "None available — use hardcoded frameworks only."
        )

        audit_system = build_audit_system_prompt(fw_list, additional_regs)

        # ── Step 4: Batched code audit ─────────────────────────────────────────
        audit_model = models["audit"]
        batches     = make_batches(files, batch_size)

        yield _p(
            "pline-work",
            f'⏳ Step 4/5: Auditing {num_files} file(s) in {len(batches)} batch(es) '
            f'<span class="model-pill model-pill-free">{escape(audit_model)}</span>…'
        )

        for batch_idx, batch in enumerate(batches):
            batch_names   = [rel for _, rel in batch]
            names_escaped = ", ".join(
                f'<span class="v-file">{escape(n)}</span>' for n in batch_names
            )
            yield _p("pline-work", f"&nbsp;&nbsp;• Batch {batch_idx + 1}/{len(batches)}: {names_escaped}")

            try:
                violations, inp, out, model_used = audit_file_batch(
                    api_key,
                    audit_model,
                    batch,
                    audit_system,
                    pricing,
                    stats,
                    timeout=api_timeout,
                )
            except Exception as batch_exc:
                # Catch any unexpected error so one bad batch never kills the stream
                violations, inp, out, model_used = [], 0, 0, audit_model
                stats.record_failure(rate_limited=False)
                errors.append(f"Batch {batch_idx + 1} unexpected error: {batch_exc}")
                yield _p(
                    "pline-err",
                    f"&nbsp;&nbsp;&nbsp;&nbsp;⚠ Batch {batch_idx + 1} unexpected error: "
                    f"{escape(str(batch_exc)[:200])}"
                )

            if model_used != audit_model:
                yield _p(
                    "pline-info",
                    f"&nbsp;&nbsp;&nbsp;&nbsp;🔀 Routed to: "
                    f'<span class="model-pill model-pill-fallback">{escape(model_used)}</span>'
                )

            if inp > 0:
                accumulate_cost(model_used, inp, out)
                all_violations.extend(violations)
                yield _p(
                    "pline-ok",
                    f"&nbsp;&nbsp;&nbsp;&nbsp;✅ {len(violations)} violation(s) found in this batch."
                )
            elif not errors or errors[-1].startswith(f"Batch {batch_idx + 1} unexpected"):
                err_msg = (
                    "Rate limit hit (all retries exhausted)"
                    if stats.rate_limit_hits > 0
                    else "All retries exhausted"
                )
                errors.append(f"Batch {batch_idx + 1} failed ({', '.join(batch_names)}): {err_msg}")
                yield _p(
                    "pline-err",
                    f"&nbsp;&nbsp;&nbsp;&nbsp;⚠ Batch {batch_idx + 1} skipped: {escape(err_msg)}"
                )

            if batch_idx < len(batches) - 1:
                time.sleep(2.0)

        # ── Step 5: Summary ────────────────────────────────────────────────────
        crit_count = sum(1 for v in all_violations if str(v.get("severity","")).lower() == "critical")

        if stats.is_incomplete:
            yield _p(
                "pline-warn",
                f"⚠ Step 5/5: Audit incomplete — "
                f"{stats.failed_batches} batch(es) skipped. "
                f"Violations below are from {stats.successful_batches} successful batch(es) only."
            )
        else:
            yield _p(
                "pline-ok",
                f"✅ Step 5/5: Audit complete — {len(all_violations)} violation(s) "
                f"({crit_count} critical). {stats.summary()}"
            )

        cost_display = "$0.00 (free-tier)" if is_demo else f"${total_cost:.4f} USD"
        yield _p(
            "pline-info",
            f"Tokens: {total_in + total_out:,} ({total_in:,} in + {total_out:,} out) — "
            f"Cost: {cost_display}"
        )

    except Exception as exc:
        tb = traceback.format_exc()
        errors.append(f"Fatal: {exc}\n{tb}")
        yield _p("pline-err", f"❌ Fatal error: {escape(str(exc)[:500])}")
    finally:
        if clone_dir and os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, ignore_errors=True)

    yield '</div>\n'  # close progress-box

    yield render_report(
        all_violations, enforcements, regulations,
        total_in, total_out, total_cost, cost_by_model,
        num_files, errors, stats,
        is_demo=is_demo,
    )

    yield """
<script>
var pb = document.getElementById('pb');
if (pb) pb.scrollTop = pb.scrollHeight;
</script>
</div></body></html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Flask Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return index_html()


@app.route("/scan", methods=["POST"])
def scan():
    """Full BYOK scan using paid models (Perplexity + Grok + Claude).
    Falls back to openrouter/auto:free if all paid retries are exhausted."""
    repo_url = request.form.get("repo_url", "").strip()
    pat      = request.form.get("pat", "").strip()
    api_key  = request.form.get("api_key", "").strip()

    if not repo_url:
        return "Missing repository URL.", 400
    if not api_key:
        return "Missing OpenRouter API key.", 400
    if not repo_url.startswith("https://"):
        return "Repository URL must start with https://", 400

    return Response(
        stream_with_context(stream_scan(repo_url, pat, api_key, is_demo=False)),
        content_type="text/html; charset=utf-8",
    )


@app.route("/demo", methods=["POST"])
def demo_scan():
    """Demo scan using openrouter/auto:free for all roles.
    OpenRouter auto-selects the best available free model per request.
    User must supply their own OpenRouter key — $0.00 billed."""
    repo_url = request.form.get("repo_url", "").strip() or DEMO_REPO_URL
    api_key  = request.form.get("demo_api_key", "").strip()

    if not repo_url.startswith("https://"):
        return "Repository URL must start with https://", 400

    if not api_key:
        return (
            "<h2>OpenRouter API Key Required</h2>"
            "<p>The free demo requires your own OpenRouter key. "
            "All calls use <code>openrouter/auto:free</code> — <strong>$0.00 billed</strong>.</p>"
            "<p>Your key is needed only for rate limiting (20 req/min · 200 req/day).</p>"
            "<p><a href='https://openrouter.ai/keys' target='_blank'>Get a free key →</a></p>"
            "<p><a href='/'>← Back to scanner</a></p>",
            400,
        )

    return Response(
        stream_with_context(stream_scan(repo_url, pat="", api_key=api_key, is_demo=True)),
        content_type="text/html; charset=utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  🇦🇪 UAE Compliance Scanner — Production Build")
    print(f"  http://0.0.0.0:{port}")
    print(f"  Demo repo    : {DEMO_REPO_URL}")
    print(f"  Demo models  : openrouter/auto:free (all roles — $0.00 billed)")
    print(f"  Full fallback: openrouter/auto:free (last resort after paid model exhaustion)")
    print(f"  Rate limit   : {_rate_limiter._min_interval}s between requests (global)")
    print(f"  Batch size   : {DEMO_BATCH_SIZE} files/call (demo)  |  {BATCH_SIZE} files/call (full)")
    print(f"  Retry        : up to 5 attempts with exponential backoff (1s→2→4→8→16)")
    print()
    app.run(debug=False, host="0.0.0.0", port=port)
