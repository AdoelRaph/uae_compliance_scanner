#!/usr/bin/env python3
"""
UAE Compliance Scanner
Single-file Flask app — scans GitHub repos for UAE regulatory compliance violations.
Uses OpenRouter (https://openrouter.ai/api/v1) to access Perplexity, Grok, and Claude.

Requirements:
    pip install flask requests gitpython
"""

import os
import json
import shutil
import tempfile
import re
import traceback
from pathlib import Path
from html import escape

from flask import Flask, request, Response, stream_with_context, session
import requests as http_requests

try:
    import git as gitpython
except ImportError:
    gitpython = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

# ─── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── BYOK (Bring Your Own Key) — original paid models ──────────────────────────
MODELS = {
    "regulations": "perplexity/sonar-deep-research",
    "enforcement": "x-ai/grok-4.20-multi-agent-beta",
    "audit":       "anthropic/claude-sonnet-4-6",
}

MODEL_PRICING = {
    "perplexity/sonar-deep-research":    {"input": 3.00,  "output": 15.00},
    "x-ai/grok-4.20-multi-agent-beta":   {"input": 3.00,  "output": 15.00},
    "anthropic/claude-sonnet-4-6":       {"input": 3.00,  "output": 15.00},
}

# ── Demo mode — free OpenRouter models (operator-funded, not charged to user) ──
#
#   Free-tier limits: 20 req/min · 200 req/day (across all :free models combined)
#   File cap for demo: MAX_DEMO_FILES to avoid exhausting the daily quota in one run.
#
#   Model rationale (March 2026 OpenRouter rankings):
#     regulations  → nvidia/nemotron-super-49b-v1:free  — 1M-token context, hybrid
#                    Mamba-Transformer MoE, best free model for long-document legal
#                    reasoning and cross-document synthesis.
#     enforcement  → meta-llama/llama-3.3-70b-instruct:free  — GPT-4-class general
#                    knowledge, fast, highly available; ideal for trend summarisation.
#     audit        → qwen/qwen3-coder-480b-a35b:free  — top-ranked free coding model
#                    on OpenRouter (262K context, state-of-the-art code generation).

DEMO_OPENROUTER_KEY = os.environ.get("DEMO_OPENROUTER_KEY", "")   # YOUR key goes here
DEMO_REPO_URL       = os.environ.get(                             # default public repo
    "DEMO_REPO_URL",
    "https://github.com/firmai/financial-machine-learning",
)
MAX_DEMO_FILES = 5     # keep well inside the 200 req/day free-tier budget

DEMO_MODELS = {
    # Long-context legal/doc reasoning — 1M token window
    "regulations": "nvidia/nemotron-super-49b-v1:free",
    # General knowledge / trend summarisation — GPT-4-class
    "enforcement": "meta-llama/llama-3.3-70b-instruct:free",
    # Best free code-audit model — 262K context, 480B params
    "audit":       "qwen/qwen3-coder-480b-a35b:free",
}

# Free-tier models are $0, so cost tracking shows $0.0000 intentionally
DEMO_MODEL_PRICING = {
    "nvidia/nemotron-super-49b-v1:free":       {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.3-70b-instruct:free":  {"input": 0.0, "output": 0.0},
    "qwen/qwen3-coder-480b-a35b:free":         {"input": 0.0, "output": 0.0},
}

UAE_FRAMEWORKS = [
    ("1", "PDPL Federal Decree-Law 45/2021",     "Personal Data Protection — consent, data minimisation, cross-border transfers, subject rights"),
    ("2", "AML Law 20/2018",                      "Anti-Money Laundering — KYC, transaction monitoring, suspicious activity reporting, record keeping"),
    ("3", "CBUAE Decree-Law 6/2025",              "Central Bank — payment services licensing, stored value, open banking, consumer protection"),
    ("4", "VARA Virtual Asset Regulations 2023",  "Virtual assets — exchange, custody, issuance, wallet services, travel rule"),
    ("5", "DIFC DPL 5/2020",                      "DIFC Data Protection Law — lawful basis, data transfers, controller obligations"),
    ("6", "NESA IA-7",                            "National Electronic Security Authority — cybersecurity controls, encryption, access management, logging"),
    ("7", "UAE Corporate Tax Law 47/2022",         "Corporate tax compliance in financial logic — revenue recognition, transfer pricing, reporting"),
    ("8", "Cabinet Resolution 58/2020",           "AML/CFT — beneficial ownership registers, enhanced due diligence, cross-border wire rules"),
    ("9", "SCA Board Decision 23/2020",           "Securities and Commodities Authority — fintech, robo-advisory, crowdfunding, investment platforms"),
    ("10","UAE Consumer Protection Law 15/2020",  "Data handling, pricing disclosure, complaint mechanisms, unfair terms in consumer-facing code"),
]

TARGET_EXTENSIONS = {".py", ".js", ".ts", ".sol"}
SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "env", "dist",
    "build", ".git", ".tox", ".mypy_cache", "coverage", ".next", ".nuxt",
}
MAX_FILES      = 50
MAX_FILE_CHARS = 30_000

# ─── OpenRouter helper ────────────────────────────────────────────────────────

def openrouter_chat(api_key, model, messages, pricing_table=None):
    """Returns (content, input_tokens, output_tokens)."""
    if pricing_table is None:
        pricing_table = MODEL_PRICING
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://uae-compliance-scanner.local",
        "X-OpenRouter-Title": "UAE Compliance Scanner",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    r = http_requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers=headers,
        data=json.dumps(body),
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    usage   = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def extract_json(text):
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


def calc_cost(model, inp, out, pricing_table=None):
    if pricing_table is None:
        pricing_table = MODEL_PRICING
    p = pricing_table.get(model, {"input": 3.0, "output": 15.0})
    return (inp * p["input"] + out * p["output"]) / 1_000_000


def find_source_files(repo_dir, limit=MAX_FILES):
    results = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if Path(fname).suffix.lower() in TARGET_EXTENSIONS:
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, repo_dir)
                results.append((full, rel))
    return results[:limit]


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0a0e17;color:#c9d1d9;line-height:1.6}
.container{max-width:980px;margin:0 auto;padding:24px 16px}

/* ── Header ── */
.site-header{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.site-header h1{color:#58a6ff;font-size:1.9rem;letter-spacing:-.5px}
.tagline{color:#8b949e;font-size:.92rem;margin-bottom:28px}

/* ── Form card ── */
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

/* ── Demo button ── */
.btn-demo{background:linear-gradient(135deg,#1a3a5c,#1f4e79);color:#58a6ff;
          border:2px solid #58a6ff55;padding:13px 24px;border-radius:7px;
          font-size:1rem;font-weight:700;cursor:pointer;width:100%;letter-spacing:.3px;
          transition:all .15s;margin-top:0}
.btn-demo:hover{background:linear-gradient(135deg,#1f4e79,#2563a8);
                border-color:#58a6ff99;opacity:1}
.btn-demo:disabled{background:#21262d;color:#484f58;cursor:not-allowed;
                   border-color:#30363d}

/* ── Demo card ── */
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

/* ── Demo scan banner ── */
.demo-scan-banner{background:#0d1520;border:2px solid #58a6ff40;border-radius:8px;
                  padding:12px 18px;margin-bottom:16px;display:flex;
                  align-items:center;gap:10px;font-size:.85rem}
.demo-scan-banner strong{color:#58a6ff}

/* ── Model pill ── */
.model-pill{display:inline-block;background:#161b22;border:1px solid #30363d;
            border-radius:6px;padding:2px 8px;font-family:monospace;font-size:.72rem;
            color:#8b949e;margin-left:4px}
.model-pill-free{border-color:#3fb95060;color:#3fb950;background:#3fb95010}

.note{text-align:center;color:#484f58;font-size:.78rem;margin-top:10px}

/* ── Framework badges ── */
.fw-grid{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 24px}
.fw-pill{background:#161b22;border:1px solid #30363d;border-radius:20px;
         padding:5px 12px;font-size:.75rem;color:#8b949e;cursor:default}
.fw-pill:hover{border-color:#58a6ff;color:#58a6ff}

/* ── Divider ── */
.or-divider{text-align:center;color:#484f58;font-size:.82rem;margin:6px 0 16px;
            position:relative}
.or-divider::before,.or-divider::after{content:'';position:absolute;top:50%;
    width:40%;height:1px;background:#21262d}
.or-divider::before{left:0}
.or-divider::after{right:0}

/* ── Progress ── */
.progress-box{background:#0d1117;border:1px solid #30363d;border-radius:8px;
              padding:16px;margin-bottom:20px;max-height:380px;overflow-y:auto;
              font-family:'SFMono-Regular',Consolas,monospace}
.pline{font-size:.82rem;padding:2px 0;color:#8b949e}
.pline-ok{color:#3fb950}
.pline-err{color:#f85149}
.pline-work{color:#58a6ff}
.pline-info{color:#d29922}
.pline-demo{color:#58a6ff;font-style:italic}

/* ── Section heading ── */
h2.section{color:#58a6ff;font-size:1.15rem;margin:32px 0 14px;
           border-bottom:1px solid #21262d;padding-bottom:8px;
           display:flex;align-items:center;gap:8px}

/* ── Stat boxes ── */
.stats-row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}
.stat-box{flex:1;min-width:120px;background:#0d1117;border:1px solid #30363d;
          border-radius:8px;padding:14px;text-align:center}
.stat-num{font-size:1.7rem;font-weight:700}
.stat-lbl{font-size:.74rem;color:#8b949e;margin-top:3px;text-transform:uppercase;
          letter-spacing:.5px}

/* ── Cards ── */
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;
      padding:16px;margin-bottom:10px;transition:border-color .15s}
.card:hover{border-color:#444c56}
.card-critical{border-left:4px solid #f85149}
.card-high{border-left:4px solid #d29922}
.card-medium{border-left:4px solid #58a6ff}
.card-enf{border-left:4px solid #d29922;border-top:1px solid #d2992240}
.card-reg{border-left:4px solid #3fb950;border-top:1px solid #3fb95040}

/* ── Violation details ── */
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

/* ── Enforcement cards ── */
.enf-header{display:flex;justify-content:space-between;flex-wrap:wrap;
            align-items:center;margin-bottom:6px}
.enf-company{color:#d29922;font-weight:700;font-size:1rem}
.enf-fine{color:#f85149;font-weight:700;font-size:1.1rem}
.enf-meta{color:#8b949e;font-size:.78rem;margin-bottom:6px}
.enf-match{margin-top:8px;padding:7px 10px;background:#f8514912;border:1px solid #f8514940;
           border-radius:5px;font-size:.83rem;color:#f85149}

/* ── Regulation cards ── */
.reg-header{display:flex;justify-content:space-between;flex-wrap:wrap;
            align-items:center;margin-bottom:4px}
.reg-name{color:#3fb950;font-weight:700}
.reg-auth{color:#8b949e;font-size:.78rem}
.reg-date{color:#8b949e;font-size:.78rem;margin-bottom:4px}
.reg-impact{margin-top:6px;padding:6px 10px;background:#3fb95012;
            border-left:3px solid #3fb950;border-radius:0 4px 4px 0;
            font-size:.83rem;color:#3fb950}

/* ── ROI box ── */
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

/* ── Error box ── */
.err-box{background:#f8514912;border:1px solid #f8514960;border-radius:7px;
         padding:10px 14px;margin:6px 0;color:#f85149;font-size:.87rem}

/* ── Footer ── */
.site-footer{text-align:center;color:#8b949e;font-size:.85rem;margin-top:48px;
             padding:24px 0;border-top:1px solid #21262d}
.site-footer a{color:#58a6ff;text-decoration:none;font-weight:600}
.site-footer a:hover{text-decoration:underline}
.contact-box{display:inline-block;background:#161b22;border:1px solid #58a6ff40;
             border-radius:8px;padding:14px 28px;margin-top:10px}
.contact-box p{color:#c9d1d9;margin-bottom:4px}
.contact-box a{font-size:1rem}

/* ── Responsive ── */
@media(max-width:600px){
  .site-header h1{font-size:1.4rem}
  .roi-big{font-size:1.6rem}
  .stats-row{gap:6px}
  .stat-box{min-width:90px}
}
"""

# ─── Landing page ─────────────────────────────────────────────────────────────

FRAMEWORKS_HTML = "".join(
    f'<span class="fw-pill" title="{escape(desc)}">#{n} {escape(name)}</span>'
    for n, name, desc in UAE_FRAMEWORKS
)

def _operator_demo_available():
    """True if the operator has pre-loaded a demo key in the environment."""
    return bool(DEMO_OPENROUTER_KEY)


def index_html():
    # Always show the demo card.
    # Key resolution order (handled in /demo route):
    #   1. User's own OpenRouter key (typed into the demo form)
    #   2. Operator's DEMO_OPENROUTER_KEY env var (if set)
    #   3. Neither → show error

    operator_note = (
        "Leave blank to use the shared demo key — or paste your own key below "
        "to run free models <strong>on your own quota</strong>."
        if _operator_demo_available()
        else
        "Free models require an OpenRouter key. "
        "<a href='https://openrouter.io/keys' target='_blank'>Get one free →</a> "
        "and paste it below — you won't be charged for <code>:free</code> models."
    )

    demo_section = f"""
<div class="demo-card">
  <h2>⚡ Free-Model Demo — Runs on Open-Source AI</h2>

  <div class="demo-badge-row">
    <span class="demo-badge">🆓 Free-tier models ($0 cost)</span>
    <span class="demo-badge">🔒 Public repos only</span>
    <span class="demo-badge">📄 First {MAX_DEMO_FILES} files scanned</span>
    <span class="demo-badge">🤖 Your quota or ours</span>
  </div>

  <div class="demo-warning">
    ⚠ <strong>How billing works:</strong>
    All three models used here carry a <code>:free</code> suffix on OpenRouter — they cost
    <strong>$0.00</strong> regardless of whose key is used.
    Your key is only needed so OpenRouter can track rate limits (20 req/min · 200 req/day per key).
    <strong>No charges will appear on your account.</strong>
  </div>

  <div class="demo-repo-row">
    <span class="demo-repo-label">Default repo:</span>
    <a href="{escape(DEMO_REPO_URL)}" class="demo-repo-url"
       target="_blank" rel="noopener">{escape(DEMO_REPO_URL)}</a>
  </div>
  <p class="hint" style="margin-bottom:14px">
    Scans a public fintech sample repo. Override below for any public GitHub URL.
  </p>

  <form method="POST" action="/demo" id="df">

    <label for="demo_repo_url">Public GitHub Repo URL
      <span style="color:#484f58;font-weight:400;text-transform:none">
        (optional — defaults to sample repo above)</span>
    </label>
    <input type="url" id="demo_repo_url" name="repo_url"
           placeholder="{escape(DEMO_REPO_URL)}">
    <p class="hint">Must be a <strong>public</strong> repository. No PAT required.</p>

    <label for="demo_api_key">OpenRouter API Key
      <span style="color:#484f58;font-weight:400;text-transform:none">(optional)</span>
    </label>
    <input type="password" id="demo_api_key" name="demo_api_key"
           placeholder="sk-or-v1-xxxxxxxxxxxxxxxxxxxx — leave blank to use shared demo key"
           autocomplete="off">
    <p class="hint">{operator_note}</p>

    <div style="background:#0d1117;border:1px solid #30363d;border-radius:7px;
                padding:10px 14px;margin-bottom:16px;font-size:.78rem;color:#8b949e">
      <strong style="color:#c9d1d9">Models used (all :free — $0 billed):</strong><br>
      <span style="color:#8b949e">Regulations:</span>
      <span class="model-pill model-pill-free">nvidia/nemotron-super-49b-v1:free</span>
      &nbsp;
      <span style="color:#8b949e">Enforcement:</span>
      <span class="model-pill model-pill-free">meta-llama/llama-3.3-70b-instruct:free</span>
      &nbsp;
      <span style="color:#8b949e">Code audit:</span>
      <span class="model-pill model-pill-free">qwen/qwen3-coder-480b-a35b:free</span>
    </div>

    <button type="submit" class="btn-demo" id="dbtn">
      ⚡ Run Free Demo Scan
    </button>
  </form>
</div>

<div class="or-divider">— or bring your own key for a full scan —</div>
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
  + live trending enforcement actions + new regulation updates.<br>
  Powered by <strong>Perplexity · Grok · Claude</strong> via one OpenRouter key.
  Free. No data stored. Credentials never logged.
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

    <label for="pat">GitHub Personal Access Token <span style="color:#484f58;font-weight:400">(optional for public repos)</span></label>
    <input type="password" id="pat" name="pat"
           placeholder="ghp_xxxxxxxxxxxxxxxxxxxx" autocomplete="off">
    <p class="hint">
      Fine-grained token, read-only, scoped to this repo only.
      <a href="https://github.com/settings/tokens?type=beta" target="_blank">Generate one →</a>
    </p>

    <label for="api_key">OpenRouter API Key</label>
    <input type="password" id="api_key" name="api_key"
           placeholder="sk-or-v1-xxxxxxxxxxxxxxxxxxxx" required autocomplete="off">
    <p class="hint">
      One key — routes to Perplexity + Grok + Claude automatically.
      <a href="https://openrouter.io/keys" target="_blank">Get your key →</a>
    </p>

    <button type="submit" class="btn-scan" id="btn">🔍 Start Full Compliance Scan</button>
  </form>
</div>

<p class="note">🔒 Your PAT and API key are used only for this scan session and are never stored or logged.</p>

<script>
document.getElementById('sf').addEventListener('submit', function() {{
  var b = document.getElementById('btn');
  b.disabled = true;
  b.textContent = '⏳ Scanning — this may take several minutes…';
}});
document.getElementById('df').addEventListener('submit', function() {{
  var b = document.getElementById('dbtn');
  b.disabled = true;
  b.textContent = '⏳ Running demo scan — please wait…';
}});
</script>

</div>
</body>
</html>"""


# ─── Progress line helpers ────────────────────────────────────────────────────

FLUSH_PAD = "<!-- " + ("x" * 1024) + " -->\n"

def _p(cls, msg):
    return f'<div class="pline {cls}">{msg}</div>\n'

def _badge(severity):
    s = str(severity).lower()
    if s == "critical":
        return '<span class="badge badge-critical">⬤ Critical</span>'
    if s == "high":
        return '<span class="badge badge-high">⬤ High</span>'
    return '<span class="badge badge-medium">⬤ Medium</span>'

def _card_cls(severity):
    s = str(severity).lower()
    if s == "critical": return "card card-critical"
    if s == "high":     return "card card-high"
    return "card card-medium"


# ─── Report renderer ──────────────────────────────────────────────────────────

def render_report(violations, enforcements, regulations,
                  total_in, total_out, total_cost, cost_by_model,
                  num_files, errors, is_demo=False, key_source="operator"):
    # key_source: "operator" = shared demo key (truly free for user)
    #             "user"     = user's own key, but :free models so $0 billed
    h = []

    # ── Demo notice banner ────────────────────────────────────────────────────
    if is_demo:
        if key_source == "user":
            key_note = (
                "Running on <strong>your OpenRouter key</strong> with free-tier models — "
                "<strong>$0.00 billed to your account.</strong>"
            )
        else:
            key_note = (
                "Running on the <strong>shared demo key</strong> — "
                "<strong>completely free for you.</strong>"
            )
        h.append(f"""
<div class="demo-scan-banner">
  <span style="font-size:1.4rem">⚡</span>
  <div>
    <strong>Demo Scan</strong> — powered by free open-source models
    (nvidia/nemotron-super · llama-3.3-70b · qwen3-coder-480b).<br>
    {key_note}<br>
    <span style="color:#8b949e">Only the first {MAX_DEMO_FILES} files were audited.
    Run a <a href="/" style="color:#58a6ff">full BYOK scan</a> for complete coverage
    with Perplexity + Grok + Claude.</span>
  </div>
</div>
""")

    crit = sum(1 for v in violations if str(v.get("severity","")).lower() == "critical")
    high = sum(1 for v in violations if str(v.get("severity","")).lower() == "high")
    med  = sum(1 for v in violations if str(v.get("severity","")).lower() == "medium")

    # ── Summary stats ──────────────────────────────────────────────────────────
    h.append('<h2 class="section">📊 Scan Summary</h2>')
    h.append('<div class="stats-row">')
    for val, lbl, col in [
        (num_files,       "Files Scanned", "#58a6ff"),
        (len(violations), "Violations",    "#f85149"),
        (crit,            "Critical",      "#f85149"),
        (high,            "High",          "#d29922"),
        (med,             "Medium",        "#58a6ff"),
        (len(regulations),"New Regs",      "#3fb950"),
        (len(enforcements),"Enfrc. Cases", "#d29922"),
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
            sev = str(v.get("severity","medium")).lower()
            h.append(f'<div class="{_card_cls(sev)}">')
            h.append(f'<div class="v-header">'
                     f'<span class="v-file">📄 {escape(str(v.get("file","?")))}</span>'
                     f'<span style="color:#484f58">›</span>'
                     f'<span class="v-func">⚙ {escape(str(v.get("function_name","N/A")))}</span>'
                     f'<span class="v-lines">L{v.get("line_start","?")}–L{v.get("line_end","?")}</span>'
                     f'{_badge(sev)}'
                     f'</div>')
            h.append(f'<div class="v-reg">⚖ {escape(str(v.get("regulation","N/A")))}</div>')
            h.append(f'<div class="v-desc">{escape(str(v.get("violation_description","")))}</div>')
            h.append('</div>')
    else:
        h.append('<div class="card" style="border-left:4px solid #3fb950">'
                 '✅ No violations detected across all 10 UAE regulatory frameworks + trending updates.</div>')

    # ── Enforcement actions ────────────────────────────────────────────────────
    enf_label = ("(via Llama 3.3 70B — demo)"
                 if is_demo else "(via Grok — trending)")
    h.append(f'<h2 class="section">⚖ Recent UAE Enforcement Actions '
             f'<span style="font-size:.75rem;color:#8b949e;font-weight:400">'
             f'{enf_label}</span></h2>')
    if enforcements:
        for e in enforcements:
            try:
                fine_usd = f"${int(float(e.get('fine_amount_usd', 0))):,}"
            except (ValueError, TypeError):
                fine_usd = str(e.get("fine_amount_usd", "N/A"))

            enf_text = str(e.get("violation","")).lower()
            kws = ["pdpl","aml","vara","cbuae","kyc","data protection","privacy",
                   "sanction","compliance","adgm","difc","nesa","sca","tax"]
            matched = [v for v in violations
                       if any(k in enf_text and
                              (k in str(v.get("violation_description","")).lower() or
                               k in str(v.get("regulation","")).lower())
                              for k in kws)]

            h.append('<div class="card card-enf">')
            h.append(f'<div class="enf-header">'
                     f'<span class="enf-company">🏢 {escape(str(e.get("company","Unknown")))}</span>'
                     f'<span class="enf-fine">{fine_usd}</span>'
                     f'</div>')
            h.append(f'<div class="enf-meta">'
                     f'{escape(str(e.get("authority","")))} &nbsp;·&nbsp; {escape(str(e.get("date","")))}')
            h.append('</div>')
            h.append(f'<div class="v-desc">{escape(str(e.get("violation","")))}</div>')
            if e.get("details"):
                h.append(f'<div style="color:#8b949e;font-size:.83rem;margin-top:4px">'
                         f'{escape(str(e.get("details",""))[:300])}</div>')
            if matched:
                h.append(f'<div class="enf-match">'
                         f'⚠ <strong>{len(matched)} violation(s) in your code match this enforcement type</strong>'
                         f'</div>')
            h.append('</div>')
    else:
        h.append('<div class="card">No enforcement data retrieved.</div>')

    # ── Regulatory updates ─────────────────────────────────────────────────────
    reg_label = ("(via Nemotron 3 Super — demo)"
                 if is_demo else "(via Perplexity)")
    h.append(f'<h2 class="section">📜 New UAE Regulatory Updates '
             f'<span style="font-size:.75rem;color:#8b949e;font-weight:400">'
             f'{reg_label}</span></h2>')
    if regulations:
        for r in regulations:
            h.append('<div class="card card-reg">')
            h.append(f'<div class="reg-header">'
                     f'<span class="reg-name">📋 {escape(str(r.get("regulation","Unknown")))}</span>'
                     f'<span class="reg-auth">{escape(str(r.get("authority","")))}</span>'
                     f'</div>')
            if r.get("date"):
                h.append(f'<div class="reg-date">🗓 {escape(str(r.get("date","")))} </div>')
            h.append(f'<div class="v-desc">{escape(str(r.get("summary","")))}</div>')
            if r.get("impact_on_code"):
                h.append(f'<div class="reg-impact">'
                         f'💻 Code Impact: {escape(str(r.get("impact_on_code",""))[:300])}</div>')
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
    lowest_fine  = min(fines) if fines else 0
    highest_fine = max(fines) if fines else 0

    roi_pct   = ((lowest_fine - total_cost) / total_cost * 100) if total_cost > 0 and lowest_fine > 0 else 0
    per_dollar = (lowest_fine / total_cost) if total_cost > 0 and lowest_fine > 0 else 0

    h.append('<div class="roi-wrap">')
    h.append('<div class="roi-title">📈 Scan Cost vs. Regulatory Risk</div>')

    h.append('<div class="roi-grid">')

    scan_cost_val = (
        '<div class="roi-cell-val roi-green">$0.0000</div>'
        if is_demo else
        f'<div class="roi-cell-val roi-green">${total_cost:.4f}</div>'
    )
    if is_demo and key_source == "user":
        cost_label = "Scan Cost (Your Key)"
        cost_sub   = "Free-tier models — $0 billed to your account."
    elif is_demo:
        cost_label = "Scan Cost (Operator-Funded)"
        cost_sub   = "Completely free — shared demo key used."
    else:
        cost_label = "Scan Cost (You Paid)"
        cost_sub   = None

    h.append(f'<div class="roi-cell"><div class="roi-cell-label">{cost_label}</div>'
             f'{scan_cost_val}</div>')

    if lowest_fine > 0:
        h.append(f'<div class="roi-cell"><div class="roi-cell-label">Lowest Recent Fine</div>'
                 f'<div class="roi-cell-val roi-red">${lowest_fine:,.0f}</div></div>')
        h.append(f'<div class="roi-cell"><div class="roi-cell-label">Highest Recent Fine</div>'
                 f'<div class="roi-cell-val roi-red">${highest_fine:,.0f}</div></div>')
    h.append(f'<div class="roi-cell"><div class="roi-cell-label">Total Tokens</div>'
             f'<div class="roi-cell-val roi-blue">{total_in+total_out:,}</div></div>')
    h.append('</div>')  # roi-grid

    if is_demo and violations:
        h.append('<div class="roi-big" style="color:#3fb950">$0.00 ✓</div>')
        if key_source == "user":
            h.append('<div class="roi-sub">Free-tier models ran on <strong>your key</strong> — '
                     '$0.00 billed to your OpenRouter account. '
                     'Run a <a href="/" style="color:#58a6ff">full BYOK scan</a> '
                     'for complete coverage with Perplexity + Grok + Claude.</div>')
        else:
            h.append('<div class="roi-sub">Shared demo key was used — '
                     'this scan cost you absolutely nothing. '
                     'Run a <a href="/" style="color:#58a6ff">full BYOK scan</a> '
                     'for complete coverage with Perplexity + Grok + Claude.</div>')
    elif lowest_fine > 0 and violations:
        h.append(f'<div class="roi-big">{roi_pct:,.0f}% ROI</div>')
        h.append(f'<div class="roi-sub">For every <strong>$1</strong> spent on this scan, '
                 f'you could avoid up to <strong>${per_dollar:,.0f}</strong> in regulatory fines.</div>')
    elif not violations:
        h.append('<div style="color:#3fb950;font-size:1.1rem;margin:12px 0">'
                 '✅ No violations found — your codebase appears compliant! 🎉</div>')
    else:
        h.append('<div class="roi-sub">No fine data available for ROI calculation.</div>')

    # Token / cost table
    h.append('<div style="margin-top:20px;overflow-x:auto">')
    h.append('<table class="token-table">')
    h.append('<tr><th>Model</th><th>Input Tokens</th><th>Output Tokens</th>'
             '<th>Total Tokens</th><th>Cost (USD)</th></tr>')
    t_in = t_out = t_cost = 0
    for model, d in cost_by_model.items():
        mi, mo, mc = d["input"], d["output"], d["cost"]
        t_in += mi; t_out += mo; t_cost += mc
        free_tag = ' <span class="model-pill model-pill-free">free</span>' if is_demo else ""
        h.append(f'<tr><td>{escape(model)}{free_tag}</td><td>{mi:,}</td><td>{mo:,}</td>'
                 f'<td>{mi+mo:,}</td>'
                 f'<td>{"$0.0000 ✓" if is_demo else f"${mc:.4f}"}</td></tr>')
    h.append(f'<tr><td>TOTAL</td><td>{t_in:,}</td><td>{t_out:,}</td>'
             f'<td>{t_in+t_out:,}</td>'
             f'<td>{"$0.0000 ✓" if is_demo else f"${t_cost:.4f}"}</td></tr>')
    h.append('</table>')
    h.append('</div>')
    h.append('</div>')  # roi-wrap

    # ── Errors ─────────────────────────────────────────────────────────────────
    if errors:
        h.append('<h2 class="section">⚠ Warnings &amp; Errors</h2>')
        for err in errors:
            h.append(f'<div class="err-box">{escape(str(err)[:500])}</div>')

    # ── Footer ─────────────────────────────────────────────────────────────────
    h.append("""
<div class="site-footer">
  <p>Scanned against 10 UAE regulatory frameworks + live enforcement trends + regulatory updates.</p>
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


# ─── Streaming scan pipeline ──────────────────────────────────────────────────

def stream_scan(repo_url, pat, api_key, is_demo=False, key_source="operator"):
    """
    Core scan generator.

    is_demo=False  → full BYOK scan (paid models, user's own key)
    is_demo=True   → demo scan (free :free models, file cap)
      key_source="operator"  → operator's DEMO_OPENROUTER_KEY, truly free for user
      key_source="user"      → user's own key, but :free models so $0 billed
    """
    models       = DEMO_MODELS        if is_demo else MODELS
    pricing      = DEMO_MODEL_PRICING  if is_demo else MODEL_PRICING
    file_limit   = MAX_DEMO_FILES      if is_demo else MAX_FILES

    page_title   = "Demo Scan… — UAE Compliance Scanner" if is_demo else "Scanning… — UAE Compliance Scanner"
    scan_label   = "Demo scan in progress…"               if is_demo else "Scan in progress…"

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

    # Demo banner in the progress page
    if is_demo:
        if key_source == "user":
            key_note = "your key · free models · <strong>$0 billed to you</strong>"
        else:
            key_note = "shared demo key · <strong>completely free for you</strong>"
        yield f"""<div class="demo-scan-banner">
  <span style="font-size:1.3rem">⚡</span>
  <div>
    <strong>Demo Mode</strong> — free open-source models ·
    first {MAX_DEMO_FILES} files only · {key_note}
  </div>
</div>
"""

    yield '<div class="progress-box" id="pb">\n'

    clone_dir      = None
    total_in       = 0
    total_out      = 0
    total_cost     = 0.0
    cost_by_model  = {}
    all_violations = []
    regulations    = []
    enforcements   = []
    errors         = []
    num_files      = 0

    fw_list = "\n".join(
        f"{n}. {name} — {desc}"
        for n, name, desc in UAE_FRAMEWORKS
    )

    try:
        # ── Step 1: Clone ──────────────────────────────────────────────────────
        step_label = "Step 1/5" if not is_demo else "Demo Step 1/5"
        yield _p("pline-work", f"⏳ {step_label}: Cloning repository…")
        if gitpython is None:
            raise RuntimeError("GitPython not installed. Run: pip install gitpython")

        clone_dir = tempfile.mkdtemp(prefix="uae_scan_")

        # Demo mode: never inject a PAT even if one was somehow passed
        if is_demo:
            auth_url = repo_url  # must be a public repo
        else:
            auth_url = (repo_url.replace("https://", f"https://x-access-token:{pat}@")
                        if pat else repo_url)

        try:
            gitpython.Repo.clone_from(auth_url, clone_dir, depth=1)
        except Exception as exc:
            raise RuntimeError(f"Clone failed: {exc}")

        files     = find_source_files(clone_dir, limit=file_limit)
        num_files = len(files)
        cap_note  = f" (demo cap: {file_limit})" if is_demo and num_files >= file_limit else (
                    " (capped at 50)" if num_files >= MAX_FILES else "")
        yield _p("pline-ok", f"✅ Cloned. {num_files} scannable file(s) found{cap_note}.")

        if num_files == 0:
            yield _p("pline-err", "⚠ No .py / .js / .ts / .sol files found.")

        # ── Step 2: Regulatory updates ─────────────────────────────────────────
        reg_model_label = models["regulations"]
        yield _p("pline-work",
                 f"⏳ Step 2/5: Fetching UAE regulatory updates "
                 f'<span class="model-pill model-pill-free">{escape(reg_model_label)}</span>…')
        try:
            content, inp, out = openrouter_chat(
                api_key, models["regulations"],
                [
                    {"role": "system",
                     "content": "You are a UAE financial regulation researcher. Return ONLY valid JSON."},
                    {"role": "user",
                     "content": (
                         "List the 5 most recent UAE regulatory updates relevant to fintech, crypto, "
                         "and banking software. Cover VARA, ADGM, CBUAE, PDPL (Federal Decree-Law 45/2021), "
                         "AML, NESA, SCA, DIFC, and Consumer Protection regulations. "
                         "Return a JSON array where each object has exactly these keys: "
                         '"regulation", "authority", "date", "summary", "impact_on_code". '
                         "Return ONLY the JSON array, no markdown, no preamble."
                     )},
                ],
                pricing_table=pricing,
            )
            total_in  += inp
            total_out += out
            mc         = calc_cost(models["regulations"], inp, out, pricing_table=pricing)
            total_cost += mc
            cost_by_model[models["regulations"]] = {"input": inp, "output": out, "cost": mc}

            parsed = extract_json(content)
            if isinstance(parsed, list):
                regulations = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        regulations = v
                        break
            if not regulations:
                errors.append(f"{reg_model_label} returned unexpected format — no regulation list parsed.")
            yield _p("pline-ok", f"✅ {len(regulations)} new regulatory update(s) fetched.")
        except Exception as exc:
            errors.append(f"Regulations model error: {exc}")
            yield _p("pline-err", f"⚠ Regulations model failed: {escape(str(exc)[:300])}")

        # ── Step 3: Enforcement actions ────────────────────────────────────────
        enf_model_label = models["enforcement"]
        yield _p("pline-work",
                 f"⏳ Step 3/5: Fetching UAE enforcement actions "
                 f'<span class="model-pill model-pill-free">{escape(enf_model_label)}</span>…')
        try:
            content, inp, out = openrouter_chat(
                api_key, models["enforcement"],
                [
                    {"role": "system",
                     "content": "You are a UAE enforcement action researcher. Return ONLY valid JSON."},
                    {"role": "user",
                     "content": (
                         "List the 3 most recent real UAE enforcement actions against companies for "
                         "financial, fintech, crypto, or data protection violations. "
                         "Include actions by VARA, ADGM, CBUAE, DIFC, NESA, or SCA. "
                         "Return a JSON array where each object has exactly these keys: "
                         '"company", "fine_amount_usd" (number only), "violation", '
                         '"authority", "date", "details". '
                         "Return ONLY the JSON array, no markdown, no preamble."
                     )},
                ],
                pricing_table=pricing,
            )
            total_in  += inp
            total_out += out
            mc         = calc_cost(models["enforcement"], inp, out, pricing_table=pricing)
            total_cost += mc
            cost_by_model[models["enforcement"]] = {"input": inp, "output": out, "cost": mc}

            parsed = extract_json(content)
            if isinstance(parsed, list):
                enforcements = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        enforcements = v
                        break
            if not enforcements:
                errors.append(f"{enf_model_label} returned unexpected format — no enforcement list parsed.")
            yield _p("pline-ok", f"✅ {len(enforcements)} enforcement action(s) fetched.")
        except Exception as exc:
            errors.append(f"Enforcement model error: {exc}")
            yield _p("pline-err", f"⚠ Enforcement model failed: {escape(str(exc)[:300])}")

        # ── Build additional_regs context ──────────────────────────────────────
        additional_regs = ""
        for r in regulations:
            additional_regs += (
                f"- NEW: {r.get('regulation','')} ({r.get('authority','')}): "
                f"{r.get('summary','')} | Code impact: {r.get('impact_on_code','')}\n"
            )
        for e in enforcements:
            additional_regs += (
                f"- TRENDING ENFORCEMENT: {e.get('company','')} was fined "
                f"${e.get('fine_amount_usd',0):,} by {e.get('authority','')} for: "
                f"{e.get('violation','')}. Pay close attention to similar patterns.\n"
            )
        if not additional_regs.strip():
            additional_regs = "None available — use hardcoded frameworks only."

        # ── Step 4: Code audit ─────────────────────────────────────────────────
        audit_model_label = models["audit"]
        yield _p("pline-work",
                 f"⏳ Step 4/5: Auditing {num_files} file(s) "
                 f'<span class="model-pill model-pill-free">{escape(audit_model_label)}</span>…')

        audit_system = f"""You are a senior UAE financial compliance auditor and code security expert.

Audit the provided source code file against ALL of the following 10 UAE regulatory frameworks:
{fw_list}

ALSO audit against these ADDITIONAL live regulatory updates and trending enforcement patterns found today:
{additional_regs}

For every violation you find, return a JSON object with EXACTLY these keys:
- "file": the filename
- "function_name": the exact function or method name containing the violation
- "line_start": integer line number where the violation begins
- "line_end": integer line number where the violation ends
- "regulation": the specific law name and article number violated (e.g. "PDPL Art.13 — cross-border transfer")
- "violation_description": clear explanation of what the code does wrong and why it violates the regulation
- "severity": one of "critical", "high", or "medium"

Return a JSON array of all violations. If the file has no violations, return an empty array [].
Do not include any text outside the JSON array.
"""

        audit_in  = 0
        audit_out = 0

        for i, (full_path, rel_path) in enumerate(files):
            try:
                source = open(full_path, "r", errors="ignore").read()
                if not source.strip():
                    continue
                if len(source) > MAX_FILE_CHARS:
                    source = source[:MAX_FILE_CHARS] + f"\n# ... [truncated at {MAX_FILE_CHARS} chars]"

                yield _p("pline-work",
                         f"&nbsp;&nbsp;• ({i+1}/{num_files}) "
                         f'<span class="v-file">{escape(rel_path)}</span>')

                content, inp, out = openrouter_chat(
                    api_key, models["audit"],
                    [
                        {"role": "system", "content": audit_system},
                        {"role": "user",
                         "content": f"File: {rel_path}\n\n```\n{source}\n```\n\nReturn violations JSON array:"},
                    ],
                    pricing_table=pricing,
                )
                audit_in  += inp
                audit_out += out

                parsed = extract_json(content)
                if isinstance(parsed, list):
                    for v in parsed:
                        if isinstance(v, dict):
                            v.setdefault("file", rel_path)
                            all_violations.append(v)
                elif isinstance(parsed, dict) and parsed:
                    parsed.setdefault("file", rel_path)
                    all_violations.append(parsed)

            except Exception as exc:
                errors.append(f"Audit ({rel_path}): {exc}")
                yield _p("pline-err",
                         f"&nbsp;&nbsp;⚠ {escape(rel_path)}: {escape(str(exc)[:250])}")

        total_in  += audit_in
        total_out += audit_out
        ac         = calc_cost(models["audit"], audit_in, audit_out, pricing_table=pricing)
        total_cost += ac
        cost_by_model[models["audit"]] = {"input": audit_in, "output": audit_out, "cost": ac}

        crit_count = sum(1 for v in all_violations
                         if str(v.get("severity","")).lower() == "critical")
        yield _p("pline-ok",
                 f"✅ Audit complete — {len(all_violations)} violation(s) found "
                 f"({crit_count} critical).")

        # ── Step 5: Cost summary ───────────────────────────────────────────────
        if is_demo:
            cost_display = ("$0.00 — free-tier models on your key"
                            if key_source == "user"
                            else "$0.00 — shared demo key, not charged to you")
        else:
            cost_display = f"${total_cost:.4f} USD"
        yield _p("pline-info",
                 f"✅ Step 5/5: Total tokens {total_in+total_out:,} "
                 f"({total_in:,} input + {total_out:,} output) — Cost: {cost_display}")

    except Exception as exc:
        errors.append(f"Fatal: {exc}\n{traceback.format_exc()}")
        yield _p("pline-err", f"❌ Fatal error: {escape(str(exc)[:500])}")
    finally:
        if clone_dir and os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, ignore_errors=True)

    yield '</div>\n'  # close progress-box

    yield render_report(
        all_violations, enforcements, regulations,
        total_in, total_out, total_cost, cost_by_model,
        num_files, errors,
        is_demo=is_demo,
        key_source=key_source,
    )

    yield """
<script>
var pb = document.getElementById('pb');
if (pb) pb.scrollTop = pb.scrollHeight;
</script>
</div></body></html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return index_html()


@app.route("/scan", methods=["POST"])
def scan():
    """BYOK full scan — unchanged from original."""
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
    """
    Demo scan — free :free models on OpenRouter.

    Key resolution order:
      1. User typed their own key into the demo form  → use it (key_source="user")
      2. Operator set DEMO_OPENROUTER_KEY env var      → use it (key_source="operator")
      3. Neither                                       → return 400 with clear message

    In all cases:
      • Only DEMO_MODELS (:free) are used — $0 billed regardless of whose key it is
      • Only public repos, no PAT ever injected
      • File cap: MAX_DEMO_FILES
    """
    repo_url     = request.form.get("repo_url",     "").strip() or DEMO_REPO_URL
    user_key     = request.form.get("demo_api_key", "").strip()

    if not repo_url.startswith("https://"):
        return "Repository URL must start with https://", 400

    # ── Resolve which key to use ───────────────────────────────────────────────
    if user_key:
        api_key    = user_key
        key_source = "user"
    elif DEMO_OPENROUTER_KEY:
        api_key    = DEMO_OPENROUTER_KEY
        key_source = "operator"
    else:
        return (
            "No OpenRouter key available. "
            "Please enter your own OpenRouter key in the demo form, "
            "or use the full BYOK scan instead. "
            "<a href='/'>← Back</a>",
            400,
        )

    return Response(
        stream_with_context(
            stream_scan(repo_url, pat="", api_key=api_key,
                        is_demo=True, key_source=key_source)
        ),
        content_type="text/html; charset=utf-8",
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  🇦🇪 UAE Compliance Scanner")
    print(f"  http://0.0.0.0:{port}")
    if _demo_available():
        print(f"  Demo mode: ENABLED  (repo: {DEMO_REPO_URL})")
    else:
        print("  Demo mode: DISABLED (set DEMO_OPENROUTER_KEY env var to enable)")
    print()
    app.run(debug=False, host="0.0.0.0", port=port)
