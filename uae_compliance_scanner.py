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

from flask import Flask, request, Response, stream_with_context
import requests as http_requests

try:
    import git as gitpython
except ImportError:
    gitpython = None

app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

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

def openrouter_chat(api_key, model, messages):
    """Returns (content, input_tokens, output_tokens)."""
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


def calc_cost(model, inp, out):
    p = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
    return (inp * p["input"] + out * p["output"]) / 1_000_000


def find_source_files(repo_dir):
    results = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if Path(fname).suffix.lower() in TARGET_EXTENSIONS:
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, repo_dir)
                results.append((full, rel))
    return results[:MAX_FILES]


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
.note{text-align:center;color:#484f58;font-size:.78rem;margin-top:10px}

/* ── Framework badges ── */
.fw-grid{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 24px}
.fw-pill{background:#161b22;border:1px solid #30363d;border-radius:20px;
         padding:5px 12px;font-size:.75rem;color:#8b949e;cursor:default}
.fw-pill:hover{border-color:#58a6ff;color:#58a6ff}

/* ── Progress ── */
.progress-box{background:#0d1117;border:1px solid #30363d;border-radius:8px;
              padding:16px;margin-bottom:20px;max-height:380px;overflow-y:auto;
              font-family:'SFMono-Regular',Consolas,monospace}
.pline{font-size:.82rem;padding:2px 0;color:#8b949e}
.pline-ok{color:#3fb950}
.pline-err{color:#f85149}
.pline-work{color:#58a6ff}
.pline-info{color:#d29922}

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

def index_html():
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

<div class="form-card">
  <h2>🔍 Start a Compliance Scan</h2>
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
      <a href="https://openrouter.ai/keys" target="_blank">Get your key →</a>
    </p>

    <button type="submit" class="btn-scan" id="btn">🔍 Start Compliance Scan</button>
  </form>
</div>

<p class="note">🔒 Your PAT and API key are used only for this scan session and are never stored or logged.</p>

<script>
document.getElementById('sf').addEventListener('submit', function() {{
  var b = document.getElementById('btn');
  b.disabled = true;
  b.textContent = '⏳ Scanning — this may take several minutes…';
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
                  num_files, errors):
    h = []

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
    h.append('<h2 class="section">⚖ Recent UAE Enforcement Actions <span style="font-size:.75rem;color:#8b949e;font-weight:400">(via Grok — trending)</span></h2>')
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
    h.append('<h2 class="section">📜 New UAE Regulatory Updates <span style="font-size:.75rem;color:#8b949e;font-weight:400">(via Perplexity)</span></h2>')
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
    h.append(f'<div class="roi-cell"><div class="roi-cell-label">Scan Cost (You Paid)</div>'
             f'<div class="roi-cell-val roi-green">${total_cost:.4f}</div></div>')
    if lowest_fine > 0:
        h.append(f'<div class="roi-cell"><div class="roi-cell-label">Lowest Recent Fine</div>'
                 f'<div class="roi-cell-val roi-red">${lowest_fine:,.0f}</div></div>')
        h.append(f'<div class="roi-cell"><div class="roi-cell-label">Highest Recent Fine</div>'
                 f'<div class="roi-cell-val roi-red">${highest_fine:,.0f}</div></div>')
    h.append(f'<div class="roi-cell"><div class="roi-cell-label">Total Tokens</div>'
             f'<div class="roi-cell-val roi-blue">{total_in+total_out:,}</div></div>')
    h.append('</div>')  # roi-grid

    if lowest_fine > 0 and violations:
        h.append(f'<div class="roi-big">{roi_pct:,.0f}% ROI</div>')
        h.append(f'<div class="roi-sub">For every <strong>$1</strong> spent on this scan, '
                 f'you could avoid up to <strong>${per_dollar:,.0f}</strong> in regulatory fines.</div>')
    elif not violations:
        h.append('<div style="color:#3fb950;font-size:1.1rem;margin:12px 0">'
                 '✅ No violations found — your codebase appears compliant! 🎉</div>')
    else:
        h.append('<div class="roi-sub">No fine data available for ROI calculation.</div>')

    h.append('<div style="margin-top:20px;overflow-x:auto">')
    h.append('<table class="token-table">')
    h.append('<tr><th>Model</th><th>Input Tokens</th><th>Output Tokens</th>'
             '<th>Total Tokens</th><th>Cost (USD)</th></tr>')
    t_in = t_out = t_cost = 0
    for model, d in cost_by_model.items():
        mi, mo, mc = d["input"], d["output"], d["cost"]
        t_in += mi; t_out += mo; t_cost += mc
        h.append(f'<tr><td>{escape(model)}</td><td>{mi:,}</td><td>{mo:,}</td>'
                 f'<td>{mi+mo:,}</td><td>${mc:.4f}</td></tr>')
    h.append(f'<tr><td>TOTAL</td><td>{t_in:,}</td><td>{t_out:,}</td>'
             f'<td>{t_in+t_out:,}</td><td>${t_cost:.4f}</td></tr>')
    h.append('</table>')
    h.append('</div>')  # overflow wrapper
    h.append('</div>')  # roi-wrap

    # ── Errors ─────────────────────────────────────────────────────────────────
    if errors:
        h.append('<h2 class="section">⚠ Warnings &amp; Errors</h2>')
        for err in errors:
            h.append(f'<div class="err-box">{escape(str(err)[:500])}</div>')

    # ── Footer ─────────────────────────────────────────────────────────────────
    h.append("""
<div class="site-footer">
  <p>Scanned against 10 UAE regulatory frameworks + live Grok enforcement trends + Perplexity regulatory updates.</p>
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

def stream_scan(repo_url, pat, api_key):
    yield f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scanning… — UAE Compliance Scanner</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
{FLUSH_PAD}
<div class="site-header">
  <span style="font-size:2.2rem">🇦🇪</span>
  <h1>UAE Compliance Scanner</h1>
</div>
<p class="tagline">Scan in progress…</p>
<div class="progress-box" id="pb">
"""

    clone_dir     = None
    total_in      = 0
    total_out     = 0
    total_cost    = 0.0
    cost_by_model = {}
    all_violations = []
    regulations   = []
    enforcements  = []
    errors        = []
    num_files     = 0

    fw_list = "\n".join(
        f"{n}. {name} — {desc}"
        for n, name, desc in UAE_FRAMEWORKS
    )

    try:
        # ── Step 1: Clone ──────────────────────────────────────────────────────
        yield _p("pline-work", "⏳ Step 1/5: Cloning repository…")
        if gitpython is None:
            raise RuntimeError("GitPython not installed. Run: pip install gitpython")

        clone_dir = tempfile.mkdtemp(prefix="uae_scan_")
        auth_url  = (repo_url.replace("https://", f"https://x-access-token:{pat}@")
                     if pat else repo_url)
        try:
            gitpython.Repo.clone_from(auth_url, clone_dir, depth=1)
        except Exception as exc:
            raise RuntimeError(f"Clone failed: {exc}")

        files     = find_source_files(clone_dir)
        num_files = len(files)
        cap_note  = " (capped at 50)" if num_files >= MAX_FILES else ""
        yield _p("pline-ok", f"✅ Cloned. {num_files} scannable file(s) found{cap_note}.")

        if num_files == 0:
            yield _p("pline-err", "⚠ No .py / .js / .ts / .sol files found.")

        # ── Step 2: Perplexity Deep Research — new regulations ─────────────────
        yield _p("pline-work", "⏳ Step 2/5: Fetching new UAE regulatory updates (Perplexity Sonar Deep Research)…")
        try:
            content, inp, out = openrouter_chat(api_key, MODELS["regulations"], [
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
            ])
            total_in  += inp
            total_out += out
            mc         = calc_cost(MODELS["regulations"], inp, out)
            total_cost += mc
            cost_by_model[MODELS["regulations"]] = {"input": inp, "output": out, "cost": mc}

            parsed = extract_json(content)
            if isinstance(parsed, list):
                regulations = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        regulations = v
                        break
            if not regulations:
                errors.append("Perplexity returned unexpected format — no regulation list parsed.")
            yield _p("pline-ok", f"✅ {len(regulations)} new regulatory update(s) fetched.")
        except Exception as exc:
            errors.append(f"Perplexity error: {exc}")
            yield _p("pline-err", f"⚠ Perplexity failed: {escape(str(exc)[:300])}")

        # ── Step 3: Grok 4 Multi-Agent — trending enforcement ─────────────────
        yield _p("pline-work", "⏳ Step 3/5: Fetching trending UAE enforcement actions (Grok 4 Multi-Agent)…")
        try:
            content, inp, out = openrouter_chat(api_key, MODELS["enforcement"], [
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
            ])
            total_in  += inp
            total_out += out
            mc         = calc_cost(MODELS["enforcement"], inp, out)
            total_cost += mc
            cost_by_model[MODELS["enforcement"]] = {"input": inp, "output": out, "cost": mc}

            parsed = extract_json(content)
            if isinstance(parsed, list):
                enforcements = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        enforcements = v
                        break
            if not enforcements:
                errors.append("Grok returned unexpected format — no enforcement list parsed.")
            yield _p("pline-ok", f"✅ {len(enforcements)} enforcement action(s) fetched.")
        except Exception as exc:
            errors.append(f"Grok error: {exc}")
            yield _p("pline-err", f"⚠ Grok failed: {escape(str(exc)[:300])}")

        # ── Build additional_regs from Perplexity + Grok for Claude ────────────
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

        # ── Step 4: Claude Sonnet 4.6 — code audit ────────────────────────────
        yield _p("pline-work",
                 f"⏳ Step 4/5: Auditing {num_files} file(s) against "
                 f"10 UAE laws + trending + new regulations (Claude Sonnet 4.6)…")

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

        claude_in  = 0
        claude_out = 0

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

                content, inp, out = openrouter_chat(api_key, MODELS["audit"], [
                    {"role": "system", "content": audit_system},
                    {"role": "user",
                     "content": f"File: {rel_path}\n\n```\n{source}\n```\n\nReturn violations JSON array:"},
                ])
                claude_in  += inp
                claude_out += out

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
                errors.append(f"Claude audit ({rel_path}): {exc}")
                yield _p("pline-err",
                         f"&nbsp;&nbsp;⚠ {escape(rel_path)}: {escape(str(exc)[:250])}")

        total_in  += claude_in
        total_out += claude_out
        cc         = calc_cost(MODELS["audit"], claude_in, claude_out)
        total_cost += cc
        cost_by_model[MODELS["audit"]] = {"input": claude_in, "output": claude_out, "cost": cc}

        crit_count = sum(1 for v in all_violations
                         if str(v.get("severity","")).lower() == "critical")
        yield _p("pline-ok",
                 f"✅ Audit complete — {len(all_violations)} violation(s) found "
                 f"({crit_count} critical).")

        # ── Step 5: Cost summary ───────────────────────────────────────────────
        yield _p("pline-info",
                 f"✅ Step 5/5: Total tokens {total_in+total_out:,} "
                 f"({total_in:,} input + {total_out:,} output) — "
                 f"Total cost ${total_cost:.4f} USD")

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
        num_files, errors
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
        stream_with_context(stream_scan(repo_url, pat, api_key)),
        content_type="text/html; charset=utf-8",
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  🇦🇪 UAE Compliance Scanner")
    print(f"  http://0.0.0.0:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
