"""
Bitcast X pre-submission validator — FastAPI backend.

Lets a creator paste a draft tweet + pick an active campaign brief and get an
instant pass/fail verdict, replicating the same LLM evaluation logic and
optimistic multi-check strategy that real Bitcast validators use.
"""

import hashlib
import html
import json
import logging
import os
import secrets
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from activity_log import hash_ip, log_event, read_events
from engagement import router as engagement_router
from engagement import warm_avatars as engagement_warm_avatars
from engagement import warm_cache as engagement_warm_cache
from legacy_briefs import get_cached_legacy_briefs
from prompts import generate_brief_evaluation_prompt

load_dotenv()

BITCAST_CAMPAIGN_MANIFEST_ENDPOINT = "https://bitcast-api.bitcast.network/api/v2/public/x/campaign-manifest-v4"
BRAND_OVERVIEW_BASE_URL = "https://brand-overviews-x.s3.us-west-2.amazonaws.com"
CHUTES_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
NUM_LLM_CHECKS = 3
MODEL = "Qwen/Qwen3-32B"

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
_admin_security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(_admin_security)) -> None:
    # ADMIN_PASSWORD unset means the panel was never configured on this
    # deployment -- refuse rather than silently falling back to some
    # default, which would leave it wide open.
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin panel is not configured.")
    # compare_digest for both, not just the password -- constant-time
    # comparison so response timing can't be used to guess either field.
    valid_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    valid_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

LOGGER = logging.getLogger("stitch3_validator")

app = FastAPI(title="Stitch3 Validator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(engagement_router, prefix="/api/engagement")


@app.on_event("startup")
async def _warm_engagement_cache_on_startup() -> None:
    # engagement.py's ecosystem-map cache has no TTL (see fetch_ecosystem_map),
    # so the cold-cache network round-trip to Bitcast's API only ever needs to
    # happen once per process lifetime -- do it here, at boot, instead of
    # letting it land on whichever real visitor opens the Engagement Value
    # tab first after a deploy/restart. Fire-and-forget: don't hold up the
    # server actually starting to accept requests, and a failure here is
    # harmless since the normal request-time fetch is still the fallback.
    import asyncio

    asyncio.create_task(engagement_warm_cache())

# /evaluate and /evaluate/stream are the only endpoints that spend real money
# (each check is a Chutes API call against our own key) -- /briefs and the
# brand-overview lookups are cheap/cached and don't need this. In-memory,
# per-process fixed-window limiter: fine for this single-instance deployment,
# same tradeoff as the existing _briefs_cache/_overview_cache dicts below.
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 5  # per IP, per window
_rate_limit_hits: dict[str, list[float]] = {}
_rate_limit_lock = Lock()


def _client_ip(request: Request) -> str:
    # Behind nginx, request.client.host is the proxy's own address unless
    # X-Forwarded-For is set -- prefer that when present.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    with _rate_limit_lock:
        hits = [t for t in _rate_limit_hits.get(ip, []) if t > cutoff]
        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(hits[0] - cutoff) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many checks in a short time — please wait a bit before trying again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        _rate_limit_hits[ip] = hits


class EvaluateRequest(BaseModel):
    brief_id: str
    tweet: str


class CheckResult(BaseModel):
    verdict: str
    summary: str
    raw_response: str


class EvaluateResponse(BaseModel):
    meets_brief: bool
    checks: list[CheckResult]
    prompt_version: int
    brief_display: str


def parse_verdict(text: str) -> tuple[str, str]:
    """Extract the YES/NO verdict and one-sentence summary from a model response."""
    verdict = "NO"
    summary = ""

    if "## Verdict" in text:
        after_verdict = text.split("## Verdict", 1)[1]
        verdict_line = after_verdict.strip().splitlines()[0].strip()
        if "YES" in verdict_line.upper():
            verdict = "YES"

    if "## Summary" in text:
        summary = text.split("## Summary", 1)[1].strip()
        if summary.endswith("```"):
            summary = summary[:-3].strip()

    return verdict, summary


# How congested Chutes' shared compute currently is, inferred from call
# latencies within the *current evaluation only* -- Chutes has no public
# per-model load/utilization API we can query (checked their docs: the only
# utilization endpoints are miner-side and need miner auth), so this is
# self-measured from real traffic rather than borrowed from an official
# metric. Deliberately scoped per-request (a plain list threaded through the
# call chain) rather than a shared global window: a process-wide window mixes
# in stale samples from other visitors' evaluations and from attempts made
# minutes ago, which showed up as a misleading "slower than usual" badge
# during a run that was actually fast throughout. Records every individual
# HTTP attempt's wall time (not counting backoff sleeps), so a timeout
# naturally shows up as a ~60s sample -- exactly the signal we want.
def chutes_congestion_state(samples: list[float]) -> dict:
    """4-state read on Chutes responsiveness so far in this evaluation, based
    on the average of its own call attempts. Thresholds are calibrated
    around this project's own documented baseline (~30-45s is normal for a
    single call on Chutes' shared/decentralized compute, not fast) -- not
    generic web-latency assumptions."""
    if len(samples) < 1:
        return {"state": "unknown", "avg_latency": None, "samples": len(samples)}

    avg = sum(samples) / len(samples)
    if avg < 15:
        state = "fast"
    elif avg < 35:
        state = "normal"
    elif avg < 55:
        state = "slow"
    else:
        state = "congested"
    return {"state": state, "avg_latency": round(avg, 1), "samples": len(samples)}


def call_chutes(prompt: str, latencies: list[float]) -> str:
    """Call Chutes' chat completions endpoint, matching the real validator's
    ChuteClient. `latencies` is the calling evaluation's own list (see above)
    -- list.append() is atomic under the GIL, so concurrent checks sharing it
    from separate threads need no extra lock."""
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4096,
    }

    # A 429 means Chutes is actively telling us to slow down -- retrying in
    # 1-2s (the normal backoff, meant for transient timeouts/connection
    # blips) just gets rejected again. Back off longer specifically for
    # that case. No Retry-After handling yet -- unconfirmed whether Chutes
    # sends that header; revisit if/when that's checked against a real 429.
    RATE_LIMIT_BACKOFF_SECONDS = 8

    last_error = None
    for attempt in range(3):
        start = time.time()
        try:
            resp = requests.post(CHUTES_ENDPOINT, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            latencies.append(time.time() - start)
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            latencies.append(time.time() - start)
            last_error = e
            if attempt < 2:
                is_rate_limited = (
                    isinstance(e, requests.HTTPError)
                    and e.response is not None
                    and e.response.status_code == 429
                )
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS if is_rate_limited else 2 ** attempt)
    raise last_error


def run_single_check(
    brief: dict, tweet: str, prompt_version: int, check_num: int, latencies: list[float]
) -> CheckResult:
    # Real validator appends " {check_num}" to bust its LLM cache so each of the
    # NUM_LLM_CHECKS runs is an independent judgment rather than a repeat of the
    # same deterministic (temperature=0) response. Replicated here for the same
    # reason: without it, our "best of 3" carries far less real variance.
    variant_tweet = f"{tweet} {check_num}"
    prompt = generate_brief_evaluation_prompt(brief, variant_tweet, version=prompt_version)
    text = call_chutes(prompt, latencies)
    verdict, summary = parse_verdict(text)
    return CheckResult(verdict=verdict, summary=summary, raw_response=text)


def run_checks_pessimistic(brief: dict, tweet: str, prompt_version: int):
    """Yield (CheckResult, chutes_state) as each check completes, stopping at
    the first NO. chutes_state reflects only this evaluation's own call
    attempts so far (see chutes_congestion_state above), not other traffic.

    Deliberately stricter than the real validator's own optimistic
    (any-YES-wins) best-of-3: a single NO here already means meets_brief
    can never be True, so there's no reason to reduce this to a numeric
    score -- the caller just needs every yielded check to be YES, and
    needs to see all NUM_LLM_CHECKS of them, for the tweet to pass. This
    makes our checker harder to satisfy than the real validator, on
    purpose -- see project notes for why (false positives here are far
    more costly to a miner than false negatives).
    """
    latencies: list[float] = []
    executor = ThreadPoolExecutor(max_workers=NUM_LLM_CHECKS)
    futures = [
        executor.submit(run_single_check, brief, tweet, prompt_version, check_num, latencies)
        for check_num in range(1, NUM_LLM_CHECKS + 1)
    ]
    try:
        for future in as_completed(futures):
            result = future.result()
            yield result, chutes_congestion_state(latencies)
            if result.verdict != "YES":
                break
    finally:
        executor.shutdown(wait=False)


def _fetch_normalized_briefs() -> list[dict]:
    """Fetch the live campaign manifest and flatten each entry into the flat
    shape the rest of this app expects (id/pool/start_date/end_date/display/
    brief/tag/prompt_version) -- the manifest nests campaign_id under
    "access", uses a "pools" array (always length 1 in practice) instead of
    a single "pool" string, and "opens_at"/"closes_at" full timestamps
    instead of date-only "start_date"/"end_date" strings."""
    try:
        resp = requests.get(BITCAST_CAMPAIGN_MANIFEST_ENDPOINT, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        # Bitcast's own API (not ours) -- surface this distinctly from a
        # real bug here, since retrying in a few minutes is the actual fix,
        # not something wrong with this tool. Confirmed live 2026-08-19:
        # their ELB returned a fast 503 with no healthy backend target.
        raise HTTPException(
            status_code=502,
            detail="Bitcast's campaign API is temporarily unavailable. Please try again in a few minutes.",
        )
    campaigns = resp.json().get("campaigns", [])
    briefs = []
    for c in campaigns:
        # "indie_hacker" is a brand-new pool that only ever had one campaign
        # (078_stitch3, using the new preclaim_v2 mining_protocol, a 1-day
        # window, and asking creators to review Stitch3 itself) -- it isn't
        # visible on Bitcast's own website, so it reads as an internal pilot
        # for the new protocol rather than a real public campaign. Hold this
        # pool back until it's confirmed live there; remove this filter once
        # it is.
        if "indie_hacker" in (c.get("pools") or []):
            continue
        access = c.get("access", {})
        briefs.append({
            "id": access.get("campaign_id"),
            "pool": (c.get("pools") or [None])[0],
            "start_date": (c.get("opens_at") or "")[:10],
            "end_date": (c.get("closes_at") or "")[:10],
            "display": c.get("display", ""),
            "brief": c.get("brief", ""),
            "tag": c.get("tag"),
            "prompt_version": c.get("prompt_version", 1),
            # Not consumed by the frontend today -- kept in case exclusive
            # campaigns (only one specific miner hotkey may submit) ever
            # need to be filtered out or flagged in the UI.
            "exclusive_miner_hotkey": access.get("exclusive_miner_hotkey"),
        })
    return briefs


def _lookup_brief(brief_id: str) -> dict:
    briefs = _get_cached_briefs()
    brief = next((b for b in briefs if b["id"] == brief_id), None)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{brief_id}' not found")
    return brief


def has_brand_overview(brief_id: str) -> bool:
    url = f"{BRAND_OVERVIEW_BASE_URL}/{brief_id}.pdf"
    resp = requests.head(url, timeout=10)
    return resp.status_code == 200


BRIEFS_CACHE_TTL = 300  # seconds
_briefs_cache = {"data": None, "expires_at": 0.0}
_overview_cache = {"data": None, "expires_at": 0.0}

STATS_FILE = Path(__file__).parent / "stats.json"
STATS_SINCE = "2026-08-10"
_stats_lock = Lock()


def _hash_ip(ip: str) -> str:
    # Store a salted-ish hash, never the raw IP -- stats.json holding real
    # visitor IPs would be a real privacy liability if it ever leaked or got
    # committed by mistake. We only ever need to dedupe, not identify.
    return hashlib.sha256(f"stitch3-validator:{ip}".encode()).hexdigest()


def _load_stats_raw() -> dict:
    if STATS_FILE.exists():
        try:
            data = json.loads(STATS_FILE.read_text())
            data.setdefault("tweets_checked", 0)
            data.setdefault("fails_caught", 0)
            data.setdefault("creator_ip_hashes", [])
            data.setdefault("since", STATS_SINCE)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"tweets_checked": 0, "fails_caught": 0, "creator_ip_hashes": [], "since": STATS_SINCE}


def _load_stats_public() -> dict:
    """What /stats actually returns -- the hash list itself never leaves
    the server, only its count, since it's still one-way-linkable to
    "did this specific visitor use the tool" even though it's not a raw IP."""
    stats = _load_stats_raw()
    return {
        "tweets_checked": stats["tweets_checked"],
        "fails_caught": stats["fails_caught"],
        "unique_creators": len(stats["creator_ip_hashes"]),
        "since": stats["since"],
    }


def _record_check(ip: str, meets_brief: bool) -> None:
    """File-backed counters (no database in this project) so they survive
    restarts/redeploys, unlike the in-memory caches above. Counts a
    completed evaluation (checks actually ran to a result), not merely an
    attempted request -- rate-limited or errored calls don't count.
    fails_caught has no historical backfill (unlike tweets_checked/
    unique_creators) -- nginx access logs only have HTTP status, not the
    verdict, so there was no way to reconstruct it for checks run before
    this counter existed."""
    with _stats_lock:
        stats = _load_stats_raw()
        stats["tweets_checked"] += 1
        if not meets_brief:
            stats["fails_caught"] += 1
        ip_hash = _hash_ip(ip)
        if ip_hash not in stats["creator_ip_hashes"]:
            stats["creator_ip_hashes"].append(ip_hash)
        STATS_FILE.write_text(json.dumps(stats))


def _get_cached_briefs() -> list[dict]:
    """Fast path: brief list only, no S3 checks. Shared by /briefs and the
    server-rendered index page, both backed by the same 5-minute cache.

    Merges the live manifest with the legacy archive (see
    legacy_briefs.get_cached_legacy_briefs), live-manifest entries taking
    precedence on an id collision (070_verathos through 074_nodexo appear
    in both -- the live manifest's copy is the fresher one). This is the
    one list every consumer reads (this endpoint's campaign selector,
    _lookup_brief for /evaluate and /evaluate/stream, and
    /briefs/brand-overviews' HEAD-check fan-out below), so a legacy
    campaign works exactly like a live one everywhere in this app -- no
    separate "legacy brief" code path to keep in sync."""
    now = time.time()
    if _briefs_cache["data"] is not None and now < _briefs_cache["expires_at"]:
        return _briefs_cache["data"]

    live_items = _fetch_normalized_briefs()
    live_ids = {b["id"] for b in live_items}
    legacy_items = [b for b in get_cached_legacy_briefs() if b["id"] not in live_ids]
    items = live_items + legacy_items

    _briefs_cache["data"] = items
    _briefs_cache["expires_at"] = now + BRIEFS_CACHE_TTL
    return items


@app.get("/briefs")
def get_briefs():
    """has_brand_overview is filled in client-side after
    /briefs/brand-overviews resolves, so this never blocks initial page
    render on the ~1.5-3s S3 HEAD-check fan-out."""
    return _get_cached_briefs()


@app.get("/briefs/brand-overviews")
def get_brand_overviews():
    now = time.time()
    if _overview_cache["data"] is not None and now < _overview_cache["expires_at"]:
        return _overview_cache["data"]

    items = _get_cached_briefs()
    ids = [b["id"] for b in items]

    with ThreadPoolExecutor(max_workers=len(ids) or 1) as executor:
        has_overview = executor.map(has_brand_overview, ids)
    result = dict(zip(ids, has_overview))

    _overview_cache["data"] = result
    _overview_cache["expires_at"] = now + BRIEFS_CACHE_TTL
    return result


@app.get("/stats")
def get_stats():
    return _load_stats_public()


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def _admin_badge(ok: bool, ok_label: str, bad_label: str) -> str:
    cls = "status-active" if ok else "status-completed"
    label = ok_label if ok else bad_label
    return f'<span class="status-badge {cls}"><span class="status-dot"></span>{html.escape(label)}</span>'


ADMIN_POOL_LABELS = {"tao": "Bittensor", "hyperliquid": "Perp DEXs", "prediction_markets": "Prediction Markets"}


def _admin_rank_rows(counts: list[tuple[str, int]]) -> str:
    if not counts:
        return '<tr class="empty-row"><td colspan="3">Nothing logged yet.</td></tr>'
    return "".join(
        f"<tr><td class=\"col-rank\">#{i}</td>"
        f"<td class=\"col-mono\">{html.escape(str(label))}</td>"
        f"<td class=\"col-count\">{count}</td></tr>"
        for i, (label, count) in enumerate(counts, start=1)
    )


@app.post("/admin/refresh-avatars", dependencies=[Depends(require_admin)])
async def admin_refresh_avatars():
    """Manually (re-)warms the Engagement Value avatar disk store -- see
    engagement.warm_avatars's own docstring. There's deliberately no
    automatic recurring refresh (removed by request); this is the only way
    it happens after the initial one at process startup. Redirects back to
    /admin with the result in the query string so the page can show what
    happened without any client-side JS."""
    stats = await engagement_warm_avatars()
    params = "&".join(f"avatar_{k}={v}" for k, v in stats.items())
    return RedirectResponse(url=f"/admin?avatar_refreshed=1&{params}", status_code=303)


@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
def admin_panel(
    avatar_refreshed: bool = False,
    avatar_considered: int = 0,
    avatar_already_fresh: int = 0,
    avatar_fetched: int = 0,
    avatar_failed: int = 0,
    avatar_budget_exhausted: bool = False,
):
    """Owner-only view of raw usage: every engagement-value handle lookup
    and every tweet check, newest first. Gated by HTTP Basic Auth
    (require_admin) -- not linked from anywhere in the public UI. Styled to
    match the public site's own design tokens (frontend/index.html's :root
    block) rather than a bare utility page, since this is still a page the
    owner will look at regularly."""
    avatar_toast_html = ""
    if avatar_refreshed:
        exhausted_note = " Daily fetch budget ran out partway through." if avatar_budget_exhausted else ""
        avatar_toast_html = (
            f'<div class="admin-toast">Avatar refresh: <strong>{avatar_fetched}</strong> fetched, '
            f"{avatar_already_fresh} already fresh, {avatar_failed} failed, "
            f"out of {avatar_considered} considered accounts checked.{exhausted_note}</div>"
        )

    events = read_events(limit=500)
    lookups = [e for e in events if e.get("type") == "engagement_lookup"]
    checks = [e for e in events if e.get("type") == "tweet_check"]

    top_handles = Counter(f"@{e['handle']}" for e in lookups if e.get("handle")).most_common(10)
    top_campaigns = Counter(e["brief_id"] for e in checks if e.get("brief_id")).most_common(10)

    ecosystem_counts = Counter(e["ecosystem_id"] for e in lookups if e.get("ecosystem_id"))
    ecosystem_total = sum(ecosystem_counts.values())
    ecosystem_rows = "".join(
        f"<tr><td class=\"col-mono\">{html.escape(ADMIN_POOL_LABELS.get(eco, eco))}</td>"
        f"<td class=\"col-count\">{count}</td>"
        f"<td class=\"col-count\">{round(count / ecosystem_total * 100)}%</td></tr>"
        for eco, count in ecosystem_counts.most_common()
    ) or '<tr class="empty-row"><td colspan="3">No lookups logged yet.</td></tr>'

    lookup_rows = "".join(
        f"<tr><td class=\"col-time\">{html.escape(_fmt_ts(e.get('ts', 0)))}</td>"
        f"<td class=\"col-mono\">@{html.escape(str(e.get('handle', '')))}</td>"
        f"<td class=\"col-mono\">{html.escape(str(e.get('campaign_id', '')))}</td>"
        f"<td>{html.escape(str(e.get('ecosystem_id', '')))}</td>"
        f"<td>{_admin_badge(bool(e.get('found')), 'Found', 'Not found')}</td>"
        f"<td class=\"col-ip\">{html.escape(str(e.get('ip_hash', ''))[:12])}</td></tr>"
        for e in lookups
    )
    check_rows = "".join(
        f"<tr><td class=\"col-time\">{html.escape(_fmt_ts(e.get('ts', 0)))}</td>"
        f"<td class=\"col-mono\">{html.escape(str(e.get('brief_id', '')))}</td>"
        f"<td>{_admin_badge(e.get('verdict') == 'YES', 'Pass', 'Fail')}</td>"
        f"<td class=\"col-ip\">{html.escape(str(e.get('ip_hash', ''))[:12])}</td></tr>"
        for e in checks
    )

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stitch3 Validator — Admin</title>
<link rel="icon" type="image/png" href="/assets/favicon-new.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@700,500&display=swap" rel="stylesheet">
<style>
  :root {{
    color-scheme: dark;
    --black-0: #050507;
    --black-1: #0a0a0c;
    --black-2: #131317;
    --gray-100: #f1f5f9;
    --gray-300: #d1d5db;
    --gray-400: #9ca3af;
    --gray-500: #6c7280;
    --purple: #b882ff;
    --purple-light: #cba3ff;
    --gradient-brand: linear-gradient(105deg, #ded2d7 0%, #b882ff 100%);
    --green: #34d399;
    --red: #f87171;
    --border-card: #ffffff1a;
    --border-input: #ffffff1f;
    --border-accent: #b882ff59;
    --radius-control: 8px;
    --radius-pill: 999px;
    --font-display: "Satoshi", "Inter", sans-serif;
    --font-body: "Inter", -apple-system, sans-serif;
    --font-mono: "JetBrains Mono", monospace;
    --tracking-label: .14em;
  }}
  * {{ box-sizing: border-box; }}
  html {{ overflow-x: hidden; }}
  body {{
    background: var(--black-0);
    color: var(--gray-400);
    font-family: var(--font-body);
    line-height: 1.5;
    margin: 0;
    padding: 48px 20px 64px;
    position: relative;
    overflow-x: hidden;
  }}
  /* Same hero glow asset/technique as the public site's body::before, so
     this doesn't read as a bare admin utility page bolted onto the side. */
  body::before {{
    content: "";
    position: absolute;
    top: -160px;
    left: 50%;
    width: max(1400px, 120vw);
    max-width: 220vw;
    height: 1000px;
    transform: translateX(-50%);
    background-image: url("/assets/hero-background.png");
    background-size: cover;
    background-position: center top;
    -webkit-mask-image: linear-gradient(black 55%, transparent 98%);
    mask-image: linear-gradient(black 55%, transparent 98%);
    opacity: 0.8;
    pointer-events: none;
    z-index: 0;
  }}
  .page {{ max-width: 920px; margin: 0 auto; position: relative; z-index: 1; animation: glide-in 1.2s ease-out; }}
  @keyframes glide-in {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .admin-header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 32px; }}
  .admin-logo {{
    height: 34px; width: 48px; flex-shrink: 0;
    background-color: var(--gray-100);
    -webkit-mask-image: url("/assets/logo-icon-only.png");
    mask-image: url("/assets/logo-icon-only.png");
    -webkit-mask-size: contain; mask-size: contain;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
  }}
  .kicker {{ font-family: var(--font-display); font-weight: 700; font-size: 22px; color: var(--gray-100); letter-spacing: -0.01em; display: block; }}
  .disclaimer {{ font-size: 12.5px; color: var(--gray-500); }}
  .card {{
    position: relative;
    background: radial-gradient(ellipse 120% 100% at 50% 0%, #b882ff14 0%, transparent 60%), var(--black-1);
    border: 1px solid var(--border-accent);
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 8px 48px #0009, 0 0 64px #b882ff14;
    margin-bottom: 24px;
  }}
  .card h2 {{
    margin: 0 0 4px;
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 700;
    color: var(--gray-100);
  }}
  .card .count {{ font-size: 12px; color: var(--gray-500); margin-bottom: 16px; }}
  .card .count strong {{ font-family: var(--font-mono); color: var(--purple-light); font-weight: 600; }}
  td.col-rank {{ font-family: var(--font-mono); color: var(--purple-light); }}
  td.col-count {{ font-family: var(--font-mono); color: var(--gray-100); }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border-input); }}
  th {{
    font-family: var(--font-mono);
    font-size: 10.5px;
    letter-spacing: var(--tracking-label);
    text-transform: uppercase;
    color: var(--gray-500);
    font-weight: 500;
  }}
  td {{ color: var(--gray-300); }}
  td.col-time {{ color: var(--gray-500); font-size: 12px; }}
  td.col-mono {{ font-family: var(--font-mono); color: var(--gray-100); }}
  td.col-ip {{ font-family: var(--font-mono); color: var(--gray-500); font-size: 11.5px; }}
  .empty-row td {{ color: var(--gray-500); font-style: italic; }}
  .status-badge {{
    display: inline-flex; align-items: center; gap: 7px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 500;
    letter-spacing: var(--tracking-label); text-transform: uppercase;
  }}
  .status-dot {{ width: 7px; height: 7px; border-radius: 50%; }}
  .status-active {{ color: var(--green); }}
  .status-active .status-dot {{ background: var(--green); box-shadow: 0 0 6px #34d399aa; }}
  .status-completed {{ color: var(--red); }}
  .status-completed .status-dot {{ background: var(--red); box-shadow: 0 0 6px #f87171aa; }}
  .admin-btn {{
    font-family: var(--font-body); font-size: 13px; font-weight: 600;
    padding: 9px 16px; border: none; border-radius: var(--radius-control);
    background: var(--gradient-brand); color: #050507; cursor: pointer;
  }}
  .admin-btn:hover {{ filter: brightness(1.05); }}
  .admin-toast {{
    font-size: 13px; color: var(--gray-300); background: var(--black-2);
    border: 1px solid var(--border-card); border-radius: var(--radius-control);
    padding: 12px 16px; margin-bottom: 24px;
  }}
</style></head>
<body>
  <div class="page">
    <div class="admin-header">
      <div class="admin-logo" role="img" aria-label="Stitch3 Validator"></div>
      <div>
        <span class="kicker">Admin</span>
        <span class="disclaimer">Owner-only usage log. Not linked from the public site.</span>
      </div>
    </div>

    {avatar_toast_html}
    <div class="card">
      <h2>Engagement Value avatars</h2>
      <div class="count">Fetches up to 20 new accounts' avatars to disk each run, highest-influence first, until every considered account eventually has one (see engagement.py). No automatic schedule; run this manually whenever coverage needs a top-up.</div>
      <form method="post" action="/admin/refresh-avatars">
        <button type="submit" class="admin-btn">Refresh avatars</button>
      </form>
    </div>

    <div class="card">
      <h2>Ecosystem breakdown</h2>
      <div class="count">Share of engagement lookups by ecosystem</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Ecosystem</th><th>Lookups</th><th>Share</th></tr></thead>
          <tbody>{ecosystem_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Most looked-up handles</h2>
      <div class="count">Top 10 by lookup count</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Handle</th><th>Lookups</th></tr></thead>
          <tbody>{_admin_rank_rows(top_handles)}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Most-checked campaigns</h2>
      <div class="count">Top 10 by check count</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Campaign</th><th>Checks</th></tr></thead>
          <tbody>{_admin_rank_rows(top_campaigns)}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Engagement Value lookups</h2>
      <div class="count"><strong>{len(lookups)}</strong> shown, most recent first</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>Handle</th><th>Campaign</th><th>Ecosystem</th><th>Status</th><th>IP hash</th></tr></thead>
          <tbody>
            {lookup_rows or '<tr class="empty-row"><td colspan="6">No lookups logged yet.</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Tweet checks</h2>
      <div class="count"><strong>{len(checks)}</strong> shown, most recent first</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>Brief</th><th>Verdict</th><th>IP hash</th></tr></thead>
          <tbody>
            {check_rows or '<tr class="empty-row"><td colspan="4">No checks logged yet.</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</body></html>"""
    return HTMLResponse(content=page)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, request: Request):
    _enforce_rate_limit(request)
    brief = _lookup_brief(req.brief_id)
    prompt_version = brief.get("prompt_version", 1)

    try:
        checks = [result for result, _ in run_checks_pessimistic(brief, req.tweet, prompt_version)]
    except Exception:
        # call_chutes() already retries 3x internally -- reaching here means
        # Chutes itself is timing out/erroring past that budget. Surface a
        # real explanation instead of letting the raw exception 500 out.
        # Log it too -- this used to be swallowed with zero server-side
        # trace, making a real Chutes-side failure indistinguishable from
        # "never happened" when checking journalctl after the fact.
        LOGGER.exception("evaluate() failed for brief=%s", req.brief_id)
        raise HTTPException(
            status_code=502,
            detail="The evaluation service is currently slow or unavailable. Please try again in a moment.",
        )
    meets_brief = len(checks) == NUM_LLM_CHECKS and all(c.verdict == "YES" for c in checks)
    _record_check(_client_ip(request), meets_brief)
    log_event(
        "tweet_check",
        brief_id=req.brief_id,
        verdict="YES" if meets_brief else "NO",
        ip_hash=hash_ip(_client_ip(request)),
    )

    return EvaluateResponse(
        meets_brief=meets_brief,
        checks=checks,
        prompt_version=prompt_version,
        brief_display=brief.get("display", brief.get("brief", "")),
    )


@app.post("/evaluate/stream")
def evaluate_stream(req: EvaluateRequest, request: Request):
    """SSE variant of /evaluate: emits a "check" event as each of the (up to
    NUM_LLM_CHECKS) checks completes, then a final "done" event with the
    same shape as EvaluateResponse. Lets the frontend show real per-check
    progress during the pessimistic (must-pass-all-3) wait instead of a
    static "please wait" message."""
    _enforce_rate_limit(request)
    brief = _lookup_brief(req.brief_id)
    prompt_version = brief.get("prompt_version", 1)

    def event_stream():
        checks: list[CheckResult] = []
        try:
            for result, chutes_state in run_checks_pessimistic(brief, req.tweet, prompt_version):
                checks.append(result)
                payload = {
                    "type": "check",
                    "index": len(checks),
                    "verdict": result.verdict,
                    "chutes_state": chutes_state["state"],
                    "chutes_avg_latency": chutes_state["avg_latency"],
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception:
            # SSE headers are already sent by this point, so a real error
            # can't be raised as an HTTP status -- emit it as its own event
            # instead. call_chutes() already retries 3x internally, so
            # reaching here means Chutes itself is timing out/erroring past
            # that budget, not a transient blip. Without this, the raw
            # exception used to crash the stream silently and the frontend
            # fell back to a generic "took too long" message that hid the
            # real cause. Also log it -- uvicorn's access log shows this
            # request as a plain "200 OK" either way (the SSE stream itself
            # succeeded even though a check inside it failed), so without an
            # explicit log line here there was zero server-side trace of
            # which requests actually failed internally.
            LOGGER.exception("evaluate_stream() failed for brief=%s", req.brief_id)
            error_payload = {
                "type": "error",
                "detail": "The evaluation service is currently slow or unavailable. Please try again in a moment.",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            return

        meets_brief = len(checks) == NUM_LLM_CHECKS and all(c.verdict == "YES" for c in checks)
        _record_check(_client_ip(request), meets_brief)
        log_event(
            "tweet_check",
            brief_id=req.brief_id,
            verdict="YES" if meets_brief else "NO",
            ip_hash=hash_ip(_client_ip(request)),
        )
        final = {
            "type": "done",
            "meets_brief": meets_brief,
            "checks": [c.model_dump() for c in checks],
            "prompt_version": prompt_version,
            "brief_display": brief.get("display", brief.get("brief", "")),
        }
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx not to buffer the response before forwarding it --
            # without this, a reverse proxy can hold the whole SSE stream
            # until the generator finishes, so the frontend receives every
            # "check" event and the final "done" event in one burst instead
            # of progressively. (Still requires `proxy_buffering off;` in
            # the actual nginx config on the VPS for this header to take
            # effect there -- see deploy notes.)
            "X-Accel-Buffering": "no",
        },
    )


FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve index.html with the current briefs embedded directly in the
    response, so a first-time (uncached-browser) visitor's very first paint
    already has the data -- no separate client-side /briefs round-trip
    needed before the campaign selector/brief text can render. Falls back
    to an empty list (client JS then does its own fetch as before) if the
    Bitcast API is briefly unavailable, so a hiccup here never blocks the
    page from loading at all."""
    html = (FRONTEND_DIR / "index.html").read_text()
    try:
        briefs = _get_cached_briefs()
    except Exception:
        briefs = []

    injected = f"<script>window.__PRELOADED_BRIEFS__ = {json.dumps(briefs)};</script>"
    html = html.replace("<head>", "<head>\n" + injected, 1)
    return HTMLResponse(content=html)


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
