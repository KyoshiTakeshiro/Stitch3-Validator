"""Engagement Value Calculator backend, ported from the standalone
~/engagement-value-calculator project into this app as a mounted router
(prefix /api/engagement) rather than a separate deployment.

Reproduces bitcast-network/bitcast-x's public scoring formula
(src/bitcast_x/scoring.py, campaigns.py) so a creator can look up what a
given engager's quote/retweet is actually worth to them, per campaign.
Covers all 3 ecosystems Bitcast runs campaigns in (tao, hyperliquid,
prediction_markets) -- same pool ids the Pre-Submission Check tool uses.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from activity_log import hash_ip, log_event
from legacy_briefs import get_cached_legacy_briefs

LOGGER = logging.getLogger(__name__)

BITCAST_API_BASE = "https://bitcast-api.bitcast.network"
MANIFEST_URL = f"{BITCAST_API_BASE}/api/v2/public/x/campaign-manifest-v4"
AVATAR_BASE_URL = "https://unavatar.io/x"

# Constants copied verbatim from src/bitcast_x/scoring.py and campaigns.py.
BASELINE_TWEET_SCORE_FACTOR = 2.0
RETWEET_WEIGHT = 1.0
QUOTE_WEIGHT = 3.0
CABAL_BASE = 0.1
CABAL_SCALE = 0.9
STALE_DECAY = 0.5

ALLOWED_ECOSYSTEMS = {"tao", "hyperliquid", "prediction_markets"}
MANIFEST_TTL = 300
AVATAR_TTL = 6 * 3600

# warm_avatars() spends its daily budget working through ecosystems in this
# order, tao (Bittensor) first by request -- since the budget is shared and
# small (see _avatar_daily_budget below), an earlier ecosystem's considered list
# is exhausted before a later one gets any budget at all, not just given a
# head start. ALLOWED_ECOSYSTEMS itself stays a set (only used for
# membership checks elsewhere) so this ordering lives in exactly one place.
AVATAR_WARM_ECOSYSTEM_ORDER = ("tao", "hyperliquid", "prediction_markets")

# Disk-backed avatar store -- see warm_avatars()/the /avatar route below for
# how this is used. Local/VPS-only runtime data, not source (gitignored).
AVATAR_STORE_DIR = Path(__file__).parent / "avatar_cache"
AVATAR_REFRESH_SECONDS = 7 * 24 * 3600  # re-fetch a stored avatar at most weekly

# unavatar.io's free tiers are DAILY quotas, not just burst/rate limits
# (confirmed live: an over-quota call returns
# {"code": "ERATE", "message": "Daily anonymous rate limit reached..."}).
# Spreading requests out over time doesn't help with a daily cap -- without
# this budget, one warm_avatars() pass across a full considered-accounts
# list (hundreds to low thousands of accounts per ecosystem) or a burst of
# real visitor traffic could exhaust the entire day's quota in seconds.
# This caps how many *new* live fetches (cache misses) this process will
# attempt per UTC day, shared between warm_avatars() and live request-time
# misses. warm_avatars() works through EVERY considered account across all
# 3 ecosystems' current campaigns, highest-influence first, not just a
# top-N slice -- it just never fetches more than this many per run, so
# coverage grows by roughly this many genuinely new accounts each time
# it's triggered (once daily at most makes sense, since the budget resets
# on a UTC day boundary) until eventually every account has a stored
# avatar.
#
# With a registered UNAVATAR_API_KEY (see _unavatar_api_key below), 50
# origin requests/day are included free per unavatar's own pricing docs
# (unavatar.io/docs#pricing); anything past that is metered billing, which
# we deliberately never want to risk incurring without being asked to, so
# the budget is capped at exactly that free-tier ceiling -- never higher,
# even though unavatar itself would technically allow more (for a price).
# Without a key (e.g. local dev), stay well under the anonymous tier's own
# 25/day per-IP ceiling. _try_consume_avatar_budget's `>=` check makes
# this a hard cap either way, never a soft/approximate one.
AVATAR_DAILY_FETCH_BUDGET_WITH_KEY = 50
AVATAR_DAILY_FETCH_BUDGET_ANONYMOUS = 20


def _unavatar_api_key() -> Optional[str]:
    # Read lazily (not a module-level constant) -- engagement.py is
    # imported by main.py before main.py's own load_dotenv() call runs,
    # so a value captured at import time would always see an unset
    # environment and silently miss the key.
    return os.getenv("UNAVATAR_API_KEY")


def _avatar_daily_budget() -> int:
    return AVATAR_DAILY_FETCH_BUDGET_WITH_KEY if _unavatar_api_key() else AVATAR_DAILY_FETCH_BUDGET_ANONYMOUS


AVATAR_BUDGET_FILE = Path(__file__).parent / "avatar_fetch_budget.json"

router = APIRouter()


def _client_ip(request: Request) -> str:
    # Same logic as main.py's _client_ip -- duplicated rather than imported
    # to avoid a circular import (main.py imports this module).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_manifest_cache: dict = {"data": None, "fetched_at": 0.0}
_map_cache: dict[str, dict] = {}
_avatar_cache: dict[str, tuple[bytes, str, float]] = {}  # in-memory L1, process-local
_augmented_manifest_cache: dict = {"data": None, "manifest_fetched_at": None}
_avatar_budget_lock = asyncio.Lock()


def _today_utc() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


async def _try_consume_avatar_budget() -> bool:
    """Returns True and consumes one unit of today's live-fetch budget, or
    False if today's budget is already spent. File-backed (not in-memory)
    so the budget survives a restart -- a deploy shouldn't hand out a fresh
    quota for free. Guarded by an asyncio.Lock since concurrent request
    handlers can race on the read-modify-write otherwise."""
    async with _avatar_budget_lock:
        today = _today_utc()
        try:
            state = json.loads(AVATAR_BUDGET_FILE.read_text())
        except (OSError, ValueError):
            state = {}
        if state.get("date") != today:
            state = {"date": today, "used": 0}
        if state["used"] >= _avatar_daily_budget():
            return False
        state["used"] += 1
        try:
            AVATAR_BUDGET_FILE.write_text(json.dumps(state))
        except OSError:
            pass  # Best-effort -- worst case we slightly overspend the budget.
        return True


def _avatar_disk_paths(username: str) -> tuple[Path, Path]:
    """Hash the (casefolded) username into the filename rather than using it
    directly -- username is client-controlled input on the /avatar/{username}
    route, and writing it straight into a filesystem path would be a path
    traversal risk. The hash also sidesteps needing to sanitize whatever
    characters a real X handle can contain."""
    digest = hashlib.sha256(username.casefold().encode()).hexdigest()
    return AVATAR_STORE_DIR / f"{digest}.img", AVATAR_STORE_DIR / f"{digest}.json"


def _read_avatar_disk(username: str) -> Optional[tuple[bytes, str, float]]:
    img_path, meta_path = _avatar_disk_paths(username)
    try:
        meta = json.loads(meta_path.read_text())
        content = img_path.read_bytes()
    except (OSError, ValueError, KeyError):
        return None
    return content, meta["content_type"], meta["fetched_at"]


def _write_avatar_disk(username: str, content: bytes, content_type: str, fetched_at: float) -> None:
    img_path, meta_path = _avatar_disk_paths(username)
    try:
        AVATAR_STORE_DIR.mkdir(exist_ok=True)
        img_path.write_bytes(content)
        meta_path.write_text(json.dumps({"content_type": content_type, "fetched_at": fetched_at}))
    except OSError:
        pass  # Best-effort -- a failed disk write just means no persistence for this one.


async def _fetch_avatar_live(username: str) -> tuple[bytes, str]:
    api_key = _unavatar_api_key()
    headers = {"x-api-key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{AVATAR_BASE_URL}/{username}", headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError("unavatar returned a non-image response")
        return resp.content, content_type


async def _get_avatar(username: str) -> tuple[bytes, str]:
    """Three-tier lookup: in-memory (fast, per-process, AVATAR_TTL) -> disk
    (persistent across restarts, AVATAR_REFRESH_SECONDS) -> live unavatar.io
    fetch. Only a genuine cold cache (nothing in memory or on disk) or a
    disk copy older than AVATAR_REFRESH_SECONDS reaches the live fetch --
    warm_avatars() below works through every considered account,
    highest-influence first, so the accounts real visitors are most likely
    to look at tend to already be on disk before they ever ask for it. A
    live fetch also costs one unit of today's daily budget (see
    _avatar_daily_budget) -- once that's spent for the day, this falls
    straight back to a stale copy (or a clean miss) without attempting the
    request at all, since unavatar.io's daily quota means it would just
    fail anyway."""
    key = username.casefold()
    now = time.time()

    cached = _avatar_cache.get(key)
    if cached and now - cached[2] < AVATAR_TTL:
        return cached[0], cached[1]

    disk = _read_avatar_disk(username)
    if disk and now - disk[2] < AVATAR_REFRESH_SECONDS:
        _avatar_cache[key] = disk
        return disk[0], disk[1]

    stale = cached or disk
    if not await _try_consume_avatar_budget():
        if stale:
            return stale[0], stale[1]
        raise HTTPException(status_code=502, detail="Avatar unavailable")

    try:
        content, content_type = await _fetch_avatar_live(username)
    except (httpx.HTTPError, ValueError):
        # Live fetch failed -- fall back to whatever's cached/stored, even
        # if stale, rather than a broken image.
        if stale:
            return stale[0], stale[1]
        raise HTTPException(status_code=502, detail="Avatar unavailable")

    _avatar_cache[key] = (content, content_type, now)
    _write_avatar_disk(username, content, content_type, now)
    return content, content_type


async def fetch_manifest() -> dict:
    now = time.time()
    if _manifest_cache["data"] is not None and now - _manifest_cache["fetched_at"] < MANIFEST_TTL:
        return _manifest_cache["data"]
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(MANIFEST_URL)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Bitcast API is unavailable right now. Try again shortly.") from exc
    _manifest_cache["data"] = data
    _manifest_cache["fetched_at"] = now
    return data


async def fetch_ecosystem_map(path: str, digest: str) -> dict:
    if digest in _map_cache:
        return _map_cache[digest]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BITCAST_API_BASE}{path}")
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Bitcast API is unavailable right now. Try again shortly.") from exc
    _map_cache[digest] = data
    return data


def validate_ecosystem(ecosystem_id: str) -> str:
    if ecosystem_id not in ALLOWED_ECOSYSTEMS:
        raise HTTPException(status_code=400, detail=f"Unsupported ecosystem: {ecosystem_id}")
    return ecosystem_id


def get_campaign(manifest: dict, campaign_id: str) -> dict:
    for c in manifest["campaigns"]:
        if c["access"]["campaign_id"] == campaign_id:
            return c
    raise HTTPException(status_code=404, detail="Campaign not found")


async def ecosystem_maps_for_campaign(manifest: dict, campaign: dict, ecosystem_id: str) -> list[dict]:
    """Mirrors campaigns.ecosystem_maps_for_campaign: every map version whose
    active interval overlaps the campaign window, for one ecosystem. Manifest
    timestamps are ISO-8601 UTC with a 'Z' suffix, so plain string comparison
    orders them correctly."""
    refs = sorted(
        (m for m in manifest["ecosystem_maps"] if m["ecosystem_id"] == ecosystem_id),
        key=lambda m: m["updated_at"],
    )
    opens_at = campaign["opens_at"]
    closes_at = campaign["closes_at"]
    relevant = [
        ref
        for i, ref in enumerate(refs)
        if ref["updated_at"] <= closes_at
        and (i + 1 == len(refs) or refs[i + 1]["updated_at"] >= opens_at)
    ]
    maps = []
    for ref in relevant:
        maps.append(await fetch_ecosystem_map(ref["path"], ref["digest"]))
    return maps


def _synthesize_legacy_campaign(item: dict) -> Optional[dict]:
    """Reshape one legacy x-briefs item into the live manifest's campaign
    shape (access.campaign_id / pools / opens_at / closes_at), so it can
    flow through get_campaign, ecosystem_maps_for_campaign, etc. exactly
    like a real manifest entry -- no separate "legacy campaign" code path
    to keep in sync in the scoring endpoints below."""
    pool = item.get("pool")
    start_date = item.get("start_date")
    end_date = item.get("end_date")
    if not item.get("id") or not pool or not start_date or not end_date:
        return None
    return {
        "access": {"campaign_id": item["id"], "mining_protocol": "legacy_connection"},
        "display": item.get("display") or item["id"],
        "pools": [pool],
        "opens_at": f"{start_date}T00:00:00Z",
        "closes_at": f"{end_date}T23:59:59Z",
    }


async def fetch_augmented_manifest() -> dict:
    """The live manifest plus legacy x-briefs campaigns (see legacy_briefs.py)
    not already in it, per ecosystem cut off at the oldest legacy campaign
    that still has real engagement data. Most of the 81 legacy campaigns
    ran before this manifest's oldest retained ecosystem map (confirmed
    live: the earliest tao map is from 2026-08-02, while the legacy
    archive starts at 2025-11-12) -- selecting one of those would 404 the
    instant someone picks it, so rather than hardcoding a date, this finds
    the actual oldest working campaign per ecosystem (the one whose window
    overlaps that ecosystem's oldest retained map) and drops only the
    data-less campaigns older than it. A data-less campaign at or after
    that cutoff is left in rather than assumed impossible -- it would
    still surface the existing "no ecosystem map data yet" error rather
    than being silently dropped.
    Cached alongside the base manifest (recomputed only when it refetches)."""
    manifest = await fetch_manifest()
    if _augmented_manifest_cache["manifest_fetched_at"] == _manifest_cache["fetched_at"]:
        return _augmented_manifest_cache["data"]

    live_ids = {c["access"]["campaign_id"] for c in manifest["campaigns"]}
    candidates = [
        campaign
        for item in get_cached_legacy_briefs()
        if item.get("id") not in live_ids
        and (campaign := _synthesize_legacy_campaign(item)) is not None
    ]

    has_data: dict[str, bool] = {}
    for campaign in candidates:
        try:
            maps = await ecosystem_maps_for_campaign(manifest, campaign, campaign["pools"][0])
        except HTTPException:
            maps = []
        has_data[campaign["access"]["campaign_id"]] = bool(maps)

    oldest_working_opens_at: dict[str, str] = {}
    for campaign in candidates:
        if not has_data[campaign["access"]["campaign_id"]]:
            continue
        ecosystem_id = campaign["pools"][0]
        current = oldest_working_opens_at.get(ecosystem_id)
        if current is None or campaign["opens_at"] < current:
            oldest_working_opens_at[ecosystem_id] = campaign["opens_at"]

    synthesized = []
    for campaign in candidates:
        campaign_id = campaign["access"]["campaign_id"]
        cutoff = oldest_working_opens_at.get(campaign["pools"][0])
        if not has_data[campaign_id] and (cutoff is None or campaign["opens_at"] < cutoff):
            continue
        synthesized.append(campaign)

    augmented = (
        manifest if not synthesized else {**manifest, "campaigns": manifest["campaigns"] + synthesized}
    )
    _augmented_manifest_cache["data"] = augmented
    _augmented_manifest_cache["manifest_fetched_at"] = _manifest_cache["fetched_at"]
    return augmented


async def warm_cache() -> None:
    """Pre-fetch the manifest and the ecosystem map(s) each ecosystem's
    most recent campaign needs, so the cold-cache network cost (fetching
    the manifest, then a multi-hundred-KB ecosystem map per pool from
    Bitcast's API) lands on server startup instead of on whichever real
    visitor happens to open the Engagement Value tab first after a
    deploy/restart. _map_cache has no TTL (see fetch_ecosystem_map), so
    once warm this stays warm until the process restarts again -- this
    only needs to run once per startup, not on a timer. Best-effort: if
    Bitcast's API is briefly down at the exact moment of startup, the
    first real request just falls back to fetching it live as before.

    Also does the initial avatar disk-store warm (see warm_avatars) --
    there's no recurring refresh loop; re-warming after that is manual
    only, triggered via the admin panel's "Refresh avatars" action
    (POST /admin/refresh-avatars in main.py)."""
    try:
        manifest = await fetch_manifest()
    except HTTPException:
        return
    for ecosystem_id in ALLOWED_ECOSYSTEMS:
        campaigns = sorted(
            (c for c in manifest["campaigns"] if ecosystem_id in c["pools"]),
            key=lambda c: c["opens_at"],
            reverse=True,
        )
        if not campaigns:
            continue
        try:
            await ecosystem_maps_for_campaign(manifest, campaigns[0], ecosystem_id)
        except HTTPException:
            continue

    await warm_avatars()


async def warm_avatars() -> dict:
    """Works through EVERY considered account on each ecosystem's most
    recently opened campaign -- not just a top-N slice -- fetching and
    persisting to disk whichever ones aren't already cached (or have gone
    stale, see AVATAR_REFRESH_SECONDS). Ecosystems are processed in
    AVATAR_WARM_ECOSYSTEM_ORDER (tao/Bittensor first, by request), and
    within an ecosystem, highest-influence account first. The only thing
    that actually limits how much happens in one call is the daily budget
    (see _avatar_daily_budget): once that's spent, this stops and picks up
    again from wherever it left off next time it's called (already-cached
    accounts are skipped instantly, so re-running doesn't redo work). Since
    the budget is small and shared across ecosystems, an earlier ecosystem
    in the order is fully exhausted before a later one gets any budget at
    all -- with tao first, that means every tao account gets covered
    before hyperliquid or prediction_markets sees a single fetch. Runs
    once at startup (via warm_cache) and otherwise only when manually
    triggered -- no automatic recurring schedule. Returns a small stats
    dict so a manual trigger (the admin panel) can show what actually
    happened."""
    stats = {"considered": 0, "already_fresh": 0, "fetched": 0, "failed": 0, "budget_exhausted": False}
    try:
        manifest = await fetch_manifest()
    except HTTPException as exc:
        LOGGER.warning("warm_avatars: manifest unavailable, aborting this run: %s", exc)
        return stats

    ordered_usernames: list[str] = []
    seen: set[str] = set()
    for ecosystem_id in AVATAR_WARM_ECOSYSTEM_ORDER:
        campaigns = sorted(
            (c for c in manifest["campaigns"] if ecosystem_id in c["pools"]),
            key=lambda c: c["opens_at"],
            reverse=True,
        )
        if not campaigns:
            continue
        try:
            maps = await ecosystem_maps_for_campaign(manifest, campaigns[0], ecosystem_id)
            considered, _ = considered_accounts_for_campaign(maps)
        except (HTTPException, ValueError) as exc:
            LOGGER.warning("warm_avatars: skipping ecosystem=%s, no map data: %s", ecosystem_id, exc)
            continue
        ranked = sorted(considered.items(), key=lambda kv: kv[1]["influence"], reverse=True)
        for _, entry in ranked:
            key = entry["display"].casefold()
            if key not in seen:
                seen.add(key)
                ordered_usernames.append(entry["display"])

    stats["considered"] = len(ordered_usernames)
    now = time.time()
    for username in ordered_usernames:
        disk = _read_avatar_disk(username)
        if disk and now - disk[2] < AVATAR_REFRESH_SECONDS:
            stats["already_fresh"] += 1
            continue
        if not await _try_consume_avatar_budget():
            stats["budget_exhausted"] = True
            break
        try:
            content, content_type = await _fetch_avatar_live(username)
        except Exception:
            # Deliberately broad, not just (httpx.HTTPError, ValueError) --
            # one account's unexpected failure (a malformed URL from an
            # unusual username, some other httpx edge case, anything) must
            # never silently kill the rest of this run. This used to be
            # the narrower pair only, which is the likely explanation for
            # a real run once stopping partway through with no
            # budget_exhausted and no visible error: this is a
            # fire-and-forget asyncio task (see warm_cache/main.py's
            # startup hook), so an uncaught exception here just vanishes
            # instead of showing up anywhere.
            LOGGER.exception("warm_avatars: live fetch failed for username=%s", username)
            stats["failed"] += 1
            continue
        _write_avatar_disk(username, content, content_type, time.time())
        stats["fetched"] += 1
    return stats


def considered_accounts_for_campaign(maps: list[dict], stale_decay: float = STALE_DECAY) -> tuple[dict, dict]:
    """Mirrors campaigns.considered_accounts_for_campaign: latest map's
    accounts at full influence, plus accounts from older maps that have
    since dropped out, at stale_decay. Returns {casefold_username: {influence,
    display}} plus the latest map (needed for relationship edges)."""
    if not maps:
        raise HTTPException(status_code=404, detail="This campaign has no ecosystem map data yet.")
    latest = maps[-1]
    considered = {
        a["username"].casefold(): {"influence": a["influence"], "display": a["username"]}
        for a in latest["accounts"]
    }
    older: dict = {}
    for m in maps[:-1]:
        for a in m["accounts"]:
            older[a["username"].casefold()] = a
    for key, a in older.items():
        considered.setdefault(key, {"influence": a["influence"] * stale_decay, "display": a["username"]})
    return considered, latest


def relationship_lookup(latest_map: dict) -> dict[tuple[str, str], float]:
    return {
        (edge["source_username"].casefold(), edge["target_username"].casefold()): edge["score"]
        for edge in latest_map["relationships"]
    }


def scale_factor(relationship_score: float) -> float:
    if relationship_score > 0:
        return CABAL_BASE + CABAL_SCALE / relationship_score
    return 1.0


@router.get("/campaigns")
async def list_campaigns(ecosystem_id: str = "tao"):
    validate_ecosystem(ecosystem_id)
    manifest = await fetch_augmented_manifest()
    campaigns = [
        {
            "id": c["access"]["campaign_id"],
            "display": c["display"],
            "opens_at": c["opens_at"],
            "closes_at": c["closes_at"],
        }
        for c in manifest["campaigns"]
        if ecosystem_id in c["pools"]
    ]
    campaigns.sort(key=lambda c: c["opens_at"], reverse=True)
    return campaigns


@router.get("/campaigns/{campaign_id}/accounts")
async def campaign_accounts(campaign_id: str, ecosystem_id: str = "tao"):
    validate_ecosystem(ecosystem_id)
    manifest = await fetch_augmented_manifest()
    campaign = get_campaign(manifest, campaign_id)
    if ecosystem_id not in campaign["pools"]:
        raise HTTPException(status_code=400, detail=f"This campaign isn't in the {ecosystem_id} ecosystem.")

    maps = await ecosystem_maps_for_campaign(manifest, campaign, ecosystem_id)
    considered, _ = considered_accounts_for_campaign(maps)
    return sorted((entry["display"] for entry in considered.values()), key=str.casefold)


@router.get("/campaigns/{campaign_id}/leaderboard")
async def leaderboard(campaign_id: str, ecosystem_id: str = "tao"):
    """Un-personalized ranking for a campaign: every considered account by
    influence, with quote/retweet value shown at full weight (no
    relationship discount, since that discount is relative to a specific
    handle). Lets the ranking table render before a creator has entered
    their own handle; /lookup replaces this with handle-relative values."""
    validate_ecosystem(ecosystem_id)
    manifest = await fetch_augmented_manifest()
    campaign = get_campaign(manifest, campaign_id)
    if ecosystem_id not in campaign["pools"]:
        raise HTTPException(status_code=400, detail=f"This campaign isn't in the {ecosystem_id} ecosystem.")

    maps = await ecosystem_maps_for_campaign(manifest, campaign, ecosystem_id)
    considered, _ = considered_accounts_for_campaign(maps)
    ranked = sorted(considered.items(), key=lambda kv: kv[1]["influence"], reverse=True)

    engagers = [
        {
            "username": entry["display"],
            "influence": round(entry["influence"], 2),
            "rank": rank,
            "relationship_score": 0.0,
            "quote_value": round(entry["influence"] * QUOTE_WEIGHT, 4),
            "retweet_value": round(entry["influence"] * RETWEET_WEIGHT, 4),
        }
        for rank, (_, entry) in enumerate(ranked, start=1)
    ]

    return {"campaign_id": campaign_id, "total_considered": len(considered), "engagers": engagers}


@router.get("/lookup")
async def lookup(campaign_id: str, handle: str, request: Request, ecosystem_id: str = "tao"):
    validate_ecosystem(ecosystem_id)
    manifest = await fetch_augmented_manifest()
    campaign = get_campaign(manifest, campaign_id)
    if ecosystem_id not in campaign["pools"]:
        raise HTTPException(status_code=400, detail=f"This campaign isn't in the {ecosystem_id} ecosystem.")

    maps = await ecosystem_maps_for_campaign(manifest, campaign, ecosystem_id)
    considered, latest_map = considered_accounts_for_campaign(maps)

    handle_key = handle.strip().lstrip("@").casefold()
    if not handle_key or handle_key not in considered:
        log_event(
            "engagement_lookup",
            handle=handle.strip().lstrip("@"),
            campaign_id=campaign_id,
            ecosystem_id=ecosystem_id,
            ip_hash=hash_ip(_client_ip(request)),
            found=False,
        )
        raise HTTPException(
            status_code=404,
            detail=f"@{handle.strip().lstrip('@')} isn't a considered account for this campaign.",
        )

    entry = considered[handle_key]
    influence = entry["influence"]
    ranked = sorted(considered.items(), key=lambda kv: kv[1]["influence"], reverse=True)
    rank = next(i for i, (u, _) in enumerate(ranked, start=1) if u == handle_key)
    baseline_score = round(influence * BASELINE_TWEET_SCORE_FACTOR, 6)

    rel_lookup = relationship_lookup(latest_map)
    rank_lookup = {u: i for i, (u, _) in enumerate(ranked, start=1)}

    engagers = []
    for username, other in considered.items():
        if username == handle_key:
            continue
        rel_score = rel_lookup.get((username, handle_key), 0.0)
        scale = scale_factor(rel_score)
        engagers.append(
            {
                "username": other["display"],
                "influence": round(other["influence"], 2),
                "rank": rank_lookup[username],
                "relationship_score": round(rel_score, 2),
                "quote_value": round(other["influence"] * QUOTE_WEIGHT * scale, 4),
                "retweet_value": round(other["influence"] * RETWEET_WEIGHT * scale, 4),
            }
        )
    engagers.sort(key=lambda e: e["quote_value"], reverse=True)

    log_event(
        "engagement_lookup",
        handle=entry["display"],
        campaign_id=campaign_id,
        ecosystem_id=ecosystem_id,
        ip_hash=hash_ip(_client_ip(request)),
        found=True,
    )

    return {
        "campaign_id": campaign_id,
        "handle": entry["display"],
        "influence": round(influence, 2),
        "rank": rank,
        "total_considered": len(considered),
        "baseline_score": baseline_score,
        "engagers": engagers,
    }


@router.get("/avatar/{username}")
async def avatar(username: str):
    """Proxy + cache unavatar.io lookups (see _get_avatar above for the
    memory -> disk -> live tiers). unavatar's anonymous tier has a very low
    daily request quota shared per source IP -- proxying here means every
    visitor shares one server IP and one cache instead of each browser
    spending its own quota directly against unavatar.io."""
    content, content_type = await _get_avatar(username)
    return Response(content=content, media_type=content_type, headers={"Cache-Control": "public, max-age=21600"})
