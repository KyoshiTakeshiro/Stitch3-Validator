"""Engagement Value Calculator backend, ported from the standalone
~/engagement-value-calculator project into this app as a mounted router
(prefix /api/engagement) rather than a separate deployment.

Reproduces bitcast-network/bitcast-x's public scoring formula
(src/bitcast_x/scoring.py, campaigns.py) so a creator can look up what a
given engager's quote/retweet is actually worth to them, per campaign.
Covers all 3 ecosystems Bitcast runs campaigns in (tao, hyperliquid,
prediction_markets) -- same pool ids the Pre-Submission Check tool uses.
"""

import time

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from activity_log import hash_ip, log_event

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
_avatar_cache: dict[str, tuple[bytes, str, float]] = {}


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
    manifest = await fetch_manifest()
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
    manifest = await fetch_manifest()
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
    manifest = await fetch_manifest()
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
    manifest = await fetch_manifest()
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
    """Proxy + cache unavatar.io lookups. unavatar's anonymous tier has a
    very low daily request quota shared per source IP. Proxying here means
    every visitor shares one server IP and one cache instead of each
    browser spending its own quota directly against unavatar.io, and a
    6h TTL means the same handle isn't re-fetched on every page view."""
    key = username.casefold()
    now = time.time()
    cached = _avatar_cache.get(key)
    if cached and now - cached[2] < AVATAR_TTL:
        content, content_type, _ = cached
        return Response(content=content, media_type=content_type, headers={"Cache-Control": "public, max-age=21600"})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{AVATAR_BASE_URL}/{username}")
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError("unavatar returned a non-image response")
            content = resp.content
    except (httpx.HTTPError, ValueError):
        if cached:
            content, content_type, _ = cached
            return Response(content=content, media_type=content_type, headers={"Cache-Control": "public, max-age=21600"})
        raise HTTPException(status_code=502, detail="Avatar unavailable")

    _avatar_cache[key] = (content, content_type, now)
    return Response(content=content, media_type=content_type, headers={"Cache-Control": "public, max-age=21600"})
