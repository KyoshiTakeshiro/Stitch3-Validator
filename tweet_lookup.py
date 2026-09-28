"""Tweet Engagement Lookup backend, mounted as a router (prefix
/api/tweet-lookup) alongside the Tweet Validator and Engagement Value tools.

Given any real tweet URL, fetches its actual retweeters and quote-tweeters
from Desearch (the same third-party X-data provider bitcast-x's own
DesearchProvider uses internally -- see src/bitcast_x/x_provider.py) and
scores each one exactly like Engagement Value does: influence (from the
ecosystem map) x RETWEET_WEIGHT/QUOTE_WEIGHT x the same cabal/relationship
discount (scale_factor). Unlike Engagement Value, this isn't scoped to any
one campaign -- there's no brief/campaign context for an arbitrary tweet, so
scoring uses the *current* (latest) ecosystem map for the chosen pool
directly, rather than a campaign-window-relative one.

Real, per-request cost: every check spends real Desearch API credits (see
DESEARCH_API_KEY below), unlike everything else in this app, which is
either free or already-paid-for. Caching + a dedicated rate limit exist
specifically to bound that cost -- don't remove either without a reason.
"""

import asyncio
import logging
import os
import re
import time
from threading import Lock
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request

from engagement import (
    QUOTE_WEIGHT,
    RETWEET_WEIGHT,
    fetch_augmented_manifest,
    fetch_ecosystem_map,
    relationship_lookup,
    scale_factor,
    validate_ecosystem,
)

LOGGER = logging.getLogger(__name__)

DESEARCH_API_BASE = "https://api.desearch.ai"
# How many retweeters/quote-tweeters to fetch per check, at most. Real cost
# and latency both scale with this -- measured live against Desearch:
# count=50 on the search (quote) side took ~11.5s, count=100 risked a 20s
# timeout. Retweeters are cursor-paginated (no count param), so the cap is
# enforced by stopping once collected reaches this many, however many
# cursor pages that takes.
ENGAGER_FETCH_CAP = 100
DESEARCH_TIMEOUT = 45.0

# Cost control: real money is spent per Desearch call, unlike anything else
# in this app. The raw Desearch data (tweet + retweeters + quote-tweeters) is
# cached per tweet, not per ecosystem -- scoring against a different pool
# only needs a different ecosystem map, so switching ecosystem or opening a
# shared link re-scores for free. The rate limit (same shape as main.py's
# /evaluate* limiter, separate budget) only applies when Desearch is
# actually called.
CACHE_TTL = 15 * 60
_raw_cache: dict[str, tuple[dict, float]] = {}
_cache_lock = Lock()

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 5
_rate_limit_hits: dict[str, list[float]] = {}
_rate_limit_lock = Lock()

router = APIRouter()

# X stores a linked cashtag in the raw tweet text as an asset ID, not the
# "$SYMBOL" the reader sees: "bittensor:native" for a chain's own coin, or
# "base:0x6f63..." for a token contract. Desearch passes that raw text through
# with no symbol mapping, so it's restored here.
NATIVE_CASHTAGS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "bittensor": "TAO",
    "hyperliquid": "HYPE",
    "dogecoin": "DOGE",
    "ripple": "XRP",
    "xrp": "XRP",
    "cardano": "ADA",
    "tron": "TRX",
    "sui": "SUI",
    "avalanche": "AVAX",
    "bnb": "BNB",
    "binance": "BNB",
    "litecoin": "LTC",
    "polkadot": "DOT",
    "near": "NEAR",
    "aptos": "APT",
    "ton": "TON",
}
ASSET_ID_RE = re.compile(
    r"(?<![\w/:.])([a-z][a-z0-9_-]*):(native|0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44})(?![\w])"
)
DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{address}"
# address.lower() -> symbol, or None if DexScreener didn't know it. Token
# symbols don't change, so this is kept for the process lifetime.
_token_symbol_cache: dict[str, Optional[str]] = {}

TWEET_URL_RE = re.compile(
    r"(?:twitter\.com|x\.com)/[^/]+/status(?:es)?/(\d+)", re.IGNORECASE
)
BARE_ID_RE = re.compile(r"^\d+$")


def _desearch_api_key() -> Optional[str]:
    # Read lazily, not a module-level constant -- this module can be
    # imported before main.py's own load_dotenv() call runs, so a value
    # captured at import time would always see an unset environment.
    return os.getenv("DESEARCH_API_KEY")


def _client_ip(request: Request) -> str:
    # Duplicated from main.py/engagement.py rather than imported, same
    # reasoning as engagement.py's own copy: avoids a circular import.
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


def parse_tweet_id(raw: str) -> Optional[str]:
    """Accepts a full x.com/twitter.com status URL (with or without query
    params like ?s=20), or a bare numeric tweet ID. Returns None if nothing
    recognizable is found."""
    raw = raw.strip()
    if not raw:
        return None
    if BARE_ID_RE.match(raw):
        return raw
    match = TWEET_URL_RE.search(raw)
    return match.group(1) if match else None


async def _desearch_get(path: str, params: dict) -> dict:
    api_key = _desearch_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Tweet lookup isn't configured on this server yet.",
        )
    headers = {"Authorization": api_key.strip()}
    try:
        async with httpx.AsyncClient(timeout=DESEARCH_TIMEOUT) as client:
            resp = await client.get(f"{DESEARCH_API_BASE}{path}", headers=headers, params=params)
    except httpx.HTTPError as exc:
        LOGGER.exception("Desearch request failed: path=%s", path)
        raise HTTPException(status_code=502, detail="Tweet lookup service is currently unavailable.") from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Tweet not found.")
    if not resp.is_success:
        LOGGER.warning("Desearch %s returned %s: %s", path, resp.status_code, resp.text[:300])
        raise HTTPException(status_code=502, detail="Tweet lookup service is currently unavailable.")
    return resp.json()


async def _token_symbol(address: str) -> Optional[str]:
    key = address.lower()
    if key in _token_symbol_cache:
        return _token_symbol_cache[key]
    symbol = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(DEXSCREENER_TOKEN_URL.format(address=address))
        if resp.is_success:
            for pair in resp.json().get("pairs") or []:
                base = pair.get("baseToken") or {}
                if (base.get("address") or "").lower() == key and base.get("symbol"):
                    symbol = base["symbol"]
                    break
    except (httpx.HTTPError, ValueError):
        LOGGER.warning("DexScreener lookup failed for %s", address)
        return None  # not cached, so a later check can retry
    _token_symbol_cache[key] = symbol
    return symbol


async def restore_cashtags(text: str) -> str:
    matches = list(ASSET_ID_RE.finditer(text))
    if not matches:
        return text
    addresses = {m.group(2) for m in matches if m.group(2) != "native"}
    resolved = dict(zip(addresses, await asyncio.gather(*(_token_symbol(a) for a in addresses))))

    def replace(m: re.Match) -> str:
        chain, asset = m.group(1), m.group(2)
        symbol = NATIVE_CASHTAGS.get(chain) if asset == "native" else resolved.get(asset)
        return f"${symbol}" if symbol else m.group(0)

    return ASSET_ID_RE.sub(replace, text)


async def fetch_tweet(tweet_id: str) -> dict:
    return await _desearch_get("/twitter/post", {"id": tweet_id})


async def fetch_retweeters(tweet_id: str, cap: int = ENGAGER_FETCH_CAP) -> tuple[list[dict], bool]:
    """Returns a list of {"user": {...}} dicts, matching the shape
    fetch_quote_tweeters() returns (full tweet objects, each with a nested
    "user") -- /twitter/post/retweeters itself returns flat user objects
    with no such wrapper, so this normalizes them to keep score_engagers()
    in lookup() able to read both engagement types the same way. Missing
    this wrapper previously meant every retweeter's username read as None
    and got silently dropped (item.get("user") on a flat user dict is
    always None) -- retweets never scored at all, only quotes did."""
    users: list[dict] = []
    cursor = None
    while len(users) < cap:
        params = {"id": tweet_id}
        if cursor:
            params["cursor"] = cursor
        data = await _desearch_get("/twitter/post/retweeters", params)
        page = data.get("users") or []
        if not page:
            break
        users.extend(page)
        cursor = data.get("next_cursor")
        if not cursor:
            break
    # Desearch's cursor pages overlap (seen live: 53 rows for a 29-retweet
    # tweet), so dedupe here -- the count is shown to the user.
    unique: dict[str, dict] = {}
    for u in users[:cap]:
        key = str(u.get("id") or (u.get("username") or "").casefold())
        if key and key not in unique:
            unique[key] = u
    return [{"user": u} for u in unique.values()], len(users) >= cap


async def fetch_quote_tweeters(tweet_id: str, cap: int = ENGAGER_FETCH_CAP) -> tuple[list[dict], bool]:
    data = await _desearch_get(
        "/twitter",
        {"query": f"quoted_tweet_id:{tweet_id}", "sort": "Latest", "count": cap},
    )
    items = data if isinstance(data, list) else []
    unique: dict[str, dict] = {}
    for item in items[:cap]:
        key = str(item.get("id") or "")
        if key and key not in unique:
            unique[key] = item
    return list(unique.values()), len(items) >= cap


async def latest_ecosystem_map(ecosystem_id: str) -> dict:
    """The single most-recently-updated ecosystem map for a pool, independent
    of any campaign window -- this feature has no campaign/brief context (any
    real tweet can be checked, not just ones tied to an active brief), so
    there's no campaign to derive a relevant-maps window from the way
    engagement.py's ecosystem_maps_for_campaign() does."""
    manifest = await fetch_augmented_manifest()
    refs = [m for m in manifest["ecosystem_maps"] if m["ecosystem_id"] == ecosystem_id]
    if not refs:
        raise HTTPException(status_code=404, detail="No ecosystem map data for this pool yet.")
    latest_ref = max(refs, key=lambda m: m["updated_at"])
    return await fetch_ecosystem_map(latest_ref["path"], latest_ref["digest"])


def _cache_get(tweet_id: str) -> Optional[dict]:
    with _cache_lock:
        entry = _raw_cache.get(tweet_id)
    if entry is None:
        return None
    data, expires_at = entry
    if time.time() >= expires_at:
        return None
    return data


def _cache_set(tweet_id: str, data: dict) -> None:
    with _cache_lock:
        _raw_cache[tweet_id] = (data, time.time() + CACHE_TTL)


async def fetch_raw(tweet_id: str, request: Request) -> tuple[dict, bool]:
    """The tweet plus its retweeters/quote-tweeters, from cache if fresh.
    Returns (raw, cached)."""
    cached = _cache_get(tweet_id)
    if cached is not None:
        return cached, True
    _enforce_rate_limit(request)
    tweet, (retweeters, retweeters_capped), (quote_tweeters, quotes_capped) = await asyncio.gather(
        fetch_tweet(tweet_id),
        fetch_retweeters(tweet_id),
        fetch_quote_tweeters(tweet_id),
    )
    raw = {
        "tweet": tweet,
        "text": await restore_cashtags(tweet.get("text", "")),
        "retweeters": retweeters,
        "quote_tweeters": quote_tweeters,
        "retweeters_capped": retweeters_capped,
        "quotes_capped": quotes_capped,
    }
    _cache_set(tweet_id, raw)
    return raw, False


@router.get("/lookup")
async def lookup(url: str, request: Request, ecosystem_id: str = "tao"):
    validate_ecosystem(ecosystem_id)

    tweet_id = parse_tweet_id(url)
    if not tweet_id:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a tweet URL or ID (expected something like https://x.com/user/status/12345…).",
        )

    (raw, cached), ecosystem_map = await asyncio.gather(
        fetch_raw(tweet_id, request),
        latest_ecosystem_map(ecosystem_id),
    )
    tweet = raw["tweet"]
    retweeters = raw["retweeters"]
    quote_tweeters = raw["quote_tweeters"]

    author = tweet.get("user") or {}
    author_username = author.get("username", "")
    author_name = author.get("name") or author_username
    author_key = author_username.casefold()

    considered = {
        a["username"].casefold(): {"influence": a["influence"], "display": a["username"]}
        for a in ecosystem_map["accounts"]
    }
    rel_lookup = relationship_lookup(ecosystem_map)
    # Same rank-derivation pattern as engagement.py's lookup()/leaderboard():
    # position in the whole ecosystem's influence ordering, 1 = most influential.
    ranked_considered = sorted(considered.items(), key=lambda kv: kv[1]["influence"], reverse=True)
    rank_lookup = {key: i for i, (key, _) in enumerate(ranked_considered, start=1)}

    def score_engagers(raw_engagers: list[dict], kind: str, weight: float) -> list[dict]:
        scored = []
        seen: set[str] = set()
        for item in raw_engagers:
            username = (item.get("user") or {}).get("username")
            if not username:
                continue
            key = username.casefold()
            if key in seen or key == author_key:
                continue  # a self-retweet/quote, or a duplicate page overlap
            seen.add(key)
            entry = considered.get(key)
            if entry is None:
                continue  # not in this ecosystem's pool -- omitted per design
            rel_score = rel_lookup.get((key, author_key), 0.0)
            scale = scale_factor(rel_score)
            scored.append(
                {
                    "username": entry["display"],
                    "type": kind,
                    "rank": rank_lookup[key],
                    "influence": round(entry["influence"], 2),
                    "relationship_score": round(rel_score, 2),
                    "value": round(entry["influence"] * weight * scale, 4),
                }
            )
        return scored

    engagers = score_engagers(retweeters, "retweet", RETWEET_WEIGHT) + score_engagers(
        quote_tweeters, "quote", QUOTE_WEIGHT
    )
    engagers.sort(key=lambda e: e["value"], reverse=True)
    total_value = round(sum(e["value"] for e in engagers), 4)

    return {
        "tweet_id": tweet_id,
        "tweet_url": tweet.get("url") or f"https://x.com/{author_username}/status/{tweet_id}",
        "author": author_username,
        "author_name": author_name,
        "text": raw["text"],
        "retweet_count": tweet.get("retweet_count", 0),
        "quote_count": tweet.get("quote_count", 0),
        # How many engagers were actually fetched and scored, so the page can
        # say so when a popular tweet hit ENGAGER_FETCH_CAP.
        "retweeters_fetched": len(retweeters),
        "quotes_fetched": len(quote_tweeters),
        "retweeters_capped": raw["retweeters_capped"],
        "quotes_capped": raw["quotes_capped"],
        "ecosystem_id": ecosystem_id,
        "total_value": total_value,
        "ranked_engager_count": len(engagers),
        "engagers": engagers,
        "cached": cached,
    }
