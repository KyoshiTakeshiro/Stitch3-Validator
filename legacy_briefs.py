"""Shared fetch/cache for the retired Bitcast x-briefs archive endpoint.

Upstream retired this endpoint's data feed after their Aug 14 rewrite -- it
still 200s but silently stopped receiving new campaigns, topping out at
074_nodexo (075_bitcast onward only ever appeared on the live
campaign-manifest-v4 endpoint). Still confirmed live and useful as a
read-only historical archive (81 campaigns, 001_score through 074_nodexo).

Both main.py (the Tweet Validator's brief selector) and engagement.py (the
Engagement Value campaign selector) merge this archive into their live
manifest data, so it lives here once rather than being fetched/cached
twice with two copies to keep in sync.
"""

import requests

BITCAST_LEGACY_BRIEFS_ENDPOINT = "https://bitcast-api.bitcast.network/api/v2/validator/x-briefs"

# The endpoint's data is frozen (dead feed, see module docstring) -- it
# can't change between requests, so this is cached for the process
# lifetime rather than re-fetched on a TTL like the live manifest.
_legacy_briefs_cache = {"data": None}


def fetch_legacy_briefs() -> list[dict]:
    """Normalizes the retired x-briefs endpoint's items into a flat shape --
    this endpoint's own shape already matches almost exactly (id/pool/
    start_date/end_date/display/brief/tag/prompt_version all present
    directly, no nested access/pools unwrapping needed like the live
    manifest requires), it just carries a few extra fields (qrt, budget,
    max_tweets, etc.) not every consumer reads, so those are dropped."""
    resp = requests.get(BITCAST_LEGACY_BRIEFS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {
            "id": i.get("id"),
            "pool": i.get("pool"),
            "start_date": i.get("start_date", ""),
            "end_date": i.get("end_date", ""),
            "display": i.get("display", ""),
            "brief": i.get("brief", ""),
            "tag": i.get("tag"),
            "prompt_version": i.get("prompt_version", 1),
            "exclusive_miner_hotkey": None,
        }
        for i in items
    ]


def get_cached_legacy_briefs() -> list[dict]:
    if _legacy_briefs_cache["data"] is not None:
        return _legacy_briefs_cache["data"]
    try:
        _legacy_briefs_cache["data"] = fetch_legacy_briefs()
    except Exception:
        # Fail open -- one dead feed shouldn't take a whole brief/campaign
        # list down with it.
        return []
    return _legacy_briefs_cache["data"]
