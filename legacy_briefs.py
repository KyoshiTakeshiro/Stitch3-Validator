"""Shared fetch/cache for the retired Bitcast x-briefs archive.

Timeline of this endpoint:
- Pre-2026-08-14: the live X validator feed.
- After the Aug 14 rewrite: frozen but still populated -- it stopped
  receiving new campaigns (topping out at 074_nodexo) but kept serving the
  81 historical ones (001_score -> 074_nodexo) as a read-only archive.
- 2026-09: fully drained. It now 200s with `{"items": []}` and is
  documented upstream as the "Retired legacy X validator feed". The 81
  historical campaigns are gone from it for good.

Since a pre-submission tool still benefits from being able to show what a
past campaign's brief looked like, those 81 are preserved in
`legacy_briefs_snapshot.json` (committed, captured from production's
in-memory cache the day the feed drained). `get_cached_legacy_briefs()`
uses the live endpoint if it ever serves data again, otherwise falls back
to the snapshot -- mirroring Bitcast's own "fall back to the on-disk copy"
pattern in their `bitcast` repo.

Both main.py (the Tweet Validator's brief selector) and engagement.py (the
Engagement Value campaign selector) merge this archive into their live
manifest data, so it lives here once rather than being fetched/cached
twice with two copies to keep in sync.
"""

import json
from pathlib import Path

import requests

BITCAST_LEGACY_BRIEFS_ENDPOINT = "https://bitcast-api.bitcast.network/api/v2/validator/x-briefs"
SNAPSHOT_FILE = Path(__file__).parent / "legacy_briefs_snapshot.json"

# The archive is immutable (dead feed, see module docstring) -- it can't
# change between requests, so this is cached for the process lifetime
# rather than re-fetched on a TTL like the live manifest.
_legacy_briefs_cache = {"data": None}


def _normalize(items: list[dict]) -> list[dict]:
    """Flatten to the shape both consumers read. The retired endpoint's own
    items already matched almost exactly (id/pool/start_date/end_date/
    display/brief/tag/prompt_version all present directly), it just carried
    a few extra fields (qrt, budget, max_tweets, etc.) nobody reads. The
    snapshot file is stored already-normalized, so this is a no-op for it."""
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


def fetch_legacy_briefs() -> list[dict]:
    resp = requests.get(BITCAST_LEGACY_BRIEFS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    return _normalize(resp.json().get("items", []))


def _load_snapshot() -> list[dict]:
    try:
        return json.loads(SNAPSHOT_FILE.read_text())
    except (OSError, ValueError):
        return []


def get_cached_legacy_briefs() -> list[dict]:
    if _legacy_briefs_cache["data"] is not None:
        return _legacy_briefs_cache["data"]
    try:
        live = fetch_legacy_briefs()
    except Exception:
        live = []
    # The endpoint is drained as of 2026-09 (returns []). Fall back to the
    # committed snapshot -- but still prefer the live feed if it ever comes
    # back, in case the archive is ever restored or extended upstream.
    _legacy_briefs_cache["data"] = live if live else _load_snapshot()
    return _legacy_briefs_cache["data"]
