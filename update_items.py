"""Server-side store for the "Bitcast Protocol Updates" banner shown in the
frontend (frontend/index.html's initUpdateBanner()).

These items used to be a hardcoded JS array in frontend/index.html, which
meant every new banner item -- even a pure copy/UI addition with no logic
change -- needed a code edit, a git commit, and a VPS deploy. Moved here so
they can be written and published straight from /admin instead.

Local-only, gitignored -- same treatment as stats.json/activity_log.jsonl,
runtime data rather than source. Seeded once from SEED_ITEMS (the original
hardcoded array, migrated verbatim) the first time this file doesn't exist
yet, so existing banner history isn't lost by this change. SEED_ITEMS is
never re-applied after that first read.
"""

import json
import time
import uuid
from pathlib import Path
from threading import Lock

UPDATE_ITEMS_FILE = Path(__file__).parent / "update_items.json"
_lock = Lock()

SEED_ITEMS = [
    {
        "id": "2026-08-28-featured-tweet-pin",
        "date": "2026-08-28",
        "title": "Featured pick locked early.",
        "text": "The tweet chosen for the featured bonus is now locked in about a day before the campaign closes, instead of possibly changing all the way to final scoring. A late view spike will no longer change who ends up featured.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/db164014df3bab30d12dc09ab6ed92c0f5002c51",
    },
    {
        "id": "2026-08-23-sponsor-negativity-restored",
        "date": "2026-08-23",
        "title": "Sponsor negativity rule restored.",
        "text": "Criticizing the sponsor is back to being an automatic fail by default, same as it's always been, undoing a brief window where it wasn't. A campaign creator can still choose to allow critical posts, but has to set that up explicitly when the campaign is created.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/edae8b5e4cb6c5bc896080a8c9ac0e14c9378421",
    },
    {
        "id": "2026-08-14-manual-tweet-submission",
        "date": "2026-08-14",
        "title": "Manual tweet submission on some campaigns.",
        "text": "Some campaigns no longer auto-detect your posts by scanning X. After publishing, you need to submit your tweet yourself (its link or ID) through Stitch3 for it to be counted and ranked.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/41552049b61f10fc7117e7c4c749aa7b597ff501",
    },
    {
        "id": "2026-08-18-campaign-eligibility-rank-overlap",
        "date": "2026-08-18",
        "title": "Campaign eligibility widened.",
        "text": "Your tweet now counts toward a campaign if you ranked in the top members list at any point during the campaign, not just at the exact moment you posted. A rank dip right after posting should no longer disqualify an otherwise-compliant tweet.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/ec4aed79fb1699c7dca06bfe1ab2833cadce57b7",
    },
    {
        "id": "2026-08-18-honest-reviews",
        "date": "2026-08-17",
        "title": "Honest reviews now allowed.",
        "text": "Some campaigns ask for genuine product/service reviews. Previously, sounding negative or critical could fail the check. Now sentiment doesn't matter. A real, substantive review passes even if it's critical, as long as it's actually about the product.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/afa0e749c39a71a872ef89cd6bdff2ce4d1b3dea",
    },
    {
        "id": "2026-08-18-rank-cutoffs",
        "date": "2026-08-17",
        "title": "Rank cutoffs on some campaigns.",
        "text": "Certain campaigns now cap how many creators can earn from them. If a campaign sets a member limit, only the highest-ranked creators (by influence) within that cap get counted, even if your tweet fully meets the brief.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/65f56004a0f14f9fa3e49bf8ff4859b66ff660f7",
    },
    {
        "id": "2026-08-18-deferred-scoring",
        "date": "2026-08-17",
        "title": "One bad tweet won't sink the batch.",
        "text": "If the validator hits a temporary hiccup checking one tweet during final scoring, it now sets that tweet aside and keeps going, instead of risking everyone else's scoring in the same run.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/c15f8a0c2b86a59889665e08f85c2503143249c6",
    },
    {
        "id": "2026-09-01-direct-submission-grace",
        "date": "2026-09-01",
        "title": "Direct submission grace window",
        "text": "Exclusive and direct campaigns now accept a tweet you already posted for a short window after new posts stop being accepted, as long as you were eligible when you posted it and the submission confirms before scoring locks in. This does not give you more time to post something new, only more time to submit one you already have.",
        "url": "https://github.com/bitcast-network/bitcast-x/commit/69ef77dcd06a57cf418120cbbb6990e23d370381",
    },
]


def _read() -> list[dict]:
    if not UPDATE_ITEMS_FILE.exists():
        _write(SEED_ITEMS)
        return list(SEED_ITEMS)
    try:
        return json.loads(UPDATE_ITEMS_FILE.read_text())
    except (OSError, ValueError):
        return []


def _write(items: list[dict]) -> None:
    try:
        UPDATE_ITEMS_FILE.write_text(json.dumps(items, indent=2))
    except OSError:
        pass  # Best-effort, same tradeoff as activity_log.py/stats.json.


def list_items() -> list[dict]:
    with _lock:
        return _read()


def add_item(title: str, text: str, url: str = "", date: str = "") -> dict:
    with _lock:
        items = _read()
        item_date = date or time.strftime("%Y-%m-%d", time.gmtime())
        item = {
            "id": f"{item_date}-{uuid.uuid4().hex[:8]}",
            "date": item_date,
            "title": title,
            "text": text,
            "url": url,
        }
        items.append(item)
        _write(items)
        return item


def delete_item(item_id: str) -> bool:
    with _lock:
        items = _read()
        remaining = [i for i in items if i["id"] != item_id]
        if len(remaining) == len(items):
            return False
        _write(remaining)
        return True
