"""Append-only local activity log, backing the /admin panel.

Records what handles get looked up on the Engagement Value tab and what
briefs get checked on the Tweet Validator tab, so the site owner can see
real usage patterns. Local-only, gitignored -- same treatment as
stats.json. IPs are never stored raw, only a SHA-256 hash (same scheme
stats.json already uses for its unique-creator counter), so a leaked log
can't be used to deanonymize visitors by IP.

Shared by main.py and engagement.py (rather than living in either) so
both can log without creating a circular import.
"""

import hashlib
import json
import time
from pathlib import Path
from threading import Lock

ACTIVITY_LOG_FILE = Path(__file__).parent / "activity_log.jsonl"
_lock = Lock()


def hash_ip(ip: str) -> str:
    return hashlib.sha256(f"stitch3-validator:{ip}".encode()).hexdigest()


def log_event(event_type: str, **fields) -> None:
    """Best-effort: a logging failure must never break the request that
    triggered it, so failures here are swallowed rather than raised."""
    entry = {"ts": time.time(), "type": event_type, **fields}
    try:
        with _lock, ACTIVITY_LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def read_events(limit: int = 500) -> list[dict]:
    """Most recent first, capped at `limit` so the admin page never grows
    unbounded even after months of runtime."""
    if not ACTIVITY_LOG_FILE.exists():
        return []
    try:
        lines = ACTIVITY_LOG_FILE.read_text().splitlines()
    except OSError:
        return []
    events = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    events.reverse()
    return events
