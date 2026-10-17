"""Brief evaluation via JEV on OpenRouter, mirroring bitcast-x 3.1.0.

Copied from upstream `src/bitcast_x/brief_filter.py` (commit f77b24e): the
thresholds, the code-level required-tag/link check, the plain-language failure
reasons and the response validation are the validators' own, so a verdict here
matches what they would produce. Only the transport differs -- `requests` with
a small in-memory verdict cache instead of httpx and their durable store. Re-pull
from upstream rather than tuning the numbers locally: they were validated
offline together with the request text in `prompts.py`.
"""

import hashlib
import json
import math
import os
import re
import time
from threading import Lock
from typing import Any, Optional

import requests

from prompts import build_request

JEV_API_URL = "https://openrouter.ai/api/v1/systemone"
# OpenRouter can return the canonical slug for the pinned Jev 1.13 model.
JEV_CANONICAL_MODEL = "typesafe/jev-1.13-20260917"
VERDICT_ACCEPT_MIN = 0.02
GATE_MIN = 0.6
IDENTITY_MISMATCH_MAX = 0.5
TWEET_MAX_LENGTH = 10_000
REQUEST_TIMEOUT = 30
REQUEST_ATTEMPTS = 3

_GATE_FAILURE_REASONS = {
    "nonnegative": (
        "The post did not meet this campaign's requirement for a positive or neutral tone "
        "toward the sponsor or product."
    ),
    "focus80": (
        "The post does not focus enough on the sponsor or its topic. "
        "This campaign requires at least 80% of the content to stay on topic."
    ),
    "primary": (
        "The post needs to focus mainly on the product or service, "
        "or a comparison with alternatives."
    ),
    "substance": (
        "The review needs a specific assessment supported by a reason, example, "
        "product feature, result, or experience."
    ),
}

_HANDLE_INSTRUCTION = re.compile(
    r"\b(?:also\s+)?(?:tag|mention|tagging|mentioning)\s*:?\s*"
    r"((?:@\w+(?:\s*(?:,|/|&|\band\b)\s*)?)+)",
    re.IGNORECASE,
)
_LINK_INSTRUCTION = re.compile(r"\binclude\b[^.\n]*?\b(?:ref(?:erral)?\s+)?link\b", re.IGNORECASE)
_URL = re.compile(
    r"https?://\S+|\b[\w-]+\.(?:io|com|xyz|ai|so|fun|app|network|org|net|dev)(?:/\S*)?",
    re.IGNORECASE,
)


def missing_required_items(brief: str, post: str) -> list[str]:
    """Return handles or links the brief explicitly requires that the post lacks.

    Brand handles in a header are not requirements, and quote-post instructions are ignored
    because quote metadata is not captured. Any URL satisfies a required link because X
    shortens links and the destination is not captured.
    """

    text = post.lower()
    handles: list[str] = []
    for match in _HANDLE_INSTRUCTION.finditer(brief):
        for handle in re.findall(r"@\w+", match.group(1)):
            if handle.lower() not in handles:
                handles.append(handle.lower())
    missing = [handle for handle in handles if handle not in text]
    if _LINK_INSTRUCTION.search(brief) and not _URL.search(post):
        missing.append("link")
    return missing


def decide(answers: dict[str, Any], brief: str, post: str) -> dict[str, Any]:
    """Apply the validated thresholds to a JEV response and the code checks.

    Returns {"meets_brief": bool, "reasons": [str], "breakdown": {...}}.
    """

    accept = float(answers["verdict"]["probabilities"]["ACCEPT"])
    mismatch = float(answers["identity"]["probabilities"]["MISMATCH"])
    gates = {
        name.removeprefix("gate_"): float(answer["noul"])
        for name, answer in answers.items()
        if name.startswith("gate_")
    }
    missing = missing_required_items(brief, post)
    failures: list[str] = []
    if mismatch >= IDENTITY_MISMATCH_MAX:
        failures.append(
            "The post appears to describe a different product or service "
            "from the one in the campaign brief."
        )
    failures.extend(
        _GATE_FAILURE_REASONS[name] for name, value in gates.items() if value < GATE_MIN
    )
    missing_handles = [item for item in missing if item != "link"]
    if missing_handles:
        failures.append(
            "The post is missing required mentions: " + ", ".join(missing_handles) + "."
        )
    if "link" in missing:
        failures.append("The post is missing a link required by the campaign brief.")
    if accept < VERDICT_ACCEPT_MIN:
        reason = "The post did not pass the overall brief check."
        if not failures:
            reason += " A specific unmet requirement was not identified."
        failures.insert(0, reason)
    return {
        "meets_brief": not failures,
        "reasons": failures,
        "breakdown": {
            "verdict_accept": accept,
            "identity_mismatch": mismatch,
            "gates": gates,
            "missing_required": missing,
        },
    }


def _validated_answers(payload: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if payload.get("model") not in (request["model"], JEV_CANONICAL_MODEL):
        raise ValueError("unexpected JEV model")
    answers = payload["answers"]
    if not isinstance(answers, dict) or set(answers) != set(request["questions"]):
        raise ValueError("JEV answers do not match the request")
    values = [
        answers["verdict"]["probabilities"]["ACCEPT"],
        answers["identity"]["probabilities"]["MISMATCH"],
    ]
    values += [answers[name]["noul"] for name in answers if name.startswith("gate_")]
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or not 0 <= v <= 1
        for v in values
    ):
        raise ValueError("invalid JEV probability")
    return answers


# JEV is deterministic for a given request (the validators cache verdicts by
# request hash for the same reason), so repeat checks of an unchanged draft
# are served from memory and cost nothing. In-memory and per-process, same
# tradeoff as the other caches in this app.
_CACHE_TTL = 3600
_CACHE_MAX = 500
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = Lock()


def _call_jev(request: dict[str, Any]) -> dict[str, Any]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            resp = requests.post(JEV_API_URL, headers=headers, json=request, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return _validated_answers(resp.json(), request)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < REQUEST_ATTEMPTS:
                # Longer wait on a 429 -- a 1-2s retry just gets rejected again.
                limited = (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code == 429
                )
                time.sleep(8 if limited else 2 ** (attempt + 1))
    assert last_error is not None
    raise last_error


def _prepare(brief: dict, tweet: str) -> tuple[dict[str, Any], str, str, str]:
    text = tweet[:TWEET_MAX_LENGTH]
    brief_text = brief["brief"]
    request = build_request(brief_text, brief.get("prompt_version", 1), text)
    request_hash = hashlib.sha256(("jev:" + json.dumps(request, sort_keys=True)).encode()).hexdigest()
    return request, request_hash, brief_text, text


def _cached(request_hash: str) -> Optional[dict[str, Any]]:
    with _cache_lock:
        hit = _cache.get(request_hash)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1]
    return None


def cached_result(brief: dict, tweet: str) -> Optional[dict[str, Any]]:
    """The stored verdict for this exact brief + draft, or None. Lets the
    caller skip rate limiting for checks that cost nothing."""

    return _cached(_prepare(brief, tweet)[1])


def evaluate_brief(brief: dict, tweet: str) -> dict[str, Any]:
    """Run one JEV evaluation of `tweet` against `brief` (a normalized brief dict)."""

    request, request_hash, brief_text, text = _prepare(brief, tweet)
    hit = _cached(request_hash)
    if hit is not None:
        return hit

    result = decide(_call_jev(request), brief_text, text)

    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            for stale in sorted(_cache, key=lambda k: _cache[k][0])[: _CACHE_MAX // 5]:
                del _cache[stale]
        _cache[request_hash] = (time.time(), result)
    return result
