"""Simple web-search action: opens the default browser at Google."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import logging
import re
import webbrowser
from urllib.parse import quote_plus

import httpx

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


_WORLD_TIME_API_BASE = "https://worldtimeapi.org/api/timezone"
_COMMON_TIMEZONES: dict[str, tuple[str, str]] = {
    "indianapolis": ("America/Indiana/Indianapolis", "Indianapolis, Indiana"),
    "indiana": ("America/Indiana/Indianapolis", "Indiana"),
    "paris": ("Europe/Paris", "Paris"),
    "berlin": ("Europe/Berlin", "Berlin"),
    "rome": ("Europe/Rome", "Rome"),
    "madrid": ("Europe/Madrid", "Madrid"),
    "amsterdam": ("Europe/Amsterdam", "Amsterdam"),
    "dubai": ("Asia/Dubai", "Dubai"),
    "singapore": ("Asia/Singapore", "Singapore"),
    "new york": ("America/New_York", "New York"),
    "london": ("Europe/London", "London"),
    "tokyo": ("Asia/Tokyo", "Tokyo"),
    "delhi": ("Asia/Kolkata", "Delhi"),
    "mumbai": ("Asia/Kolkata", "Mumbai"),
    "kolkata": ("Asia/Kolkata", "Kolkata"),
}


def _normalize_place(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\b(what\s+is\s+the\s+time|time|current|right\s+now|in)\b", " ", t)
    t = re.sub(r"[^a-z0-9/]+", " ", t)
    return " ".join(t.split())


@lru_cache(maxsize=1)
def _timezone_index() -> dict[str, str]:
    """Best-effort city/timezone lookup table from worldtimeapi."""
    try:
        resp = httpx.get(_WORLD_TIME_API_BASE, timeout=6.0)
        resp.raise_for_status()
        raw = resp.json()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch timezone index")
        return {}

    if not isinstance(raw, list):
        return {}

    out: dict[str, str] = {}
    for item in raw:
        tz = str(item).strip()
        if not tz or "/" not in tz:
            continue
        parts = tz.split("/")
        city = parts[-1].replace("_", " ")
        region = parts[-2].replace("_", " ") if len(parts) >= 2 else ""
        for key in {
            _normalize_place(city),
            _normalize_place(f"{region} {city}"),
            _normalize_place(tz),
        }:
            if key and key not in out:
                out[key] = tz
    return out


def _pick_timezone(place: str) -> tuple[str, str] | None:
    p = _normalize_place(place)
    if not p:
        return None
    for needle, mapping in _COMMON_TIMEZONES.items():
        if needle in p:
            return mapping
    if "/" in place:
        # Caller may pass an explicit IANA timezone, e.g. America/Los_Angeles.
        return place.strip(), place.strip()

    index = _timezone_index()
    if not index:
        return None
    if p in index:
        return index[p], place.strip().title()

    tokens = p.split()
    for size in (3, 2, 1):
        if len(tokens) < size:
            continue
        key = " ".join(tokens[-size:])
        if key in index:
            return index[key], place.strip().title()

    best = ""
    for key in index:
        if key and f" {key} " in f" {p} " and len(key) > len(best):
            best = key
    if best:
        return index[best], place.strip().title()
    return None


def _fetch_world_time(timezone: str) -> tuple[str, str] | None:
    url = f"{_WORLD_TIME_API_BASE}/{timezone}"
    try:
        resp = httpx.get(url, timeout=6.0)
        resp.raise_for_status()
        payload = resp.json()
        iso_dt = str(payload.get("datetime", "")).strip()
        if not iso_dt:
            return None
        dt = datetime.fromisoformat(iso_dt)
        pretty = dt.strftime("%I:%M %p")
        tz_abbr = str(payload.get("abbreviation", "")).strip()
        return pretty, tz_abbr
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch world time for timezone=%s", timezone)
        return None


def current_time(intent: Intent) -> ActionResult:
    """Return the current time for a place/timezone using worldtimeapi."""
    place = intent.slots.get("place", "").strip()
    if not place:
        return ActionResult.failure("empty place")

    tz = _pick_timezone(place)
    if not tz:
        return ActionResult.failure("unsupported place for live time lookup")
    timezone, label = tz

    resolved = _fetch_world_time(timezone)
    if not resolved:
        return ActionResult.failure(f"could not fetch current time for {label}")
    clock, tz_abbr = resolved
    suffix = f" ({tz_abbr})" if tz_abbr else ""
    return ActionResult.success(speak_en=f"The current time in {label} is {clock}{suffix}.")


def web_search(intent: Intent) -> ActionResult:
    query = intent.slots.get("query", "").strip()
    if not query:
        return ActionResult.failure("empty search query")
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    logger.info("Opening web search: %s", url)
    webbrowser.open(url, new=2)
    return ActionResult.success(
        speak_en=f"Searching for {query}.",
        speak_hi=f"{query} search kar rahi hoon.",
    )
