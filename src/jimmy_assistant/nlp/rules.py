"""Regex-based fast-path intent parser.

Handles common English and Hinglish (Roman-script Hindi) phrasings so we
don't pay LLM latency for the top-N commands. Anything not matched here
is escalated to the Ollama-backed router.

Design rules:
* Patterns operate on already-normalized text (lowercased, punctuation
  stripped, whitespace collapsed) but Devanagari characters survive
  normalization, so Hindi words used in commands (song names,
  Hinglish verbs) still match.
* Every branch returns exactly one :class:`Intent` or ``None``.
* Never crash on odd input; parsing failure means "no rule matched".
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Pattern, Tuple

from jimmy_assistant.nlp import intent as A
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.utils.text import normalize_text


logger = logging.getLogger(__name__)


# ---- Utility: compile once ------------------------------------------------
def _c(pattern: str) -> Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


# ---- Pattern tables -------------------------------------------------------
# Each row: (action_name, compiled_pattern, slot_extractor)
# ``slot_extractor`` receives the regex match; returns a dict of slot values.

def _no_slots(_m: "re.Match[str]") -> dict:
    return {}


def _query_slot(m: "re.Match[str]") -> dict:
    return {"query": m.group("query").strip()}


def _target_slot(m: "re.Match[str]") -> dict:
    return {"target": m.group("target").strip()}


def _app_slot(m: "re.Match[str]") -> dict:
    return {"app": m.group("app").strip()}


def _app_close_slot(m: "re.Match[str]") -> dict:
    app = m.group("app").strip()
    app = re.sub(r"\b(browser|app|application|window)\b$", "", app, flags=re.IGNORECASE).strip()
    return {"app": app}


def _level_slot(m: "re.Match[str]") -> dict:
    return {"level": m.group("level")}


def _wa_slot(m: "re.Match[str]") -> dict:
    contact = (m.groupdict().get("contact") or "").strip()
    message = (m.groupdict().get("message") or "").strip()
    return {"contact": contact, "message": message}


# --- Order matters: more specific patterns before generic ones. -----------
_PATTERNS: List[Tuple[str, Pattern[str], callable]] = [
    # ---- Cancel / stop --------------------------------------------------
    (A.ACTION_CANCEL, _c(r"^(cancel|stop|abort|never\s*mind|nevermind|ruko|ruk|rehne\s*do|rehne\s*de|chhod\s*do)$"), _no_slots),

    # ---- Power ----------------------------------------------------------
    (A.ACTION_HIBERNATE, _c(r"\b(hibernate|hybrid\s*sleep)\b(?:\s+(?:the\s+)?(pc|computer|laptop|system))?"), _no_slots),
    (A.ACTION_HIBERNATE, _c(r"\bpc\s+ko\s+hibernate\b"), _no_slots),
    (A.ACTION_HIBERNATE, _c(r"\b(system|pc|laptop)?\s*hibernate\s*(kar\s*(do|de)|karo)\b"), _no_slots),

    (A.ACTION_SHUTDOWN, _c(r"\b(shut\s*down|power\s*off|switch\s*off|turn\s*off)\b(?:\s+(?:the\s+)?(pc|computer|laptop|system))?"), _no_slots),
    (A.ACTION_SHUTDOWN, _c(r"\b(pc|computer|laptop|system)\s+(shut\s*down|band|bandh)\s*(kar\s*(do|de)|karo)?\b"), _no_slots),
    (A.ACTION_SHUTDOWN, _c(r"\b(pc|computer|laptop|system)\s+ko\s+bandh?\s*(kar\s*(do|de)|karo)\b"), _no_slots),

    (A.ACTION_RESTART, _c(r"\b(restart|reboot)\b(?:\s+(?:the\s+)?(pc|computer|laptop|system))?"), _no_slots),
    (A.ACTION_RESTART, _c(r"\b(pc|computer|laptop|system)\s+restart\s*(kar\s*(do|de)|karo)?\b"), _no_slots),

    (A.ACTION_SLEEP, _c(r"\b(go\s+to\s+sleep|sleep\s+mode|put\s+(?:the\s+)?(?:pc|computer|laptop)\s+to\s+sleep)\b"), _no_slots),
    (A.ACTION_SLEEP, _c(r"\bpc\s+ko\s+sula\s*(do|de)\b"), _no_slots),
    (A.ACTION_SLEEP, _c(r"^sleep$"), _no_slots),

    (A.ACTION_LOCK, _c(r"\block\b(?:\s+(?:the\s+)?(pc|computer|laptop|screen|system))?"), _no_slots),
    (A.ACTION_LOCK, _c(r"\b(pc|computer|laptop|screen|system)\s+lock\s*(kar\s*(do|de)|karo)?\b"), _no_slots),

    # ---- Volume ---------------------------------------------------------
    (A.ACTION_VOLUME_MUTE, _c(r"\b(mute|silence)\b"), _no_slots),
    (A.ACTION_VOLUME_MUTE, _c(r"\b(volume|awaaz|aawaz|awaj)\s+(band|bandh|mute)\s*(kar\s*(do|de)|karo)?\b"), _no_slots),
    (A.ACTION_VOLUME_UNMUTE, _c(r"\bunmute\b"), _no_slots),
    (A.ACTION_VOLUME_UNMUTE, _c(r"\b(volume|awaaz|aawaz)\s+(chalu|chalao|on)\s*(kar\s*(do|de)|karo)?\b"), _no_slots),

    (A.ACTION_VOLUME_SET, _c(r"\b(?:set\s+)?volume\s+(?:to\s+)?(?P<level>\d{1,3})\s*(?:percent|%)?\b"), _level_slot),
    (A.ACTION_VOLUME_SET, _c(r"\bvolume\s+(?P<level>\d{1,3})\s*(?:pe|par)\s*(kar\s*(do|de)|karo)?\b"), _level_slot),

    (A.ACTION_VOLUME_UP, _c(r"\b(volume\s+up|increase\s+(?:the\s+)?volume|louder|turn\s+it\s+up)\b"), _no_slots),
    (A.ACTION_VOLUME_UP, _c(r"\b(volume|awaaz|aawaz)\s+(badha|bada|tez|zyada)\s*(kar\s*(do|de)|karo)?\b"), _no_slots),

    (A.ACTION_VOLUME_DOWN, _c(r"\b(volume\s+down|decrease\s+(?:the\s+)?volume|quieter|turn\s+it\s+down)\b"), _no_slots),
    (A.ACTION_VOLUME_DOWN, _c(r"\b(volume|awaaz|aawaz)\s+(kam|kum|dheere|halka)\s*(kar\s*(do|de)|karo)?\b"), _no_slots),

    # ---- Media ---------------------------------------------------------
    (A.ACTION_MEDIA_PLAY_PAUSE, _c(r"^(play|pause|resume|toggle\s+play)$"), _no_slots),
    (A.ACTION_MEDIA_PLAY_PAUSE, _c(r"\b(pause|resume)\s+(the\s+)?(music|song|video)\b"), _no_slots),
    (A.ACTION_MEDIA_PLAY_PAUSE, _c(r"\b(gaana|song|music)\s+(rok|band|paus)\s*(kar\s*(do|de)|karo)?\b"), _no_slots),
    (A.ACTION_MEDIA_NEXT, _c(r"\b(next\s+(song|track|video)|skip(?:\s+(?:this|track))?)\b"), _no_slots),
    (A.ACTION_MEDIA_NEXT, _c(r"\b(agla|next)\s+(gaana|song|track)\b"), _no_slots),
    (A.ACTION_MEDIA_PREV, _c(r"\b(previous|prev|last)\s+(song|track|video)\b"), _no_slots),
    (A.ACTION_MEDIA_PREV, _c(r"\b(pichla|previous)\s+(gaana|song|track)\b"), _no_slots),
    (A.ACTION_MEDIA_STOP, _c(r"\bstop\s+(the\s+)?(music|song|video|playback)\b"), _no_slots),

    # ---- YouTube (very common — put ahead of generic "play") ------------
    (A.ACTION_YOUTUBE_PLAY, _c(r"^play\s+(?P<query>.+?)\s+on\s+youtube$"), _query_slot),
    (A.ACTION_YOUTUBE_PLAY, _c(r"^(?P<query>.+?)\s+(?:ko\s+)?youtube\s+(pe|par)\s+(chala|play|laga)\s*(do|de|deo|dena)?$"), _query_slot),
    (A.ACTION_YOUTUBE_PLAY, _c(r"^youtube\s+(pe|par)\s+(?P<query>.+?)\s+(chala|play|laga)\s*(do|de|deo|dena)?$"), _query_slot),
    (A.ACTION_YOUTUBE_PLAY, _c(r"^play\s+(?P<query>.+?)$"), _query_slot),
    (A.ACTION_YOUTUBE_PLAY, _c(r"^(?P<query>.+?)\s+(chala|laga)\s*(do|de|deo|dena)$"), _query_slot),

    # ---- Open path / folder / file (must precede generic "open <app>") --
    (A.ACTION_OPEN_THING, _c(r"^(?:open|show)\s+(?:the\s+)?folder\s+(?P<target>.+?)$"), _target_slot),
    (A.ACTION_OPEN_THING, _c(r"^(?:open|show)\s+(?:the\s+)?file\s+(?P<target>.+?)$"), _target_slot),

    # ---- WhatsApp send -------------------------------------------------
    (A.ACTION_WHATSAPP_SEND, _c(r"^(?:open\s+)?whatsapp\s+and\s+send\s+(?:a\s+)?message\s+to\s+(?P<contact>.+?)\s+(?:saying|that\s+says|with\s+message)\s+(?P<message>.+)$"), _wa_slot),
    (A.ACTION_WHATSAPP_SEND, _c(r"^send\s+(?:a\s+)?whatsapp\s+message\s+to\s+(?P<contact>.+?)\s+(?:saying|that\s+says|with\s+message)\s+(?P<message>.+)$"), _wa_slot),

    # ---- Launch apps ---------------------------------------------------
    (A.ACTION_APP_CLOSE, _c(r"^(?:close|quit|exit|terminate|kill)\s+(?P<app>.+?)$"), _app_close_slot),
    (A.ACTION_APP_CLOSE, _c(r"^(?P<app>.+?)\s+(?:app|application|browser)?\s*(?:close|quit|exit|band|bandh)\s*(?:kar\s*(do|de)|karo)?$"), _app_close_slot),
    (A.ACTION_APP_LAUNCH, _c(r"^(?:open|launch|start)\s+(?P<app>.+?)$"), _app_slot),
    (A.ACTION_APP_LAUNCH, _c(r"^(?P<app>.+?)\s+(kholo|khol\s*do|chalao|start\s*karo)$"), _app_slot),

    # ---- Web search -----------------------------------------------------
    (A.ACTION_WEB_SEARCH, _c(r"^(?:search|google|search\s+for)\s+(?P<query>.+?)$"), _query_slot),
    (A.ACTION_WEB_SEARCH, _c(r"^(?P<query>.+?)\s+(google|search)\s+(kar\s*(do|de)|karo)$"), _query_slot),
]


# ---- Parser ---------------------------------------------------------------
class RulesParser:
    """Applies the pattern table in order and returns the first match."""

    def parse(self, transcript: str) -> Optional[Intent]:
        text = normalize_text(transcript)
        if not text:
            return None

        for action, pattern, extract in _PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            try:
                slots = extract(match)
            except Exception:  # noqa: BLE001
                logger.exception("Slot extractor raised for pattern %r", pattern.pattern)
                continue
            # Reject empty required slots so we can fall back to the LLM.
            if action in {
                A.ACTION_YOUTUBE_PLAY,
                A.ACTION_WEB_SEARCH,
                A.ACTION_APP_LAUNCH,
                A.ACTION_APP_CLOSE,
                A.ACTION_OPEN_THING,
                A.ACTION_WHATSAPP_SEND,
            } and not any(slots.values()):
                continue
            logger.info("Rules matched %s slots=%s", action, slots)
            return Intent(name=action, slots=slots, source="rules", raw_text=transcript)
        return None
