"""Public TTS facade.

Speaks Jimmy's replies using **edge-tts** as the primary engine — its
Indian-English (``en-IN-NeerjaNeural``) and Hindi (``hi-IN-SwaraNeural``)
neural voices sound natural and consistent for both languages. Falls
back to offline **pyttsx3** (Windows SAPI5) if edge-tts fails or isn't
installed.

Threading model
---------------
Jimmy's orchestrator runs on a worker thread. TTS synthesis + playback
takes ~1–3 seconds, so a fire-and-forget worker with a bounded queue
avoids blocking the main flow. Use :meth:`speak` for background
announcements; use :meth:`speak_and_wait` when the next step depends
on the phrase being fully heard (confirmation prompts, "Yes?" before
opening the mic for a command).

Phrase cache
------------
Very short prompts we say all the time (``"Yes?"``, ``"Okay,
cancelled."``, ``"Sorry, I didn't understand."``) are synthesized
once at startup and stored as MP3s under ``models/tts_cache/``. On
subsequent calls we skip the ~1 s edge-tts round-trip and just play
the cached file, which lands in ~150 ms via Windows MCI.

Failure isolation: any exception in any backend is logged and swallowed.
TTS is a nicety; it must never crash Jimmy.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


# Phrases that get pre-synthesised at startup so their playback is
# instant instead of paying an edge-tts network round trip every wake.
_CACHED_PHRASES: Dict[str, str] = {
    "yes?":                        "en",
    "yes.":                        "en",
    "okay, cancelled.":            "en",
    "sorry, i didn't understand.": "en",
    "i didn't catch that.":        "en",
    "jimmy is ready.":             "en",
}


class Speaker:
    def __init__(
        self,
        enabled: bool = True,
        voice_en: str = "en-IN-NeerjaNeural",
        voice_hi: str = "hi-IN-SwaraNeural",
        rate: str = "+0%",
        low_latency: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._enabled = enabled
        self._voice_en = voice_en
        self._voice_hi = voice_hi
        self._rate = rate
        self._low_latency = low_latency
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: Dict[str, Path] = {}

        self._edge = None
        self._pyttsx3 = None

        # Worker queue. Each item is ``(text, lang, done_event | None)``.
        # A ``None`` sentinel item stops the worker on shutdown.
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run_worker,
            name="jimmy-tts",
            daemon=True,
        )
        self._worker.start()

    # -- engine lazy-init ------------------------------------------------
    def _get_edge(self):
        if self._edge is None:
            try:
                from jimmy_assistant.tts.edge_tts_engine import EdgeTtsEngine

                self._edge = EdgeTtsEngine(
                    voice_en=self._voice_en,
                    voice_hi=self._voice_hi,
                    rate=self._rate,
                )
            except Exception:  # noqa: BLE001
                logger.exception("edge-tts unavailable")
                self._edge = False  # sentinel: don't retry
        return self._edge or None

    def _get_pyttsx3(self):
        if self._pyttsx3 is None:
            try:
                from jimmy_assistant.tts.pyttsx3_engine import Pyttsx3Engine

                self._pyttsx3 = Pyttsx3Engine()
            except Exception:  # noqa: BLE001
                logger.exception("pyttsx3 unavailable")
                self._pyttsx3 = False
        return self._pyttsx3 or None

    # -- phrase cache ---------------------------------------------------
    def preload_phrase_cache(self) -> None:
        """Pre-synthesise the canned phrases into MP3s. No-op if
        ``cache_dir`` was not provided or edge-tts is unavailable."""
        if not self._enabled or self._cache_dir is None:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create TTS cache dir %s", self._cache_dir)
            return

        try:
            import edge_tts  # local import
        except Exception:  # noqa: BLE001
            logger.info("edge-tts unavailable; skipping phrase cache")
            return

        for phrase, lang in _CACHED_PHRASES.items():
            key = phrase.lower().strip()
            path = self._cache_dir / (self._filename_for(phrase, lang) + ".mp3")
            self._cache[key] = path
            if path.exists() and path.stat().st_size > 0:
                continue
            voice = self._voice_hi if lang == "hi" else self._voice_en
            try:
                asyncio.run(
                    edge_tts.Communicate(text=phrase, voice=voice, rate=self._rate)
                    .save(str(path))
                )
                logger.info("Cached TTS phrase %r -> %s", phrase, path.name)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cache TTS phrase %r", phrase)

    @staticmethod
    def _filename_for(phrase: str, lang: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in phrase.lower()).strip("_")
        return f"{lang}_{safe[:40]}"

    def _cached_playback(self, text: str) -> bool:
        """Return True if the phrase is cached and was played."""
        if self._cache_dir is None:
            return False
        key = text.lower().strip()
        path = self._cache.get(key)
        if path is None or not path.exists():
            return False
        try:
            from jimmy_assistant.tts.edge_tts_engine import _mci_play

            return bool(_mci_play(path))
        except Exception:  # noqa: BLE001
            logger.exception("Cached playback failed for %r", text)
            return False

    # -- public API -----------------------------------------------------
    def speak(self, text: str, lang: Optional[str] = None, drop_pending: bool = False) -> None:
        """Queue ``text`` for TTS. Returns immediately."""
        if not self._enabled or not text or not text.strip():
            return
        if drop_pending:
            self._clear_pending_queue()
        self._queue.put((text, lang or "en", None))

    def speak_and_wait(self, text: str, lang: Optional[str] = None, timeout: float = 15.0) -> None:
        """Queue ``text`` and block until it has finished playing (or
        the timeout elapses)."""
        if not self._enabled or not text or not text.strip():
            return
        done = threading.Event()
        self._queue.put((text, lang or "en", done))
        done.wait(timeout=timeout)

    def stop(self) -> None:
        """Signal the worker to exit and wait briefly for it."""
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def _clear_pending_queue(self) -> None:
        """Drop queued (not-yet-playing) utterances.

        Useful when a fresh response should start immediately instead of
        waiting behind stale confirmations/status lines.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                # Preserve shutdown signal if it was queued.
                self._queue.put(None)
                break
            _, _, done = item
            if done is not None:
                done.set()

    # -- optional low-latency wake-ack ----------------------------------
    @staticmethod
    def chime() -> None:
        """Play a short non-blocking beep (~90 ms) as an instant wake
        acknowledgement. Uses ``winsound`` (stdlib) — no synthesis
        latency at all."""
        try:
            import sys
            import winsound

            if sys.platform.startswith("win"):
                # A soft, high-pitched two-tone pip: pleasant, unmistakable.
                winsound.Beep(1200, 45)
                winsound.Beep(1600, 45)
        except Exception:  # noqa: BLE001
            logger.debug("chime failed", exc_info=True)

    # -- worker ---------------------------------------------------------
    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                logger.debug("TTS worker exiting")
                return
            text, lang, done = item
            try:
                self._speak_one(text, lang)
            except Exception:  # noqa: BLE001
                logger.exception("TTS worker crash for %r", text)
            finally:
                if done is not None:
                    done.set()

    def _speak_one(self, text: str, lang: str) -> None:
        """Try phrase cache first (instant), then edge-tts (~1 s round
        trip, natural voice), then pyttsx3 (offline SAPI5 fallback).

        The ``lang`` argument is only a hint. The actual voice is
        selected from the *content* of the text — a Devanagari-heavy
        response gets the Hindi voice, everything else gets the
        English voice. This prevents the English reply "Playing X on
        YouTube" from being spoken with the Hindi voice just because
        the user's input happened to be Hindi.
        """
        effective_lang = "hi" if _has_devanagari(text) else "en"
        if self._cached_playback(text):
            return

        # For low-latency mode, prefer local SAPI first for immediate start.
        # Edge remains available as a quality fallback.
        if self._low_latency:
            pyttsx = self._get_pyttsx3()
            if pyttsx is not None:
                try:
                    if pyttsx.speak(text, _lang=effective_lang):
                        return
                except Exception:  # noqa: BLE001
                    logger.exception("pyttsx3 raised for %r", text)

            edge = self._get_edge()
            if edge is not None:
                try:
                    edge.speak(text, lang=effective_lang)
                except Exception:  # noqa: BLE001
                    logger.exception("edge-tts raised for %r", text)
            return

        edge = self._get_edge()
        if edge is not None:
            try:
                if edge.speak(text, lang=effective_lang):
                    return
            except Exception:  # noqa: BLE001
                logger.exception("edge-tts raised for %r", text)

        pyttsx = self._get_pyttsx3()
        if pyttsx is not None:
            try:
                pyttsx.speak(text, _lang=effective_lang)
            except Exception:  # noqa: BLE001
                logger.exception("pyttsx3 raised for %r", text)


def _has_devanagari(text: str) -> bool:
    """True if the string contains any Devanagari (Hindi) character."""
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return True
    return False
