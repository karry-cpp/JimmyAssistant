"""Jimmy orchestrator.

High-level control flow:

    ┌─────────────────────────────────────────────────────────────┐
    │  mic stream (16 kHz mono, 30 ms frames)                     │
    └───────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────▼──────────┐
        │ VoskWakeDetector      │  Vosk small-en-in model, "hey jimmy"
        └───────────┬──────────┘
                    │ (wake event)
        ┌───────────▼──────────┐
        │ VadCommandRecorder    │  webrtcvad, records until silence
        └───────────┬──────────┘
                    │ (PCM bytes)
        ┌───────────▼──────────┐
        │ WhisperSTT            │  faster-whisper large-v3 (Hindi + accent)
        └───────────┬──────────┘
                    │ (text, lang)
        ┌───────────▼──────────┐
        │ IntentRouter          │  rules → Ollama qwen2.5:3b JSON fallback
        └───────────┬──────────┘
                    │ (Intent)
        ┌───────────▼──────────┐
        │ Confirmation gate     │  spoken "yes/no" for destructive actions
        └───────────┬──────────┘
                    │ (Intent)
        ┌───────────▼──────────┐
        │ ActionRegistry        │  runs the whitelisted handler
        └───────────┬──────────┘
                    │ (ActionResult)
        ┌───────────▼──────────┐
        │ Speaker               │  TTS confirmation → back to waiting for wake
        └──────────────────────┘

If the user says everything in one breath ("hey jimmy lock the pc")
Vosk returns the whole utterance and, when the residual matches a rule,
we skip re-capture and Whisper for that turn. Anything not recognized
by rules is always re-captured via Whisper so accented English and
Hindi words are transcribed accurately.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from jimmy_assistant.actions import apps as apps_actions
from jimmy_assistant.actions import answer as answer_actions
from jimmy_assistant.actions import media as media_actions
from jimmy_assistant.actions import open_things as open_actions
from jimmy_assistant.actions import power as power_actions
from jimmy_assistant.actions import volume as volume_actions
from jimmy_assistant.actions import web as web_actions
from jimmy_assistant.actions import whatsapp as whatsapp_actions
from jimmy_assistant.actions import window_control as window_actions
from jimmy_assistant.actions import youtube as youtube_actions
from jimmy_assistant.actions.registry import ActionRegistry, ActionResult, ToolSchema
from jimmy_assistant.audio.mic import MicStream
from jimmy_assistant.audio.vad import VadCommandRecorder
from jimmy_assistant.audio.wake import VoskWakeDetector
from jimmy_assistant.config import Settings, load_settings
from jimmy_assistant.nlp import intent as A
from jimmy_assistant.nlp.agent import Agent
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.nlp.ollama_client import OllamaClient
from jimmy_assistant.nlp.ollama_launcher import find_ollama_exe, try_start_ollama
from jimmy_assistant.nlp.rules import RulesParser
from jimmy_assistant.stt.whisper_stt import WhisperSTT
from jimmy_assistant.tts.speaker import Speaker
from jimmy_assistant.ui.events import JimmyListener, NullListener
from jimmy_assistant.utils.logging import configure_logging
from jimmy_assistant.utils.text import classify_yes_no


logger = logging.getLogger("jimmy")


# ---------------------------------------------------------------------------
# Action wiring
# ---------------------------------------------------------------------------
def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


_EMPTY = _obj({})


def _build_registry(settings: Settings) -> ActionRegistry:
    registry = ActionRegistry()

    # Generic direct answer / reasoning
    registry.register(
        A.ACTION_ANSWER_DIRECT,
        answer_actions.answer_direct,
        schema=ToolSchema(
            description=(
                "Answer a generic user question directly without opening apps or the web. "
                "Use this for arithmetic, date differences, definitions, short explanations, "
                "and factual reasoning that you can solve from the user's prompt."
            ),
            parameters=_obj(
                {
                    "answer": {
                        "type": "string",
                        "description": "Concise answer to speak to the user. Include the calculation if helpful.",
                    }
                },
                required=["answer"],
            ),
        ),
    )

    # Power (destructive → require verbal confirmation)
    registry.register(
        A.ACTION_HIBERNATE,
        power_actions.hibernate,
        destructive=True,
        confirm_en="Should I hibernate the pc?",
        confirm_hi="PC hibernate karun?",
        schema=ToolSchema(
            description="Hibernate the PC (destructive — user will be asked to confirm).",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_SHUTDOWN,
        power_actions.shutdown,
        destructive=True,
        confirm_en="Should I shut down the pc?",
        confirm_hi="PC band karun?",
        schema=ToolSchema(
            description="Shut down the PC (destructive — user will be asked to confirm).",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_RESTART,
        power_actions.restart,
        destructive=True,
        confirm_en="Should I restart the pc?",
        confirm_hi="PC restart karun?",
        schema=ToolSchema(
            description="Restart the PC (destructive — user will be asked to confirm).",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_SLEEP,
        power_actions.sleep,
        schema=ToolSchema(description="Put the PC to sleep.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_LOCK,
        power_actions.lock,
        schema=ToolSchema(description="Lock the Windows session.", parameters=_EMPTY),
    )

    # Apps
    registry.register(
        A.ACTION_APP_LAUNCH,
        apps_actions.launch_app,
        schema=ToolSchema(
            description="Launch a Windows app by name (chrome, notepad, vs code, etc.).",
            parameters=_obj(
                {"app": {"type": "string", "description": "Application name."}},
                required=["app"],
            ),
        ),
    )
    registry.register(
        A.ACTION_APP_CLOSE,
        apps_actions.close_app,
        schema=ToolSchema(
            description="Close a running Windows app by name (for example: edge, chrome, notepad).",
            parameters=_obj(
                {"app": {"type": "string", "description": "Application name to close."}},
                required=["app"],
            ),
        ),
    )

    # Volume
    registry.register(
        A.ACTION_VOLUME_UP,
        volume_actions.volume_up,
        schema=ToolSchema(description="Turn the master volume up one step.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_DOWN,
        volume_actions.volume_down,
        schema=ToolSchema(description="Turn the master volume down one step.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_MUTE,
        volume_actions.volume_mute,
        schema=ToolSchema(description="Mute the master volume.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_UNMUTE,
        volume_actions.volume_unmute,
        schema=ToolSchema(description="Unmute the master volume.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_VOLUME_SET,
        volume_actions.volume_set,
        schema=ToolSchema(
            description="Set master volume to a specific level (0-100).",
            parameters=_obj(
                {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
                required=["level"],
            ),
        ),
    )

    # Media transport
    registry.register(
        A.ACTION_MEDIA_PLAY_PAUSE,
        media_actions.play_pause,
        schema=ToolSchema(description="Toggle play/pause on the active media player.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_MEDIA_NEXT,
        media_actions.next_track,
        schema=ToolSchema(description="Skip to the next track.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_MEDIA_PREV,
        media_actions.prev_track,
        schema=ToolSchema(description="Go to the previous track.", parameters=_EMPTY),
    )
    registry.register(
        A.ACTION_MEDIA_STOP,
        media_actions.stop,
        schema=ToolSchema(description="Stop media playback.", parameters=_EMPTY),
    )

    # Open / Web / YouTube
    registry.register(
        A.ACTION_OPEN_THING,
        open_actions.open_thing,
        schema=ToolSchema(
            description="Open a URL, local file, or folder in its default handler.",
            parameters=_obj(
                {"target": {"type": "string", "description": "URL, absolute file path, or folder path."}},
                required=["target"],
            ),
        ),
    )
    registry.register(
        A.ACTION_WEB_SEARCH,
        web_actions.web_search,
        schema=ToolSchema(
            description="Open a Google search for the given query in the default browser.",
            parameters=_obj(
                {"query": {"type": "string", "description": "Search query."}},
                required=["query"],
            ),
        ),
    )
    registry.register(
        A.ACTION_WEB_CURRENT_TIME,
        web_actions.current_time,
        schema=ToolSchema(
            description=(
                "Get the current clock time for a place or IANA timezone "
                "(for example: Indianapolis, Indiana or America/Indiana/Indianapolis)."
            ),
            parameters=_obj(
                {
                    "place": {
                        "type": "string",
                        "description": "City/state/country name or IANA timezone.",
                    }
                },
                required=["place"],
            ),
        ),
    )
    registry.register(
        A.ACTION_YOUTUBE_PLAY,
        youtube_actions.play_on_youtube,
        schema=ToolSchema(
            description=(
                "Search YouTube for the given query and play the first result "
                "in the default browser. Keep the query in the user's original "
                "language (do NOT translate Hindi titles)."
            ),
            parameters=_obj(
                {
                    "query": {
                        "type": "string",
                        "description": "Song or video title, verbatim in the user's language.",
                    }
                },
                required=["query"],
            ),
        ),
    )
    registry.register(
        A.ACTION_WINDOW_INFO,
        window_actions.active_window_info,
        schema=ToolSchema(
            description="Return the title of the currently focused desktop window.",
            parameters=_EMPTY,
        ),
    )
    registry.register(
        A.ACTION_WINDOW_CONTROL,
        window_actions.control_window,
        schema=ToolSchema(
            description=(
                "Control the currently focused desktop window. "
                "Allowed operations: minimize, maximize, restore, close."
            ),
            parameters=_obj(
                {
                    "operation": {
                        "type": "string",
                        "description": "One of: minimize, maximize, restore, close.",
                    }
                },
                required=["operation"],
            ),
        ),
    )
    registry.register(
        A.ACTION_WHATSAPP_SEND,
        whatsapp_actions.build_sender(
            contacts=settings.whatsapp_contacts,
            auto_send=settings.whatsapp_auto_send,
        ),
        schema=ToolSchema(
            description=(
                "Send a WhatsApp message to a saved contact by name. "
                "Use this when the user asks to message someone on WhatsApp."
            ),
            parameters=_obj(
                {
                    "contact": {"type": "string", "description": "Contact name as spoken by the user."},
                    "message": {"type": "string", "description": "Message text to send."},
                },
                required=["contact", "message"],
            ),
        ),
    )

    return registry


# ---------------------------------------------------------------------------
# Jimmy — the running assistant object
# ---------------------------------------------------------------------------
class Jimmy:
    def __init__(
        self,
        settings: Settings,
        listener: JimmyListener | None = None,
    ) -> None:
        self._settings = settings
        self._stop = threading.Event()
        self._listener: JimmyListener = listener or NullListener()

        self._mic = MicStream(
            sample_rate=settings.sample_rate,
            device_index=settings.mic_device_index,
        )
        self._wake = VoskWakeDetector(
            model_path=str(settings.resolve_path(settings.vosk_model_path)),
            sample_rate=settings.sample_rate,
            wake_phrases=settings.wake_phrases,
            fuzzy_threshold=settings.wake_fuzzy_threshold,
        )
        self._vad = VadCommandRecorder(
            sample_rate=settings.sample_rate,
            aggressiveness=settings.vad_aggressiveness,
            silence_ms=settings.command_silence_ms,
            max_seconds=settings.command_max_seconds,
        )

        self._stt = WhisperSTT(
            model_name=settings.whisper_model,
            compute_type=settings.whisper_compute_type,
            device=settings.whisper_device,
            language=settings.whisper_language or None,
            download_root=str(settings.resolve_path(settings.whisper_model_dir)),
        )
        self._llm: OllamaClient | None = None
        if settings.ollama_enabled:
            self._llm = OllamaClient(
                url=settings.ollama_url,
                model=settings.ollama_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            )
        self._registry = _build_registry(settings)
        self._rules = RulesParser()
        self._agent: Agent | None = None  # built lazily after mic is open
        self._speaker = Speaker(
            enabled=settings.tts_enabled,
            voice_en=settings.tts_voice_en,
            voice_hi=settings.tts_voice_hi,
            rate=settings.tts_rate,
            low_latency=settings.tts_low_latency,
            cache_dir=settings.resolve_path("models/tts_cache"),
        )

    # -- lifecycle ------------------------------------------------------
    def request_stop(self) -> None:
        logger.info("Stop requested")
        self._stop.set()

    def _ensure_ollama_ready(self) -> None:
        """Probe Ollama; if unreachable, try to auto-start ``ollama serve``.

        Called once at startup. Prints a clear status line either way,
        so the transcript window / console shows exactly what happened.
        """
        assert self._llm is not None
        url = self._settings.ollama_url
        if self._llm.is_reachable():
            print(f"[jimmy] Ollama reachable at {url} (model={self._settings.ollama_model})")
            self._listener.on_status(f"Ollama connected ({self._settings.ollama_model})")
            return

        exe = find_ollama_exe()
        if exe is None:
            print(f"[jimmy] WARNING: Ollama not reachable at {url} and ollama.exe not found — rules-only mode")
            self._listener.on_error(
                "Ollama not installed or unreachable — install from https://ollama.com/download"
            )
            return

        print(f"[jimmy] Ollama not reachable — auto-starting `{exe.name} serve`...")
        self._listener.on_status("starting Ollama server...")
        try_start_ollama()
        # Poll for readiness for up to ~10 seconds.
        for _ in range(20):
            if self._stop.is_set():
                return
            if self._llm.is_reachable():
                print(f"[jimmy] Ollama online at {url} (model={self._settings.ollama_model})")
                self._listener.on_status(f"Ollama connected ({self._settings.ollama_model})")
                return
            time.sleep(0.5)

        print(f"[jimmy] WARNING: Ollama still not reachable at {url} — will retry on each command")
        self._listener.on_error(
            f"Ollama did not come online at {url} — Jimmy will retry per command"
        )

    def run(self) -> None:
        logger.info("Jimmy starting")
        self._listener.on_status("starting")
        # Open the mic FIRST so the Windows recording-in-use indicator
        # lights up immediately — that gives visible confirmation the app
        # is alive during long first-run downloads. Wake detection only
        # begins after Whisper is ready.
        with self._mic:
            print("[jimmy] microphone open (check the taskbar mic icon)")
            self._listener.on_status("microphone open")
            logger.info("Microphone opened; loading Whisper model...")
            print("[jimmy] loading Whisper model — first run may download ~1.6 GB, please wait...")
            self._listener.on_status("loading Whisper model...")
            try:
                self._stt._ensure_loaded()  # noqa: SLF001
                print("[jimmy] Whisper model ready")
                self._listener.on_status("Whisper ready")
            except Exception:  # noqa: BLE001
                logger.exception("Whisper preload failed; commands will not work")
                print("[jimmy] WARNING: Whisper failed to load — see log for details")
                self._listener.on_error("Whisper failed to load")

            # Warm-up: run a tiny transcription now so the first real
            # command doesn't pay cuBLAS JIT / kernel-compile latency.
            try:
                self._listener.on_status("warming Whisper...")
                self._stt.warm_up()
            except Exception:  # noqa: BLE001
                logger.exception("Whisper warm-up failed (non-fatal)")

            # Pre-synthesise canned prompts ("Yes?", "Okay, cancelled.",
            # etc.) so playback is instant on subsequent wakes.
            try:
                self._listener.on_status("preparing TTS cache...")
                self._speaker.preload_phrase_cache()
            except Exception:  # noqa: BLE001
                logger.exception("TTS cache preload failed (non-fatal)")

            # Verify Ollama at startup. If it's installed but not
            # currently running, auto-launch `ollama serve` — Ollama
            # doesn't stay up between logins by default, so this saves
            # the user from starting it manually.
            if self._llm is not None:
                self._ensure_ollama_ready()
                # Build the agent regardless of reachability — the
                # OllamaClient handles per-request failures gracefully,
                # so Jimmy can still reconnect if Ollama comes online
                # later in the session.
                self._agent = Agent(
                    registry=self._registry,
                    llm=self._llm,
                    confirm_cb=self._confirm_intent,
                    listener=self._listener,
                )

            self._mic.drain()  # discard audio buffered during model load
            print(f"[jimmy] listening. Say '{self._settings.wake_phrases[0]}'.")
            self._listener.on_status(f"listening for '{self._settings.wake_phrases[0]}'")
            self._speaker.speak("Jimmy is ready.", lang="en")

            while not self._stop.is_set():
                event = self._wake.wait_for_wake(self._mic, self._stop)
                if event is None:
                    break
                self._handle_wake(event.residual_text)
                self._listener.on_status(f"listening for '{self._settings.wake_phrases[0]}'")

        logger.info("Jimmy stopped")
        print("[jimmy] stopped")
        self._listener.on_status("stopped")
        try:
            self._speaker.stop()
        except Exception:  # noqa: BLE001
            pass

    # -- one turn -------------------------------------------------------
    def _handle_wake(self, residual: str) -> None:
        self._listener.on_status("wake detected")
        # Fast path: rules match on the residual (English/ASCII only).
        # We do NOT publish the residual as "heard" until we've actually
        # matched a rule — otherwise Vosk's noisy transcription of the
        # wake phrase itself (e.g. "gary") shows up as a fake user line
        # in the transcript window.
        if residual:
            fast_intent = self._rules.parse(residual)
            if fast_intent is not None and not fast_intent.is_unknown:
                self._listener.on_heard(residual, "en")
                self._speaker.speak("Yes.", lang="en")
                self._execute_intent_directly(fast_intent, lang_hint="en")
                return

        # Prompt + capture command via VAD → Whisper → agent.
        # Use a short chime instead of TTS "Yes?" — instant, no ~1 s
        # audio playback latency between wake and being ready to hear.
        self._speaker.chime()
        self._listener.on_status("listening for command...")
        self._mic.drain()
        pcm = self._vad.record(self._mic, self._stop)
        if not pcm:
            logger.info("No command speech captured")
            self._listener.on_status("no speech captured")
            return

        self._listener.on_status("transcribing...")
        transcript = self._stt.transcribe_pcm(pcm, sample_rate=self._settings.sample_rate)
        if not transcript.text.strip():
            self._speaker.speak("I didn't catch that.", lang="en")
            self._listener.on_error("didn't catch that")
            return
        self._listener.on_heard(transcript.text, transcript.language)

        # Rules on the full transcript (still a fast-path — skips LLM).
        rule_intent = self._rules.parse(transcript.text)
        if rule_intent is not None and not rule_intent.is_unknown:
            print(f"[jimmy] rules matched: {rule_intent.name}")
            self._execute_intent_directly(rule_intent, lang_hint=transcript.language)
            return

        # Agentic path.
        if self._agent is None:
            # Agent isn't built (Ollama was disabled in config). Reply
            # with a clear diagnostic instead of a generic error.
            msg = "AI backend disabled in config"
            self._speaker.speak(
                "Sorry, my AI backend is disabled. I only recognise fixed commands right now.",
                lang="en",
            )
            self._listener.on_error(msg)
            return

        # Check reachability just before we spend LLM time. If Ollama
        # went away since startup, try one auto-reconnect.
        if self._llm is not None and not self._llm.is_reachable():
            self._listener.on_status("reconnecting to Ollama...")
            self._ensure_ollama_ready()
            if not self._llm.is_reachable():
                self._speaker.speak(
                    "Sorry, my AI backend is offline. Try running ollama serve.",
                    lang="en",
                )
                self._listener.on_error(
                    f"Ollama not reachable at {self._settings.ollama_url}"
                )
                return

        print(f"[jimmy] agent handling: {transcript.text!r}")
        self._listener.on_status("thinking...")
        context_text = transcript.text
        try:
            active = window_actions.active_window_info(Intent(name=A.ACTION_WINDOW_INFO))
            if active.ok and active.speak_en:
                # Give the LLM lightweight UI context for pronouns like
                # "this window" while preserving the original user text.
                context_text = (
                    f"Context: {active.speak_en}\n"
                    f"User request: {transcript.text}"
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to fetch active window context")

        outcome = self._agent.run(context_text)
        speech = self._agent.final_utterance(outcome, lang=transcript.language)
        if speech:
            self._listener.on_response(speech)
            self._speaker.speak(speech, lang=transcript.language, drop_pending=True)

    # -- direct dispatch (rules fast-path) ------------------------------
    def _execute_intent_directly(self, intent: Intent, lang_hint: str = "en") -> None:
        if intent.is_unknown:
            self._speaker.speak("Sorry, I didn't understand.", lang="en")
            self._listener.on_error("didn't understand")
            return
        if intent.is_cancel:
            self._speaker.speak("Okay, cancelled.", lang="en")
            self._listener.on_response("cancelled")
            return

        self._listener.on_tool_call(intent.name, dict(intent.slots))

        if (
            self._settings.confirm_destructive
            and self._registry.is_destructive(intent.name)
        ):
            if not self._confirm_intent(intent, lang_hint=lang_hint):
                self._speaker.speak("Okay, cancelled.", lang="en")
                self._listener.on_tool_result(intent.name, False, "user did not confirm")
                return

        result = self._registry.dispatch(intent)
        summary = result.speak_en if result.ok else result.error
        self._listener.on_tool_result(intent.name, result.ok, summary or "")
        self._speak_result(result, lang_hint)

    # -- confirmation callback used by BOTH direct-dispatch AND Agent --
    def _confirm_intent(self, intent: Intent, lang_hint: str = "en") -> bool:
        prompt = self._registry.confirm_prompt(intent.name, lang=lang_hint)
        logger.info("Confirmation prompt: %s", prompt)
        self._listener.on_confirm_prompt(intent.name, prompt)
        self._speaker.speak_and_wait(prompt, lang=lang_hint, timeout=6.0)

        self._mic.drain()
        pcm = self._vad.record(self._mic, self._stop)
        if not pcm:
            logger.info("No response — treating as no")
            self._listener.on_confirm_answer("(no answer)")
            return False
        answer = self._stt.transcribe_pcm(pcm, sample_rate=self._settings.sample_rate)
        verdict = classify_yes_no(answer.text)
        logger.info("Confirmation response %r → %s", answer.text, verdict)
        self._listener.on_confirm_answer(f"{answer.text!r} → {verdict}")
        return verdict == "yes"

    def _speak_result(self, result: ActionResult, lang_hint: str) -> None:
        if not self._settings.tts_enabled:
            return
        if result.ok:
            phrase = (
                result.speak_hi if (lang_hint == "hi" and result.speak_hi) else result.speak_en
            )
            if phrase:
                lang = "hi" if (lang_hint == "hi" and result.speak_hi) else "en"
                self._listener.on_response(phrase)
                self._speaker.speak(phrase, lang=lang, drop_pending=True)
        else:
            msg = f"Sorry, that didn't work: {result.error}"
            self._listener.on_error(result.error or "unknown error")
            self._speaker.speak(msg, lang="en", drop_pending=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    print("[jimmy] starting…  (Ctrl+C to quit)")
    print(f"[jimmy] whisper_model={settings.whisper_model}  ollama_model={settings.ollama_model}")
    logger.info("Loaded settings for Jimmy")

    jimmy = Jimmy(settings)

    def _signal_handler(signum: int, frame: object) -> None:
        del signum, frame
        jimmy.request_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (AttributeError, ValueError):
        pass

    jimmy.run()


if __name__ == "__main__":
    run()
