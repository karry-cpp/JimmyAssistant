from __future__ import annotations

from pathlib import Path

from jimmy_assistant.tts.speaker import Speaker


class _DummyPyttsx:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[str] = []

    def speak(self, text: str, _lang: str = "en") -> bool:
        self.calls.append(text)
        return self.ok


class _DummyEdge:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[str] = []

    def speak(self, text: str, lang: str = "en") -> bool:
        self.calls.append(text)
        return self.ok


def test_speak_drop_pending_keeps_latest() -> None:
    s = Speaker(enabled=True)

    # Stop worker so queue state is deterministic for this unit test.
    s.stop()

    s.speak("first")
    s.speak("second")
    s.speak("latest", drop_pending=True)

    item = s._queue.get_nowait()  # noqa: SLF001
    assert item is not None
    text, _lang, _done = item
    assert text == "latest"


def test_low_latency_prefers_pyttsx_before_edge(tmp_path: Path) -> None:
    s = Speaker(enabled=True, low_latency=True, cache_dir=tmp_path)
    s._cache = {}  # noqa: SLF001
    py = _DummyPyttsx(ok=True)
    edge = _DummyEdge(ok=True)
    s._pyttsx3 = py  # noqa: SLF001
    s._edge = edge  # noqa: SLF001

    s._speak_one("Hello there", "en")  # noqa: SLF001

    assert py.calls == ["Hello there"]
    assert edge.calls == []
    s.stop()


def test_non_low_latency_prefers_edge_before_pyttsx(tmp_path: Path) -> None:
    s = Speaker(enabled=True, low_latency=False, cache_dir=tmp_path)
    s._cache = {}  # noqa: SLF001
    py = _DummyPyttsx(ok=True)
    edge = _DummyEdge(ok=True)
    s._pyttsx3 = py  # noqa: SLF001
    s._edge = edge  # noqa: SLF001

    s._speak_one("Hello there", "en")  # noqa: SLF001

    assert edge.calls == ["Hello there"]
    assert py.calls == []
    s.stop()
