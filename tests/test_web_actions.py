from __future__ import annotations

from jimmy_assistant.actions import web
from jimmy_assistant.nlp.intent import Intent


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_current_time_indianapolis_success(monkeypatch) -> None:
    def fake_get(_url: str, timeout: float):  # noqa: ARG001
        return _Resp(
            {
                "datetime": "2026-08-02T14:08:00-04:00",
                "abbreviation": "EDT",
            }
        )

    monkeypatch.setattr(web.httpx, "get", fake_get)

    result = web.current_time(
        Intent(name="web.current_time", slots={"place": "Indianapolis, Indiana"})
    )

    assert result.ok is True
    assert "Indianapolis, Indiana" in result.speak_en
    assert "EDT" in result.speak_en


def test_current_time_unknown_place_fails() -> None:
    result = web.current_time(Intent(name="web.current_time", slots={"place": "Mars"}))
    assert result.ok is False
    assert "unsupported place" in result.error


def test_current_time_paris_success(monkeypatch) -> None:
    def fake_get(_url: str, timeout: float):  # noqa: ARG001
        return _Resp(
            {
                "datetime": "2026-08-02T20:40:00+02:00",
                "abbreviation": "CEST",
            }
        )

    monkeypatch.setattr(web.httpx, "get", fake_get)

    result = web.current_time(Intent(name="web.current_time", slots={"place": "Paris"}))

    assert result.ok is True
    assert "Paris" in result.speak_en
    assert "CEST" in result.speak_en


def test_current_time_uses_timezone_index(monkeypatch) -> None:
    web._timezone_index.cache_clear()  # noqa: SLF001

    class _ListResp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return ["Europe/Prague", "Europe/Paris"]

    class _TimeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"datetime": "2026-08-02T20:40:00+02:00", "abbreviation": "CEST"}

    def fake_get(url: str, timeout: float):  # noqa: ARG001
        if url.endswith("/api/timezone"):
            return _ListResp()
        return _TimeResp()

    monkeypatch.setattr(web.httpx, "get", fake_get)

    result = web.current_time(Intent(name="web.current_time", slots={"place": "Prague"}))

    assert result.ok is True
    assert "Prague" in result.speak_en


def test_web_search_opens_browser(monkeypatch) -> None:
    opened: list[str] = []

    def fake_open(url: str, new: int):  # noqa: ARG001
        opened.append(url)
        return True

    monkeypatch.setattr(web.webbrowser, "open", fake_open)

    result = web.web_search(Intent(name="web.search", slots={"query": "current time indianapolis"}))

    assert result.ok is True
    assert opened
    assert "google.com/search" in opened[0]
