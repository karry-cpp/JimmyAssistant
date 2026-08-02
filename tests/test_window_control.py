from __future__ import annotations

from jimmy_assistant.actions import window_control as wc
from jimmy_assistant.nlp.intent import Intent


def test_active_window_info_success(monkeypatch) -> None:
    monkeypatch.setattr(wc, "_get_foreground_hwnd", lambda: 123)
    monkeypatch.setattr(wc, "_window_title", lambda hwnd: "Notepad")

    result = wc.active_window_info(Intent(name="window.info"))

    assert result.ok is True
    assert "Notepad" in result.speak_en


def test_control_window_close(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setattr(wc, "_get_foreground_hwnd", lambda: 321)
    monkeypatch.setattr(wc, "_window_title", lambda hwnd: "Calculator")
    monkeypatch.setattr(wc, "_post_close", lambda hwnd: calls.append(hwnd) or True)

    result = wc.control_window(
        Intent(name="window.control", slots={"operation": "close"})
    )

    assert result.ok is True
    assert calls == [321]
    assert "Closing Calculator" in result.speak_en


def test_control_window_rejects_unknown_operation(monkeypatch) -> None:
    monkeypatch.setattr(wc, "_get_foreground_hwnd", lambda: 111)

    result = wc.control_window(
        Intent(name="window.control", slots={"operation": "explode"})
    )

    assert result.ok is False
    assert "unsupported operation" in result.error
