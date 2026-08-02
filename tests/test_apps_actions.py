from __future__ import annotations

from jimmy_assistant.actions import apps
from jimmy_assistant.nlp.intent import Intent


class _Done:
    def __init__(self, code: int = 0, out: str = "", err: str = "") -> None:
        self.returncode = code
        self.stdout = out
        self.stderr = err


def test_close_app_edge_success(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001
        calls.append(list(args))
        return _Done(code=0)

    monkeypatch.setattr(apps.subprocess, "run", fake_run)

    result = apps.close_app(Intent(name="apps.close", slots={"app": "edge"}))

    assert result.ok is True
    assert calls
    assert calls[0][0].lower() == "taskkill"
    assert "msedge.exe" in " ".join(calls[0]).lower()


def test_close_app_unknown_fails() -> None:
    result = apps.close_app(Intent(name="apps.close", slots={"app": "crazy thing !!!"}))
    assert result.ok is False
    assert "unknown app" in result.error
