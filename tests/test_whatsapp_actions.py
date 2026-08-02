from __future__ import annotations

from jimmy_assistant.actions import whatsapp
from jimmy_assistant.nlp.intent import Intent


def test_open_only_mode_opens_prefilled_chat(monkeypatch) -> None:
    opened: list[str] = []

    def fake_open(url: str, new: int):  # noqa: ARG001
        opened.append(url)
        return True

    monkeypatch.setattr(whatsapp.webbrowser, "open", fake_open)

    send = whatsapp.build_sender(
        contacts={"sid original": "+919999999999"},
        auto_send=False,
    )
    result = send(
        Intent(
            name="whatsapp.send",
            slots={"contact": "Sid original", "message": "running late by 10 mins"},
        )
    )

    assert result.ok is True
    assert opened
    assert "web.whatsapp.com/send" in opened[0]
    assert "919999999999" in opened[0]


def test_auto_send_calls_pywhatkit(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeKit:
        @staticmethod
        def sendwhatmsg_instantly(phone_no: str, message: str, **kwargs):  # noqa: ANN001
            del kwargs
            calls.append((phone_no, message))

    monkeypatch.setitem(__import__("sys").modules, "pywhatkit", _FakeKit)

    send = whatsapp.build_sender(
        contacts={"sid original": "+919999999999"},
        auto_send=True,
    )
    result = send(
        Intent(
            name="whatsapp.send",
            slots={"contact": "sid original", "message": "where are you"},
        )
    )

    assert result.ok is True
    assert calls == [('+919999999999', 'where are you')]


def test_unknown_contact_fails() -> None:
    send = whatsapp.build_sender(contacts={"mom": "+919888888888"}, auto_send=False)
    result = send(
        Intent(
            name="whatsapp.send",
            slots={"contact": "sid original", "message": "hello"},
        )
    )
    assert result.ok is False
    assert "unknown contact" in result.error
