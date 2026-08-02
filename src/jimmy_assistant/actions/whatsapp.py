"""WhatsApp messaging action.

Supports opt-in automation with two modes:
- open-only (default): open WhatsApp Web chat with prefilled text
- auto-send: send immediately via pywhatkit
"""

from __future__ import annotations

import logging
import re
import webbrowser
from urllib.parse import quote_plus

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.utils.text import normalize_text


logger = logging.getLogger(__name__)


def _normalize_phone(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    keep_plus = raw.startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return ""
    return f"+{digits}" if keep_plus else digits


def _resolve_contact(name: str, contacts: dict[str, str]) -> tuple[str, str] | None:
    key = normalize_text(name)
    if not key:
        return None

    if key in contacts:
        phone = _normalize_phone(contacts[key])
        if phone:
            return key, phone

    tokens = key.split()
    for size in (len(tokens), 2, 1):
        if size <= 0 or size > len(tokens):
            continue
        for start in range(0, len(tokens) - size + 1):
            candidate = " ".join(tokens[start : start + size])
            if candidate in contacts:
                phone = _normalize_phone(contacts[candidate])
                if phone:
                    return candidate, phone
    return None


def build_sender(contacts: dict[str, str], auto_send: bool = False):
    """Factory that binds contact book + automation mode into a handler."""

    def send_whatsapp(intent: Intent) -> ActionResult:
        contact = intent.slots.get("contact", "").strip()
        message = intent.slots.get("message", "").strip()
        if not contact:
            return ActionResult.failure("missing contact")
        if not message:
            return ActionResult.failure("missing message")

        found = _resolve_contact(contact, contacts)
        if not found:
            return ActionResult.failure(
                f"unknown contact: {contact!r}. Add it in JIMMY_WHATSAPP_CONTACTS"
            )
        resolved_name, phone = found

        if auto_send:
            try:
                import pywhatkit  # local import: optional dep

                pywhatkit.sendwhatmsg_instantly(
                    phone_no=phone,
                    message=message,
                    wait_time=12,
                    tab_close=True,
                    close_time=2,
                )
                return ActionResult.success(
                    speak_en=f"Sent your WhatsApp message to {resolved_name}.",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Auto-send WhatsApp failed")
                return ActionResult.failure(str(exc))

        # Open-only mode: user reviews and presses send manually.
        wa_phone = re.sub(r"\D", "", phone)
        url = f"https://web.whatsapp.com/send?phone={wa_phone}&text={quote_plus(message)}"
        webbrowser.open(url, new=2)
        return ActionResult.success(
            speak_en=(
                f"Opened WhatsApp chat for {resolved_name} with your message ready. "
                "Please press send."
            ),
        )

    return send_whatsapp
