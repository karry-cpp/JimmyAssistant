"""Volume control on Windows.

Uses ``pycaw`` (Core Audio API wrapper) for absolute control and mute
state; falls back to sending Windows media keys via ``ctypes`` if pycaw
is unavailable. All handlers degrade gracefully — nothing crashes on a
missing dep, we just report failure.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)


# Windows virtual-key codes for media keys.
_VK_VOLUME_MUTE = 0xAD
_VK_VOLUME_DOWN = 0xAE
_VK_VOLUME_UP = 0xAF
_KEYEVENTF_KEYUP = 0x0002


def _press_key(vk_code: int, times: int = 1) -> None:
    user32 = ctypes.windll.user32
    for _ in range(max(1, times)):
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


def _get_endpoint() -> Optional[object]:
    """Return a pycaw ``IAudioEndpointVolume`` handle, or ``None`` if
    pycaw is not installed / COM init fails.

    Newer pycaw releases (>=2024) expose ``EndpointVolume`` directly on
    the :class:`AudioDevice` object returned by ``GetSpeakers``; older
    releases require an explicit ``Activate`` call. We try the new
    attribute first and fall back to the legacy path so the same
    handler works across versions.
    """
    try:
        from pycaw.pycaw import AudioUtilities  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.debug("pycaw unavailable: %s", exc)
        return None

    try:
        speakers = AudioUtilities.GetSpeakers()
    except Exception:  # noqa: BLE001
        logger.exception("Could not query default audio device")
        return None

    # New API: direct attribute access.
    endpoint = getattr(speakers, "EndpointVolume", None)
    if endpoint is not None:
        return endpoint

    # Legacy API: activate the IAudioEndpointVolume interface via COM.
    try:
        from pycaw.pycaw import IAudioEndpointVolume  # type: ignore
        from comtypes import CLSCTX_ALL  # type: ignore
        from ctypes import POINTER, cast

        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception:  # noqa: BLE001
        logger.exception("Legacy pycaw activation failed")
        return None


def volume_up(_intent: Intent) -> ActionResult:
    ep = _get_endpoint()
    if ep is not None:
        try:
            current = ep.GetMasterVolumeLevelScalar()
            new = min(1.0, current + 0.05)
            ep.SetMasterVolumeLevelScalar(new, None)
            return ActionResult.success(speak_en="Volume up.", speak_hi="Awaaz badhai.")
        except Exception:  # noqa: BLE001
            logger.exception("pycaw volume up failed; using media key")
    _press_key(_VK_VOLUME_UP, times=3)
    return ActionResult.success(speak_en="Volume up.", speak_hi="Awaaz badhai.")


def volume_down(_intent: Intent) -> ActionResult:
    ep = _get_endpoint()
    if ep is not None:
        try:
            current = ep.GetMasterVolumeLevelScalar()
            new = max(0.0, current - 0.05)
            ep.SetMasterVolumeLevelScalar(new, None)
            return ActionResult.success(speak_en="Volume down.", speak_hi="Awaaz kam ki.")
        except Exception:  # noqa: BLE001
            logger.exception("pycaw volume down failed; using media key")
    _press_key(_VK_VOLUME_DOWN, times=3)
    return ActionResult.success(speak_en="Volume down.", speak_hi="Awaaz kam ki.")


def volume_mute(_intent: Intent) -> ActionResult:
    ep = _get_endpoint()
    if ep is not None:
        try:
            ep.SetMute(1, None)
            return ActionResult.success(speak_en="Muted.", speak_hi="Mute kar diya.")
        except Exception:  # noqa: BLE001
            logger.exception("pycaw mute failed; using media key")
    _press_key(_VK_VOLUME_MUTE)
    return ActionResult.success(speak_en="Muted.", speak_hi="Mute kar diya.")


def volume_unmute(_intent: Intent) -> ActionResult:
    ep = _get_endpoint()
    if ep is not None:
        try:
            ep.SetMute(0, None)
            return ActionResult.success(speak_en="Unmuted.", speak_hi="Awaaz chalu.")
        except Exception:  # noqa: BLE001
            logger.exception("pycaw unmute failed; using media key")
    # Media-key mute toggles both ways; safe to press once.
    _press_key(_VK_VOLUME_MUTE)
    return ActionResult.success(speak_en="Unmuted.", speak_hi="Awaaz chalu.")


def volume_set(intent: Intent) -> ActionResult:
    raw = intent.slots.get("level", "").strip()
    try:
        level = int(raw)
    except ValueError:
        return ActionResult.failure(f"invalid volume level: {raw!r}")
    level = max(0, min(100, level))

    ep = _get_endpoint()
    if ep is not None:
        try:
            ep.SetMasterVolumeLevelScalar(level / 100.0, None)
            return ActionResult.success(
                speak_en=f"Volume set to {level} percent.",
                speak_hi=f"Volume {level} par set kar di.",
            )
        except Exception:  # noqa: BLE001
            logger.exception("pycaw set volume failed")

    # No fine-grained fallback; approximate with steps of 2%.
    steps = level // 2
    _press_key(_VK_VOLUME_DOWN, times=50)  # zero out
    _press_key(_VK_VOLUME_UP, times=steps)
    return ActionResult.success(
        speak_en=f"Volume approximately {level} percent.",
        speak_hi=f"Volume takreeban {level} par kar di.",
    )
