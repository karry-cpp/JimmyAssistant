from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# --- Canonical action names ------------------------------------------------
# Every value returned by the parser or router must be one of these. New
# actions are added here first, then wired into the action registry.
ACTION_HIBERNATE = "power.hibernate"
ACTION_SHUTDOWN = "power.shutdown"
ACTION_RESTART = "power.restart"
ACTION_SLEEP = "power.sleep"
ACTION_LOCK = "power.lock"

ACTION_APP_LAUNCH = "apps.launch"
ACTION_APP_CLOSE = "apps.close"

ACTION_VOLUME_UP = "volume.up"
ACTION_VOLUME_DOWN = "volume.down"
ACTION_VOLUME_MUTE = "volume.mute"
ACTION_VOLUME_UNMUTE = "volume.unmute"
ACTION_VOLUME_SET = "volume.set"

ACTION_MEDIA_PLAY_PAUSE = "media.play_pause"
ACTION_MEDIA_NEXT = "media.next"
ACTION_MEDIA_PREV = "media.previous"
ACTION_MEDIA_STOP = "media.stop"

ACTION_OPEN_THING = "open.thing"
ACTION_WEB_SEARCH = "web.search"
ACTION_WEB_CURRENT_TIME = "web.current_time"
ACTION_YOUTUBE_PLAY = "youtube.play"
ACTION_WINDOW_INFO = "window.info"
ACTION_WINDOW_CONTROL = "window.control"
ACTION_WHATSAPP_SEND = "whatsapp.send"

ACTION_ANSWER_DIRECT = "answer.direct"

ACTION_CANCEL = "system.cancel"
ACTION_UNKNOWN = "unknown"


KNOWN_ACTIONS = frozenset(
    {
        ACTION_HIBERNATE,
        ACTION_SHUTDOWN,
        ACTION_RESTART,
        ACTION_SLEEP,
        ACTION_LOCK,
        ACTION_APP_LAUNCH,
        ACTION_APP_CLOSE,
        ACTION_VOLUME_UP,
        ACTION_VOLUME_DOWN,
        ACTION_VOLUME_MUTE,
        ACTION_VOLUME_UNMUTE,
        ACTION_VOLUME_SET,
        ACTION_MEDIA_PLAY_PAUSE,
        ACTION_MEDIA_NEXT,
        ACTION_MEDIA_PREV,
        ACTION_MEDIA_STOP,
        ACTION_OPEN_THING,
        ACTION_WEB_SEARCH,
        ACTION_WEB_CURRENT_TIME,
        ACTION_YOUTUBE_PLAY,
        ACTION_WINDOW_INFO,
        ACTION_WINDOW_CONTROL,
        ACTION_WHATSAPP_SEND,
        ACTION_ANSWER_DIRECT,
        ACTION_CANCEL,
    }
)


@dataclass(frozen=True)
class Intent:
    """A parsed user intent.

    ``name`` is a canonical action id (see constants above). ``slots``
    contains action-specific parameters (query text, app name, volume
    level, etc.). ``source`` records which stage produced the intent
    (``rules`` or ``llm``) — useful for debugging."""

    name: str
    slots: Dict[str, str] = field(default_factory=dict)
    source: str = "rules"
    raw_text: str = ""

    @property
    def is_unknown(self) -> bool:
        return self.name == ACTION_UNKNOWN

    @property
    def is_cancel(self) -> bool:
        return self.name == ACTION_CANCEL
