"""Actions for interacting with the currently focused desktop window.

These handlers intentionally expose a small, safe control surface:
- identify the active window title
- minimize / maximize / restore / close the active window

No arbitrary keystroke injection or shell execution is performed.
"""

from __future__ import annotations

import ctypes
import logging

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent


logger = logging.getLogger(__name__)

_SW_MINIMIZE = 6
_SW_MAXIMIZE = 3
_SW_RESTORE = 9
_WM_CLOSE = 0x0010


def _get_foreground_hwnd() -> int:
    return int(ctypes.windll.user32.GetForegroundWindow())


def _window_title(hwnd: int) -> str:
    if hwnd <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(512)
    n = int(ctypes.windll.user32.GetWindowTextW(hwnd, buf, len(buf)))
    if n <= 0:
        return ""
    return buf.value[:n].strip()


def _show_window(hwnd: int, cmd: int) -> bool:
    return bool(ctypes.windll.user32.ShowWindow(hwnd, cmd))


def _post_close(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0))


def active_window_info(_intent: Intent) -> ActionResult:
    hwnd = _get_foreground_hwnd()
    if hwnd <= 0:
        return ActionResult.failure("no active window")
    title = _window_title(hwnd) or "(untitled window)"
    return ActionResult.success(speak_en=f"The active window is: {title}.")


def control_window(intent: Intent) -> ActionResult:
    op = intent.slots.get("operation", "").strip().lower()
    if not op:
        return ActionResult.failure("missing operation")

    hwnd = _get_foreground_hwnd()
    if hwnd <= 0:
        return ActionResult.failure("no active window")

    title = _window_title(hwnd) or "current window"

    if op == "minimize":
        _show_window(hwnd, _SW_MINIMIZE)
        return ActionResult.success(speak_en=f"Minimized {title}.")
    if op == "maximize":
        _show_window(hwnd, _SW_MAXIMIZE)
        return ActionResult.success(speak_en=f"Maximized {title}.")
    if op == "restore":
        _show_window(hwnd, _SW_RESTORE)
        return ActionResult.success(speak_en=f"Restored {title}.")
    if op == "close":
        _post_close(hwnd)
        return ActionResult.success(speak_en=f"Closing {title}.")

    logger.info("Unsupported window operation: %s", op)
    return ActionResult.failure(
        "unsupported operation; use minimize, maximize, restore, or close"
    )
