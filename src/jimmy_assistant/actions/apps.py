"""Launch Windows apps by voice-friendly name.

Two-stage resolution:

1. **Alias table** — a hardcoded map from spoken names ("chrome",
   "vs code", "file explorer") to a command Windows Explorer knows how
   to run. This is the primary safety layer: only names on the alias
   table produce a launch.
2. **Fallback** — if the raw slot is a short alnum token, we try
   ``start "" <name>`` which lets Explorer resolve it via the
   ``App Paths`` registry. Values with path separators or unusual
   characters are rejected.
"""

from __future__ import annotations

import logging
import re
import subprocess

from jimmy_assistant.actions.registry import ActionResult
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.utils.text import normalize_text


logger = logging.getLogger(__name__)


CREATE_NO_WINDOW = 0x08000000
_SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9 .\-_]{0,40}$")


# Canonical alias table. Keys are the *normalized* spoken form.
_ALIASES: dict[str, str] = {
    # Browsers
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "brave": "brave",
    # Editors / IDEs
    "notepad": "notepad",
    "notepad plus plus": "notepad++",
    "notepad++": "notepad++",
    "code": "code",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "sublime": "subl",
    # System
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "calculator": "calc",
    "calc": "calc",
    "settings": "ms-settings:",
    "task manager": "taskmgr",
    "cmd": "cmd",
    "command prompt": "cmd",
    "powershell": "powershell",
    "windows terminal": "wt",
    "terminal": "wt",
    "paint": "mspaint",
    # Media / Misc
    "spotify": "spotify",
    "vlc": "vlc",
    "discord": "discord",
    "steam": "steam",
    "obs": "obs",
    "zoom": "zoom",
    "whatsapp": "whatsapp",
    "camera": "microsoft.windows.camera:",
    "photos": "ms-photos:",
}


def _resolve(app_name: str) -> str | None:
    key = normalize_text(app_name)
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    # Try last two words (e.g. "the chrome browser" -> "chrome")
    tokens = key.split()
    for size in (len(tokens), 2, 1):
        if size <= 0 or size > len(tokens):
            continue
        for start in range(0, len(tokens) - size + 1):
            candidate = " ".join(tokens[start : start + size])
            if candidate in _ALIASES:
                return _ALIASES[candidate]
    # Fallback: allow plain alnum tokens; let Windows shell resolve.
    if _SAFE_TOKEN.match(key) and "/" not in key and "\\" not in key:
        return key
    return None


def launch_app(intent: Intent) -> ActionResult:
    raw = intent.slots.get("app", "").strip()
    resolved = _resolve(raw)
    if not resolved:
        return ActionResult.failure(f"unknown app: {raw!r}")

    args = ["cmd.exe", "/c", "start", "", resolved]
    logger.info("Launching app %r via %s", raw, args)
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        return ActionResult.failure(
            (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
        )
    return ActionResult.success(
        speak_en=f"Opening {raw}.",
        speak_hi=f"{raw} khol rahi hoon.",
    )
