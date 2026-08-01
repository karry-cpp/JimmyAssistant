# Jimmy Assistant — Always-Listening Voice Assistant for Windows

Jimmy Assistant is a local, extensible voice assistant that:

- **Wakes on "Hey Jimmy"** (Vosk small Indian-English model, offline).
  The wake phrase is a common name Vosk transcribes reliably even on
  accented speech; the app itself is still called Jimmy.
- **Understands Indian-accented English and Hindi** via `faster-whisper`
  (`large-v3-turbo` by default — GPU-accelerated when a CUDA runtime is
  available).
- **Resolves intent** with a regex fast-path + a local **Ollama** LLM
  (`qwen2.5:3b-instruct`, multilingual) for open-ended phrasing.
- **Runs whitelisted commands only** — the LLM chooses *which* action to
  run, never the shell command itself.
- **Speaks back** via Edge TTS (Hindi voices) or SAPI5 (`pyttsx3`) offline.
- **Asks for verbal confirmation** before destructive actions
  (hibernate / shutdown / restart).
- **Lives in the system tray** and can be packaged as a single Windows `.exe`.

## Supported commands (v1)

Say **"Hey Jimmy"** followed by any of these (or free-form phrasing — the
local LLM handles anything the rules don't recognise).

| Category | Examples |
|---|---|
| Power | *"lock the pc"*, *"hibernate the pc"*, *"shut down"*, *"restart"*, *"sleep"*, *"pc band kar do"* |
| Apps | *"open chrome"*, *"launch notepad"*, *"vs code start karo"* |
| Volume | *"volume up"*, *"mute"*, *"set volume to 50"*, *"awaaz kam kar do"* |
| Media | *"play"*, *"pause"*, *"next song"*, *"agla gaana"* |
| YouTube | *"play aaoge jab tum on youtube"*, *"aaoge jab tum youtube pe chala do"* |
| Web | *"google weather in mumbai"*, *"search for python tutorials"* |
| Open | *"open https://github.com"*, *"open the folder downloads"* |

Everything not matched by the rules is escalated to Ollama — say a
command in plain English or Hinglish and it will still work.

## Architecture

```
mic → Vosk (wake word) → VAD → faster-whisper → rules / Ollama → whitelisted action → TTS
```

- `src/jimmy_assistant/audio/`      — mic stream, wake-word detector, VAD recorder
- `src/jimmy_assistant/stt/`        — faster-whisper wrapper
- `src/jimmy_assistant/nlp/`        — regex rules, Ollama client, router
- `src/jimmy_assistant/actions/`    — one file per action group; whitelisted registry
- `src/jimmy_assistant/tts/`        — Edge TTS + pyttsx3 fallback
- `src/jimmy_assistant/ui/tray.py`  — system-tray icon
- `src/jimmy_assistant/main.py`     — orchestrator (wake → capture → transcribe → dispatch)

## Prerequisites

1. **Python 3.10 – 3.14** works. On Python 3.14, `webrtcvad` doesn't yet
   ship a wheel — the app gracefully falls back to an RMS-energy voice
   activity detector. If you want the higher-quality webrtcvad-based
   VAD, use Python 3.10–3.12 and uncomment `webrtcvad` in `requirements.txt`.
2. **Vosk small Indian-English model** — download and extract into `models/`:
   - https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip → `models/vosk-model-small-en-in-0.4/`
3. **Ollama** (for LLM intent fallback):
   - Install from https://ollama.com/download
   - `ollama pull qwen2.5:3b-instruct`
4. **Windows 10 / 11** (power/volume/media handlers are Windows-specific).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # optional — tweak defaults
```

## Run

```powershell
python run.py            # tray icon
python run.py --console  # headless (log to stderr + %APPDATA%\Jimmy\logs)
```

Try:

- *"Hey Jimmy, lock the pc"*
- *"Hey Jimmy, play aaoge jab tum on youtube"*
- *"Hey Jimmy, chrome kholo"*
- *"Hey Jimmy, volume band karo"*
- *"Hey Jimmy, shut down the pc"* → **"Should I shut down the pc?"** → *"yes"*

The first Whisper transcription downloads the `large-v3-turbo` model
(~1.6 GB) into `models/whisper/`. Subsequent runs use the cached copy.

## Configuration

All settings can be overridden via env vars or a `.env` file. See
`.env.example` for the full list. Notable knobs:

- `JIMMY_WHISPER_MODEL` — swap between `tiny`, `base`, `small`,
  `medium`, `large-v3`, `large-v3-turbo`. Default `large-v3-turbo`.
- `JIMMY_OLLAMA_MODEL` — swap the intent LLM (e.g. `llama3.2:3b`).
- `JIMMY_OLLAMA_ENABLED=false` — rules-only mode.
- `JIMMY_CONFIRM_DESTRUCTIVE=false` — disable the "are you sure?" prompt.
- `JIMMY_WAKE_PHRASES=hey jimmy,hi jimmy,hey jim` — customise the wake phrases.

## Tests

```powershell
pip install pytest
pytest tests/
```

## How to add a new command

1. Add an action constant in `src/jimmy_assistant/nlp/intent.py`.
2. Add regex patterns in `src/jimmy_assistant/nlp/rules.py`.
3. Add the action name + a short example in
   `src/jimmy_assistant/nlp/prompts.py` (so the LLM knows it exists).
4. Write a handler function returning `ActionResult` in
   `src/jimmy_assistant/actions/<group>.py`.
5. Register it in `_build_registry()` in `src/jimmy_assistant/main.py`.

## Packaging as `.exe`

```powershell
pip install pyinstaller
.\packaging\build.ps1
```

Output: `dist/jimmy/jimmy.exe`. See `packaging/jimmy.spec` for details.
Copy the Vosk model directory into `dist/jimmy/models/` before launch.
