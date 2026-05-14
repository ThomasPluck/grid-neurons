"""ElevenLabs text-to-speech wrapper.

Usage as a module:
    import tts
    tts.synthesize(text="...", out_path="audio/01_cold_open.mp3")

Usage from the command line (smoke test):
    python tts.py narration/01_cold_open.txt audio/01_cold_open.mp3
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

import config

load_dotenv(config.ROOT / ".env")

_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
if not _API_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or config.DEFAULT_VOICE_ID
MODEL_ID = config.ELEVEN_MODEL_ID

_client = ElevenLabs(api_key=_API_KEY)

_MAX_RETRIES = 4
_BASE_DELAY = 2.0  # seconds; doubled each retry


def synthesize(text: str, out_path: str) -> str:
    """Render ``text`` to an mp3 at ``out_path``. Returns ``out_path``.

    Retries with exponential backoff on transient API errors (rate limits, 5xx).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            stream = _client.text_to_speech.convert(
                voice_id=VOICE_ID,
                model_id=MODEL_ID,
                text=text,
                voice_settings={"stability": 0.5, "similarity_boost": 0.8},
            )
            data = b"".join(stream)
            if not data:
                raise RuntimeError("ElevenLabs returned an empty audio stream.")
            out.write_bytes(data)
            return str(out)
        except ApiError as e:
            status = getattr(e, "status_code", None)
            # 429 = rate limit, 5xx = server side. Retry those; fail fast otherwise.
            if status is not None and status != 429 and status < 500:
                raise
            last_err = e
        except Exception as e:  # noqa: BLE001 - network hiccups, connection resets
            last_err = e

        if attempt < _MAX_RETRIES - 1:
            delay = _BASE_DELAY * (2 ** attempt)
            print(f"  TTS attempt {attempt + 1} failed ({last_err}); retrying in {delay:.0f}s")
            time.sleep(delay)

    raise RuntimeError(f"TTS failed after {_MAX_RETRIES} attempts: {last_err}")


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    text = Path(argv[0]).read_text(encoding="utf-8").strip()
    path = synthesize(text, argv[1])
    print(f"wrote {path} ({Path(path).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
