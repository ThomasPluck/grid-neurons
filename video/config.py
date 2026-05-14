"""Central configuration: scene list, paths, model IDs.

Note on layout: the implementation brief calls the rendered-clip directory
``video/``. Because the whole project already lives in a folder called
``video/``, that subdirectory is renamed ``clips/`` here to avoid a confusing
``video/video/`` nesting. Everything else matches the brief.
"""

from pathlib import Path

ROOT = Path(__file__).parent.resolve()

NARRATION_DIR = ROOT / "narration"
SCENES_DIR = ROOT / "scenes"
AUDIO_DIR = ROOT / "audio"
CLIPS_DIR = ROOT / "clips"          # per-scene .mp4 from manim (brief calls this video/)
STITCHED_DIR = ROOT / "stitched"    # per-scene audio+video merged
OUTPUT_DIR = ROOT / "output"        # final concatenated video
MEDIA_DIR = ROOT / "media"          # manim's render dir

# (scene_id, Manim Scene subclass name). Order is the final video order.
SCENES = [
    ("01_cold_open", "ColdOpen"),
    ("02_act1_cell_and_grid", "Act1CellAndGrid"),
    ("03_act2_learning", "Act2Learning"),
    ("04_aside_snap1", "AsideSnap1"),
    ("05_act3_noise", "Act3Noise"),
    ("06_act4_outro", "Act4Outro"),
]

# ElevenLabs. Model verified against elevenlabs SDK 2.47.0 (text_to_speech.convert).
ELEVEN_MODEL_ID = "eleven_v3"
# Fallback public voice if ELEVENLABS_VOICE_ID is unset. "George" - warm,
# measured narrator. The supplied API key has tts but not voices_read, so the
# voice list cannot be queried; this is a known stable public voice id.
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# Render quality flag. Keep identical across every scene so the final concat
# can stream-copy without re-encoding.
QUALITY_FLAG = "-qh"


def ensure_dirs() -> None:
    for d in (AUDIO_DIR, CLIPS_DIR, STITCHED_DIR, OUTPUT_DIR, MEDIA_DIR):
        d.mkdir(parents=True, exist_ok=True)
