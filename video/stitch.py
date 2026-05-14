"""FFmpeg wrappers for merging audio+video and concatenating scenes.

Usage as a module:
    import stitch
    stitch.merge_av("clips/01.mp4", "audio/01.mp3", "stitched/01.mp4")
    stitch.concat(["stitched/01.mp4", ...], "output/final.mp4")

Usage from the command line (smoke test):
    python stitch.py 01_cold_open          # merge that one scene
    python stitch.py --concat              # concat all stitched scenes
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import config


def _ffmpeg() -> str:
    """Locate the ffmpeg executable.

    Prefers PATH. Falls back to the winget install location, since a freshly
    winget-installed ffmpeg is not on PATH until the shell is restarted.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        pkgs = Path(local) / "Microsoft" / "WinGet" / "Packages"
        hits = list(pkgs.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
        if hits:
            return str(hits[0])
    raise RuntimeError(
        "ffmpeg not found. Install it via your system package manager "
        "(winget install Gyan.FFmpeg / brew install ffmpeg / apt install ffmpeg)."
    )


def merge_av(video: str, audio: str, out: str) -> str:
    """Mux a video clip with its narration track.

    Stream-copies the video and re-encodes audio to AAC. ``-shortest`` trims to
    whichever stream ends first; build.py keeps scene durations matched so the
    drift this papers over is sub-second.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg(), "-y",
            "-i", video,
            "-i", audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out,
        ],
        check=True,
    )
    return out


def concat(inputs: list[str], out: str) -> str:
    """Concatenate per-scene mp4s into the final video.

    Uses the ffmpeg concat demuxer with stream copy. Every input must share
    codec parameters; build.py renders every scene with the same quality flag
    to guarantee that. Falls back to a re-encode if stream copy fails.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    listfile = config.ROOT / "concat_list.txt"
    listfile.write_text(
        "".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in inputs),
        encoding="utf-8",
    )
    base = [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listfile)]
    try:
        subprocess.run(base + ["-c", "copy", out], check=True)
    except subprocess.CalledProcessError:
        print("  stream-copy concat failed; falling back to re-encode")
        subprocess.run(
            base + ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", out],
            check=True,
        )
    finally:
        listfile.unlink(missing_ok=True)
    return out


def _main(argv: list[str]) -> int:
    config.ensure_dirs()
    if argv and argv[0] == "--concat":
        inputs = [str(config.STITCHED_DIR / f"{sid}.mp4") for sid, _ in config.SCENES]
        out = concat(inputs, str(config.OUTPUT_DIR / "final.mp4"))
        print(f"wrote {out}")
        return 0
    if len(argv) == 1:
        sid = argv[0]
        out = merge_av(
            str(config.CLIPS_DIR / f"{sid}.mp4"),
            str(config.AUDIO_DIR / f"{sid}.mp3"),
            str(config.STITCHED_DIR / f"{sid}.mp4"),
        )
        print(f"wrote {out}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
