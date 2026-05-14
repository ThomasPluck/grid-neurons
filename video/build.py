"""Orchestrator: narration -> audio -> manim -> per-scene merge -> final concat.

    python build.py                 # build every scene, then concat
    python build.py 01_cold_open    # build one scene only (smoke test)
    python build.py --skip-tts      # reuse existing audio/*.mp3, re-render video
    python build.py --no-concat     # build scenes but skip the final concat

Pipeline order matters: audio is synthesised first so each manim scene can be
rendered to fit its narration's exact duration (passed via TARGET_DURATION).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import mutagen

import config
import stitch


def _audio_duration(path: Path) -> float:
    f = mutagen.File(str(path))
    if f is None or not getattr(f, "info", None):
        raise RuntimeError(f"could not read audio duration from {path}")
    return float(f.info.length)


def _find_manim_output(scene_id: str) -> Path:
    """Glob for the file manim actually produced.

    Output naming varies by manim version; with --media_dir set and -o NAME.mp4
    it lands under <media>/videos/<file_stem>/<res>/NAME.mp4, but glob anyway.
    """
    hits = sorted(
        config.MEDIA_DIR.glob(f"videos/**/{scene_id}.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise RuntimeError(f"manim output for {scene_id} not found under {config.MEDIA_DIR}")
    return hits[0]


def build_scene(scene_id: str, scene_class: str, *, skip_tts: bool) -> Path:
    print(f"\n=== {scene_id} ({scene_class}) ===")
    narration = config.NARRATION_DIR / f"{scene_id}.txt"
    audio_path = config.AUDIO_DIR / f"{scene_id}.mp3"
    clip_path = config.CLIPS_DIR / f"{scene_id}.mp4"
    stitched_path = config.STITCHED_DIR / f"{scene_id}.mp4"

    # 1. Audio from narration text.
    if skip_tts and audio_path.exists():
        print(f"  [skip-tts] reusing {audio_path}")
    else:
        import tts  # imported lazily so --skip-tts works without an API key
        print(f"  synthesising audio -> {audio_path}")
        tts.synthesize(narration.read_text(encoding="utf-8").strip(), str(audio_path))

    # 2. Probe duration.
    duration = _audio_duration(audio_path)
    print(f"  narration duration: {duration:.2f}s")

    # 3. Manim render, parameterised by audio duration.
    scene_file = config.SCENES_DIR / f"{scene_id}.py"
    env = {**os.environ, "TARGET_DURATION": f"{duration:.3f}"}
    print(f"  rendering {scene_file.name} ...")
    subprocess.run(
        [
            "manim", config.QUALITY_FLAG,
            "--media_dir", str(config.MEDIA_DIR),
            "-o", f"{scene_id}.mp4",
            str(scene_file), scene_class,
        ],
        env=env,
        check=True,
    )
    produced = _find_manim_output(scene_id)
    shutil.copy(produced, clip_path)
    print(f"  video -> {clip_path}")

    # 4. Per-scene merge.
    stitch.merge_av(str(clip_path), str(audio_path), str(stitched_path))
    print(f"  stitched -> {stitched_path}")
    return stitched_path


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    wanted = [a for a in argv if not a.startswith("--")]
    skip_tts = "--skip-tts" in flags
    do_concat = "--no-concat" not in flags

    config.ensure_dirs()

    scenes = config.SCENES
    if wanted:
        by_id = dict(config.SCENES)
        missing = [s for s in wanted if s not in by_id]
        if missing:
            print(f"unknown scene id(s): {', '.join(missing)}")
            print(f"valid: {', '.join(s for s, _ in config.SCENES)}")
            return 2
        scenes = [(s, by_id[s]) for s in wanted]

    stitched = [build_scene(sid, cls, skip_tts=skip_tts) for sid, cls in scenes]

    # 5. Final concat (only meaningful for a full build).
    if do_concat and len(stitched) == len(config.SCENES):
        out = config.OUTPUT_DIR / "final.mp4"
        stitch.concat([str(p) for p in stitched], str(out))
        print(f"\n=== done: {out} ===")
    else:
        print(f"\n=== done: {len(stitched)} scene(s) stitched (concat skipped) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
