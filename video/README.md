# A Cellular Automaton That Learns to Read — video build

Pipeline that renders the ~10 minute explainer video for the grid-neurons
paper: ElevenLabs narration + Manim visuals, stitched with FFmpeg.

## Setup

```bash
pip install -r requirements.txt
# FFmpeg must come from a system package manager, not pip:
#   winget install Gyan.FFmpeg   |   brew install ffmpeg   |   apt install ffmpeg
cp .env.example .env
# then put your ELEVENLABS_API_KEY (tts permission required) in .env
```

## Build

```bash
python build.py                 # full video -> output/final.mp4
python build.py 01_cold_open    # one scene only (smoke test)
python build.py --skip-tts      # reuse existing audio/, re-render visuals only
```

The pipeline, per scene: synthesise narration audio, probe its duration, render
the Manim scene with `TARGET_DURATION` set to that duration, mux audio+video,
then concatenate all scenes.

## Layout

| path           | committed | contents                                  |
|----------------|-----------|-------------------------------------------|
| `narration/`   | yes       | one `.txt` per scene, source-of-truth copy |
| `scenes/`      | yes       | one Manim `Scene` per file                 |
| `config.py`    | yes       | scene list, paths, model/voice IDs         |
| `tts.py`       | yes       | ElevenLabs wrapper (`text_to_speech.convert`) |
| `stitch.py`    | yes       | FFmpeg merge + concat                      |
| `build.py`     | yes       | orchestrator                               |
| `audio/`       | no        | per-scene `.mp3`                           |
| `clips/`       | no        | per-scene `.mp4` from Manim (brief: `video/`) |
| `stitched/`    | no        | per-scene audio+video merged               |
| `output/`      | no        | `final.mp4`                                |
| `media/`       | no        | Manim's render cache                       |

> The brief names the rendered-clip directory `video/`. Since this whole
> project already sits in a folder called `video/`, that subdirectory is
> `clips/` here to avoid a `video/video/` nesting. Nothing else differs.

## Notes

- **Voice / model.** `eleven_v3` via `text_to_speech.convert` (verified against
  elevenlabs SDK 2.47.0). The supplied API key has `tts` but not `voices_read`,
  so the voice list can't be queried; `config.DEFAULT_VOICE_ID` is a known
  stable public voice ("George"). Override with `ELEVENLABS_VOICE_ID` in `.env`.
- **Scene timing.** Each scene reads `TARGET_DURATION` from the environment and
  divides it across its beats, so visuals fit the narration without re-rendering.
- **Concat.** Every scene renders at the same quality flag (`-qh`) so the final
  concat can stream-copy; `stitch.concat` falls back to a re-encode if not.
