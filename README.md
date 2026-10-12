# Caption Behind the Subject

Text sits **behind** the person: `background → text → person (mask) → optional front text`.

## Run
```bash
pip install -r requirements.txt        # needs ffmpeg on PATH
python fonts/download_fonts.py
python -m behind_caption talk.mp4 --srt talk.srt --template all --out outputs/   # both demos
python -m behind_caption talk.mp4 --srt talk.srt --template luxe --out luxe.mp4
```
Options: `--segmenter mediapipe|rembg|dummy`, `--smooth 0.5` (temporal mask smoothing).
In the SRT, wrap a word in `*stars*` to force it as the keyword; otherwise the longest word is used.

## Structure
- `segmentation.py` – `Segmenter` interface + registry. Add RVM/GPU model = one new class with `@register("rvm")`.
- `templates/*.json` – font, colours, size ratios, pill, stroke, `behind` flag, animation (`pop`, `fade_slide`). New template = new JSON.
- `renderer.py` – auto-fit text to video size, animations, two layers (behind / front).
- `pipeline.py` – frame loop, compositing, FFmpeg encode with original audio copied. Frames without a caption skip segmentation.

## Templates

### Dynamic (Estate Dynamic)
- Bold uppercase keyword in orange pill badge, behind subject
- White support text in front
- Word-by-word pop animation with overshoot easing

### Luxe (Elegant)
- Teal gradient keyword (Poppins Bold) behind subject
- White italic support text (Poppins Light Italic) in front
- Smooth fade + upward slide animation

## Notes
Best with one person, good contrast. Keyword is placed at `y_ratio` (0.30 = upper-centre) so the subject occludes its middle.
