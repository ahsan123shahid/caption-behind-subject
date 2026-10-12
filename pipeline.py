import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from typing import List, Optional

from segmenters import get_segmenter, BaseSegmenter
from templates import get_template, BaseTemplate
from subtitle_parser import parse_srt, CaptionItem

class CaptionBehindPipeline:
    """
    Core video processing pipeline for 'Caption Behind the Subject' effect.
    1. Segments person from background per-frame.
    2. Renders animated typography layer synced to captions.
    3. Composites layers: Background -> Text -> Person on top (occluding text).
    4. Merges original audio with high-quality H.264 MP4 export via FFmpeg.
    """
    def __init__(self, segmenter: Optional[BaseSegmenter] = None, segmenter_name: str = "mediapipe"):
        self.segmenter = segmenter or get_segmenter(segmenter_name)

    def blend_rgba_over_bgr(self, bgr_base: np.ndarray, rgba_layer: np.ndarray) -> np.ndarray:
        """
        Alpha blends an RGBA (H, W, 4) layer onto a BGR (H, W, 3) frame.
        """
        if rgba_layer is None or rgba_layer.shape[-1] != 4:
            return bgr_base

        # Convert RGBA to BGR and Alpha
        layer_rgb = rgba_layer[:, :, :3]
        layer_bgr = cv2.cvtColor(layer_rgb, cv2.COLOR_RGBA2BGR) if rgba_layer.shape[2] == 4 else layer_rgb
        alpha = (rgba_layer[:, :, 3] / 255.0)[:, :, np.newaxis]

        blended = (layer_bgr * alpha + bgr_base * (1.0 - alpha)).astype(np.uint8)
        return blended

    def composite_frame(
        self,
        original_frame: np.ndarray,
        person_mask: np.ndarray,
        text_behind_rgba: np.ndarray,
        text_overlay_rgba: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Composites: Background -> Text Behind -> Person on Top -> Overlay Text.
        person_mask: (H, W) float32 in range [0.0, 1.0] where 1.0 = person
        """
        # Step 1: Blend text over the original background
        bg_with_text = self.blend_rgba_over_bgr(original_frame, text_behind_rgba)

        # Step 2: Layer person foreground on top so person occludes the text
        # person_mask expanded to (H, W, 1)
        m = person_mask[:, :, np.newaxis]
        occluded = (original_frame * m + bg_with_text * (1.0 - m)).astype(np.uint8)

        # Step 3: Layer any foreground overlay text (if present)
        if text_overlay_rgba is not None:
            occluded = self.blend_rgba_over_bgr(occluded, text_overlay_rgba)

        return occluded

    def process_video(
        self,
        input_video_path: str,
        srt_path: str,
        output_video_path: str,
        template_name: str = "dynamic",
        max_frames: Optional[int] = None
    ):
        """
        Executes end-to-end processing:
        - Reads input video and captions.
        - Processes and composites every frame.
        - Retains original audio and saves to output_video_path.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        captions: List[CaptionItem] = parse_srt(srt_path)
        template: BaseTemplate = get_template(template_name)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[Processing Video]")
        print(f"  Input: {input_video_path} ({width}x{height} @ {fps:.2f}fps, {total_frames} frames)")
        print(f"  Template: {template_name.capitalize()}")
        print(f"  Captions loaded: {len(captions)} entries")

        temp_video = output_video_path.replace(".mp4", "_temp_novideoaudio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Rendering Frames", unit="frame")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

                current_time = frame_idx / fps

                # Find active captions for this exact timestamp
                active_caps = [c for c in captions if c.start <= current_time <= c.end]

                # 1. Segment person
                person_mask = self.segmenter.segment(frame)

                # 2. Render text layers from template
                text_behind_rgba = template.render_behind_layer((height, width), current_time, active_caps)
                text_overlay_rgba = template.render_overlay_layer((height, width), current_time, active_caps)

                # 3. Composite layers
                final_frame = self.composite_frame(frame, person_mask, text_behind_rgba, text_overlay_rgba)

                # 4. Write frame
                out.write(final_frame)

                frame_idx += 1
                pbar.update(1)

        finally:
            cap.release()
            out.release()
            pbar.close()

        print("\n[Compositing Complete] Merging original audio with FFmpeg...")
        # Step 4: Use FFmpeg to merge audio and encode with standard H.264 for universal playback
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", input_video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_video_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(temp_video):
                os.remove(temp_video)
            print(f"[Success] Exported final video: {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Warning] FFmpeg audio merge issue, fallback to temp video: {e}")
            if os.path.exists(temp_video):
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
                os.rename(temp_video, output_video_path)
