import cv2
import numpy as np
import math

def generate_sample_talking_head(output_path="sample_video.mp4", duration_sec=5, fps=30, width=720, height=1280):
    """
    Generates a vertical talking-head sample video:
    - Office/indoor room background
    - Moving human silhouette with natural head sway / gestures in foreground
    """
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[Generating Sample Video] {output_path} ({width}x{height}, {duration_sec}s)...")

    for i in range(total_frames):
        t = i / fps
        
        # 1. Background: Warm modern gradient wall & floor
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        for y in range(height):
            ratio = y / height
            color = (int(210 - ratio * 40), int(220 - ratio * 30), int(225 - ratio * 20))
            frame[y, :] = color

        # Add some indoor background geometry (window / frames)
        cv2.rectangle(frame, (60, 150), (280, 550), (180, 190, 195), -1)
        cv2.rectangle(frame, (75, 165), (265, 535), (220, 230, 235), -1)

        # 2. Subject (Person Talking-Head):
        # Subtle organic movement: breathing and talking sway
        sway_x = int(math.sin(t * 2.0) * 8)
        sway_y = int(math.cos(t * 3.0) * 4)

        center_x = width // 2 + sway_x
        head_y = int(height * 0.38) + sway_y
        head_radius = int(width * 0.16)

        # Shoulders and Torso
        torso_pts = np.array([
            [center_x - int(width * 0.38), height],
            [center_x - int(width * 0.28), int(height * 0.60) + sway_y],
            [center_x - int(width * 0.16), int(height * 0.48) + sway_y],
            [center_x + int(width * 0.16), int(height * 0.48) + sway_y],
            [center_x + int(width * 0.28), int(height * 0.60) + sway_y],
            [center_x + int(width * 0.38), height]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [torso_pts], (70, 60, 50))  # Dark blazer/clothing

        # Neck
        neck_pts = np.array([
            [center_x - int(width * 0.08), int(height * 0.48) + sway_y],
            [center_x + int(width * 0.08), int(height * 0.48) + sway_y],
            [center_x + int(width * 0.07), int(height * 0.42) + sway_y],
            [center_x - int(width * 0.07), int(height * 0.42) + sway_y]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [neck_pts], (165, 190, 220)) # Skin tone

        # Head / Face
        cv2.circle(frame, (center_x, head_y), head_radius, (165, 190, 220), -1)

        # Hair
        hair_pts = np.array([
            [center_x - head_radius - 8, head_y],
            [center_x - head_radius + 10, head_y - head_radius - 15],
            [center_x + head_radius - 10, head_y - head_radius - 15],
            [center_x + head_radius + 8, head_y],
            [center_x + head_radius - 20, head_y - 20],
            [center_x, head_y - head_radius + 5],
            [center_x - head_radius + 20, head_y - 20]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [hair_pts], (30, 35, 45))

        out.write(frame)

    out.release()
    print(f"[Done] Created sample talking head video: {output_path}")

def generate_sample_srt(output_path="sample_captions.srt"):
    """
    Generates SRT file matching the reference styles:
    - 00:00:00 - 00:00:02: REAL ESTATE CAPTIONS
    - 00:00:02 - 00:00:05: GREETED BY THIS LOOK
    """
    srt_content = """1
00:00:00,500 --> 00:00:02,600
Real estate captions

2
00:00:02,800 --> 00:00:04,900
Greeted by this look
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"[Done] Created sample SRT: {output_path}")

if __name__ == "__main__":
    generate_sample_talking_head()
    generate_sample_srt()
