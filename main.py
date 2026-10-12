import argparse
import os
import sys
from pipeline import CaptionBehindPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Caption Behind the Subject Generator - bolti.ai / Roosh Tech pipeline"
    )
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input talking-head video (.mp4)")
    parser.add_argument("--srt", "-s", type=str, required=True, help="Path to captions SRT file (.srt)")
    parser.add_argument(
        "--template", "-t", type=str, default="dynamic", choices=["dynamic", "luxe"],
        help="Template style: 'dynamic' (Estate Dynamic) or 'luxe' (Luxe Elegant)"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output MP4 file path")
    parser.add_argument(
        "--segmenter", type=str, default="mediapipe",
        help="Segmentation backend ('mediapipe', extensible for GPU models like RVM)"
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames for fast preview testing")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.srt):
        print(f"Error: SRT file not found: {args.srt}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_behind_{args.template}.mp4"

    pipeline = CaptionBehindPipeline(segmenter_name=args.segmenter)
    pipeline.process_video(
        input_video_path=args.video,
        srt_path=args.srt,
        output_video_path=args.output,
        template_name=args.template,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
