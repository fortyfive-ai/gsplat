#!/usr/bin/env python3
"""
Video/Images to hloc (SuperPoint+SuperGlue) reconstruction pipeline.
Usage:
  python video_to_hloc.py input.mp4 output_dir [--fps 2]
  python video_to_hloc.py /path/to/images_dir output_dir

Requires:
- pip install hloc
- git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
- Set PYTHONPATH to include SuperGluePretrainedNetwork parent directory
"""

import argparse
from pathlib import Path

from core.frame_extractor import extract_frames, setup_images_from_directory
from core.feature_matching import run_hloc


def main():
    parser = argparse.ArgumentParser(description="Video/Images to hloc (SuperPoint+SuperGlue) reconstruction")
    parser.add_argument("input", type=Path, help="Input video file (.mp4) or image directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for video (default: 2)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # Check if input is a directory or video file
    if args.input.is_dir():
        print(f"Input is a directory: {args.input}")
        images_dir = setup_images_from_directory(args.input, args.output)
        if images_dir is None:
            return 1
    else:
        print(f"Input is a video file: {args.input}")
        images_dir, _ = extract_frames(args.input, args.output, args.fps, "frame")

    # Run hloc
    sparse_dir = run_hloc(args.output, images_dir)

    if sparse_dir:
        print(f"\nResults saved to: {sparse_dir}")
        return 0
    return 1


if __name__ == "__main__":
    exit(main())
