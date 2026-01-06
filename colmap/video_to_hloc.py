#!/usr/bin/env python3
"""
Video/Images to hloc (SuperPoint+SuperGlue) reconstruction pipeline.
Usage:
  python video_to_hloc.py --input input.mp4 --output output_dir [--fps 2]
  python video_to_hloc.py --input /path/to/images_dir --output output_dir

Requires:
- pip install hloc
- git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
- Set PYTHONPATH to include SuperGluePretrainedNetwork parent directory
"""

from pathlib import Path
from params_proto import proto

from .core.frame_extractor import extract_frames, setup_images_from_directory
from .core.feature_matching import run_hloc


@proto.cli
def main(
    input: str = None,  # Input video file (.mp4) or image directory
    output: str = None,  # Output directory
    fps: float = 2.0,  # Frame extraction rate for video
):
    """Video/Images to hloc (SuperPoint+SuperGlue) reconstruction"""
    if input is None or output is None:
        print("Error: input and output are required arguments")
        print("Usage: python video_to_hloc.py --input input.mp4 --output output_dir")
        return 1

    input_path = Path(input)
    output_path = Path(output)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    # Check if input is a directory or video file
    if input_path.is_dir():
        print(f"Input is a directory: {input_path}")
        images_dir = setup_images_from_directory(input_path, output_path)
        if images_dir is None:
            return 1
    else:
        print(f"Input is a video file: {input_path}")
        images_dir, _ = extract_frames(input_path, output_path, fps, "frame")

    # Run hloc
    sparse_dir = run_hloc(output_path, images_dir)

    if sparse_dir:
        print(f"\nResults saved to: {sparse_dir}")
        return 0
    return 1


if __name__ == "__main__":
    main()
