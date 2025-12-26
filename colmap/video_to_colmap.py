#!/usr/bin/env python3
"""
Video/Images to COLMAP reconstruction pipeline.
Usage:
  python video_to_colmap.py input.mp4 output_dir [--fps 2] [--gpu 0]
  python video_to_colmap.py video1.mp4 video2.mp4 video3.mp4 output_dir [--fps 2] [--gpu 0]
  python video_to_colmap.py /path/to/images_dir output_dir [--gpu 0]

Features:
  - Automatic brightness and white balance matching for videos with 'nav_rear' in the filename
  - Videos are matched to the brightness and color balance of the first input video
  - When multiple videos are provided, frames are named with video-specific prefixes
    (e.g., video1_navfront_0001.jpg, video2_navrear_0001.jpg) to preserve source information
  - Camera-specific masks are automatically matched based on video names
  - Use --skip-colmap to test video processing without running COLMAP
"""

import argparse
import json
from pathlib import Path

from .core.frame_extractor import extract_frames, extract_frames_from_multiple_videos, setup_images_from_directory
from .core.mask_generator import setup_masks
from .core.feature_matching import run_colmap


def main():
    parser = argparse.ArgumentParser(description="Video/Images to COLMAP reconstruction")
    parser.add_argument("inputs", nargs='+', type=Path, help="Input video file(s) (.mp4) or image directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for video (default: 2)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--no-color-matching", action="store_true",
                        help="Disable automatic color matching between videos (default: enabled)")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP reconstruction (useful for testing video processing)")
    parser.add_argument("--set-config", type=Path, required=True,
                        help="JSON config file defining set groupings (required)")
    args = parser.parse_args()

    # Use argparse parsed values directly
    input_paths = args.inputs
    output_dir = args.output

    if len(input_paths) < 1:
        print("Error: At least one input required")
        return 1

    # Validate inputs exist
    for input_path in input_paths:
        if not input_path.exists():
            print(f"Error: Input not found: {input_path}")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if first input is a directory (assume all inputs are same type)
    video_paths_for_masks = None
    use_video_prefixes = False
    frame_counts = None

    if input_paths[0].is_dir():
        if len(input_paths) > 1:
            print("Error: Multiple directories not supported. Please provide video files or a single image directory.")
            return 1
        print(f"Input is a directory: {input_paths[0]}")
        images_dir = setup_images_from_directory(input_paths[0], output_dir)
        if images_dir is None:
            return 1
    else:
        # Handle video file(s)
        video_files = input_paths
        video_paths_for_masks = video_files  # Keep track for mask detection

        if len(video_files) == 1:
            print(f"Input is a video file: {video_files[0]}")
            images_dir, _ = extract_frames(video_files[0], output_dir, args.fps, "frame")
        else:
            print(f"Input is {len(video_files)} video files")
            # Extract frames from each video separately with unique prefixes
            images_dir, frame_counts = extract_frames_from_multiple_videos(video_files, output_dir, args.fps)
            use_video_prefixes = True

    # Setup masks
    print("\nSetting up masks...")
    masks_dir = setup_masks(output_dir, images_dir, video_paths_for_masks,
                           video_frame_counts=frame_counts, fps=args.fps,
                           use_video_prefixes=use_video_prefixes)

    # Run COLMAP unless skipped
    if args.skip_colmap:
        print("\nSkipping COLMAP reconstruction (--skip-colmap flag set)")
        print(f"Images saved to: {images_dir}")
        print(f"Masks saved to: {masks_dir}")
    else:
        # Load config file
        if not args.set_config.exists():
            print(f"Error: Config file not found: {args.set_config}")
            return 1

        print(f"\nLoading config from: {args.set_config}")
        with open(args.set_config, 'r') as f:
            config = json.load(f)

        # Extract parameters from config
        group_by = config.get("group_by", "video_camera")
        set_definitions = config.get("sets", None)

        # Support both new nested format and legacy flat format
        matching_config = config.get("matching", {})
        intra_set_overlap = matching_config.get("intra_set_overlap", config.get("intra_set_overlap", 30))
        inter_set_sample_rate = matching_config.get("inter_set_sample_rate", config.get("inter_set_sample_rate", 5))

        print(f"  group_by: {group_by}")
        print(f"  intra_set_overlap: {intra_set_overlap}")
        print(f"  inter_set_sample_rate: {inter_set_sample_rate}")
        if set_definitions:
            print(f"  sets defined: {len(set_definitions)}")

        sparse_dir = run_colmap(
            output_dir,
            images_dir,
            args.gpu,
            group_by=group_by,
            set_definitions=set_definitions,
            intra_set_overlap=intra_set_overlap,
            inter_set_sample_rate=inter_set_sample_rate
        )
        print(f"\nReconstruction complete!")
        print(f"Results saved to: {sparse_dir}")


if __name__ == "__main__":
    main()
