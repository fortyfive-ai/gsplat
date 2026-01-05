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

import json
from pathlib import Path
from typing import List
from params_proto import proto

from .core.frame_extractor import extract_frames, extract_frames_from_multiple_videos, setup_images_from_directory
from .core.mask_generator import setup_masks
from .core.feature_matching import run_colmap


@proto.cli
def main(
    inputs: List[str] = None,  # Input video file(s) (.mp4) or image directory
    output: str = None,  # Output directory
    fps: float = 2.0,  # Frame extraction rate for video
    gpu: int = 0,  # GPU index
    no_color_matching: bool = False,  # Disable automatic color matching between videos
    skip_colmap: bool = False,  # Skip COLMAP reconstruction (useful for testing video processing)
    set_config: str = None,  # JSON config file defining set groupings (required)
    camera_intrinsics: str = None,  # Path to camera_intrinsics.json file for fixed intrinsics
):
    """Video/Images to COLMAP reconstruction"""
    # Convert string inputs to Path objects
    if inputs is None or output is None:
        print("Error: inputs and output are required arguments")
        print("Usage: python video_to_colmap.py --inputs video1.mp4 [video2.mp4 ...] --output output_dir")
        return 1

    input_paths = [Path(p) for p in inputs]
    output_dir = Path(output)

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
            images_dir, _ = extract_frames(video_files[0], output_dir, fps, "frame")
        else:
            print(f"Input is {len(video_files)} video files")
            # Extract frames from each video separately with unique prefixes
            images_dir, frame_counts = extract_frames_from_multiple_videos(video_files, output_dir, fps)
            use_video_prefixes = True

    # Setup masks
    print("\nSetting up masks...")
    masks_dir = setup_masks(output_dir, images_dir, video_paths_for_masks,
                           video_frame_counts=frame_counts, fps=fps,
                           use_video_prefixes=use_video_prefixes)

    # Run COLMAP unless skipped
    if skip_colmap:
        print("\nSkipping COLMAP reconstruction (--skip-colmap flag set)")
        print(f"Images saved to: {images_dir}")
        print(f"Masks saved to: {masks_dir}")
        return 0
    else:
        # Load config file
        if set_config is None:
            print("Error: --set-config is required when not skipping COLMAP")
            return 1

        set_config_path = Path(set_config)
        if not set_config_path.exists():
            print(f"Error: Config file not found: {set_config_path}")
            return 1

        print(f"\nLoading config from: {set_config_path}")
        with open(set_config_path, 'r') as f:
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

        # Load camera intrinsics if provided
        intrinsics_dict = None
        if camera_intrinsics is not None:
            intrinsics_path = Path(camera_intrinsics)
            if not intrinsics_path.exists():
                print(f"Error: Camera intrinsics file not found: {intrinsics_path}")
                return 1
            print(f"  Loading camera intrinsics from: {intrinsics_path}")
            with open(intrinsics_path, 'r') as f:
                intrinsics_dict = json.load(f)
            print(f"  Loaded intrinsics for {len(intrinsics_dict)} camera(s)")

        sparse_dir = run_colmap(
            output_dir,
            images_dir,
            gpu,
            group_by=group_by,
            set_definitions=set_definitions,
            intra_set_overlap=intra_set_overlap,
            inter_set_sample_rate=inter_set_sample_rate,
            camera_intrinsics=intrinsics_dict
        )
        print(f"\nReconstruction complete!")
        print(f"Results saved to: {sparse_dir}")
        return 0


if __name__ == "__main__":
    main()
