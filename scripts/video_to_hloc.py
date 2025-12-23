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
import subprocess
import sys
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, fps: float = 2.0):
    """Extract frames from video using ffmpeg."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing frames
    for f in images_dir.glob("*.jpg"):
        f.unlink()

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(images_dir / "frame_%04d.jpg")
    ]

    print(f"Extracting frames at {fps} fps...")
    subprocess.run(cmd, check=True)

    num_frames = len(list(images_dir.glob("*.jpg")))
    print(f"Extracted {num_frames} frames")
    return images_dir


def run_hloc(output_dir: Path, images_dir: Path):
    """Run hloc reconstruction pipeline with SuperPoint+SuperGlue."""
    from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

    sparse_dir = output_dir / "sparse_hloc"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs.txt"

    # Clean up old files
    for f in [features_path, matches_path, pairs_path]:
        if f.exists():
            f.unlink()
            print(f"Removed existing {f.name}")

    # Step 1: Generate exhaustive image pairs
    print("\nStep 1: Generating exhaustive image pairs...")
    image_names = [f.name for f in sorted(images_dir.glob("*.jpg"))]
    if not image_names:
        image_names = [f.name for f in sorted(images_dir.glob("*.png"))]

    if not image_names:
        print("Error: No images found in", images_dir)
        return None

    print(f"Found {len(image_names)} images")
    pairs_from_exhaustive.main(pairs_path, image_list=image_names)
    print(f"Pairs file created: {pairs_path}")

    # Step 2: Extract SuperPoint features
    print("\nStep 2: Extracting SuperPoint features...")
    extract_features.main(
        extract_features.confs["superpoint_max"],
        images_dir,
        feature_path=features_path
    )
    print("Feature extraction complete")

    # Step 3: Match with SuperGlue
    print("\nStep 3: Matching with SuperGlue...")
    match_features.main(
        match_features.confs["superglue"],
        pairs_path,
        features_path,
        matches=matches_path
    )
    print("Feature matching complete")

    # Step 4: Run COLMAP reconstruction
    print("\nStep 4: Running COLMAP reconstruction...")
    model = reconstruction.main(
        sparse_dir,
        images_dir,
        pairs_path,
        features_path,
        matches_path
    )

    print(f"\nReconstruction complete!")
    print(f"Registered images: {model.num_reg_images()}")
    print(f"3D points: {model.num_points3D()}")

    return sparse_dir


def setup_images_from_directory(input_dir: Path, output_dir: Path):
    """Setup images directory from an existing image directory."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing images in output
    for f in images_dir.glob("*.jpg"):
        f.unlink()
    for f in images_dir.glob("*.png"):
        f.unlink()

    # Find images in input directory
    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")) + \
                   list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG"))

    if not input_images:
        print(f"Error: No images found in {input_dir}")
        return None

    # Create symlinks to original images
    print(f"Linking {len(input_images)} images from {input_dir}...")
    for img in sorted(input_images):
        link_path = images_dir / img.name
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(img.resolve())

    print(f"Linked {len(input_images)} images")
    return images_dir


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
        images_dir = extract_frames(args.input, args.output, args.fps)

    # Run hloc
    sparse_dir = run_hloc(args.output, images_dir)

    if sparse_dir:
        print(f"\nResults saved to: {sparse_dir}")
        return 0
    return 1


if __name__ == "__main__":
    exit(main())
