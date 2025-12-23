#!/usr/bin/env python3
"""
Video/Images to COLMAP reconstruction pipeline.
Usage:
  python video_to_colmap.py input.mp4 output_dir [--fps 2] [--gpu 0]
  python video_to_colmap.py /path/to/images_dir output_dir [--gpu 0]
"""

import argparse
import subprocess
import shutil
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, fps: float = 2.0):
    """Extract frames from video using ffmpeg."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/home/yanbinghan/bin/ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(images_dir / "frame_%04d.jpg")
    ]

    print(f"Extracting frames at {fps} fps...")
    subprocess.run(cmd, check=True)

    num_frames = len(list(images_dir.glob("*.jpg")))
    print(f"Extracted {num_frames} frames")
    return images_dir


def run_colmap(output_dir: Path, images_dir: Path, gpu_index: int = 0):
    """Run COLMAP reconstruction pipeline using pycolmap."""
    import pycolmap

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing database if present
    if database_path.exists():
        database_path.unlink()

    # Feature extraction (auto device selection)
    print("\nStep 1: Extracting features...")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    # Feature matching
    print("\nStep 2: Matching features...")
    pycolmap.match_exhaustive(
        database_path=str(database_path),
    )

    # Sparse reconstruction with pycolmap
    print("\nStep 3: Running sparse reconstruction...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
    )

    # Report results
    print("\nStep 4: Analyzing reconstruction...")
    for idx, reconstruction in reconstructions.items():
        print(f"\nModel {idx}:")
        print(f"  Registered images: {reconstruction.num_reg_images()}")
        print(f"  3D points: {reconstruction.num_points3D()}")

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
    parser = argparse.ArgumentParser(description="Video/Images to COLMAP reconstruction")
    parser.add_argument("input", type=Path, help="Input video file (.mp4) or image directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for video (default: 2)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
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

    # Run COLMAP
    sparse_dir = run_colmap(args.output, images_dir, args.gpu)

    print(f"\nReconstruction complete!")
    print(f"Results saved to: {sparse_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
