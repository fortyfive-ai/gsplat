#!/usr/bin/env python3
"""
Video/Images to COLMAP reconstruction pipeline.
Usage:
  python video_to_colmap.py input.mp4 output_dir [--fps 2] [--gpu 0]
  python video_to_colmap.py video1.mp4 video2.mp4 video3.mp4 output_dir [--fps 2] [--gpu 0]
  python video_to_colmap.py /path/to/images_dir output_dir [--gpu 0]
"""

import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path


def get_ffmpeg_path():
    """Get ffmpeg executable path."""
    ffmpeg_path = "/home/yanbinghan/bin/ffmpeg"
    if not Path(ffmpeg_path).exists():
        ffmpeg_path = "ffmpeg"
    return ffmpeg_path


def get_ffprobe_path():
    """Get ffprobe executable path."""
    ffprobe_path = "/home/yanbinghan/bin/ffprobe"
    if not Path(ffprobe_path).exists():
        ffprobe_path = "ffprobe"
    return ffprobe_path


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """Get video resolution (width, height) using ffprobe."""
    ffprobe_path = get_ffprobe_path()

    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = map(int, result.stdout.strip().split('x'))
    return width, height


def concat_videos(video_paths: list[Path], output_path: Path):
    """Concatenate multiple videos into a single video using ffmpeg.

    If videos have different resolutions, they will be resized to match
    the resolution of the first video before concatenation.
    """
    ffmpeg_path = get_ffmpeg_path()

    # Get resolution of the first video (target resolution)
    target_width, target_height = get_video_resolution(video_paths[0])
    print(f"Target resolution: {target_width}x{target_height} (from first video)")

    # Check if all videos have the same resolution
    need_resize = False
    for i, video in enumerate(video_paths):
        width, height = get_video_resolution(video)
        print(f"Video {i+1}: {video.name} - {width}x{height}")
        if width != target_width or height != target_height:
            need_resize = True

    # If resizing is needed, create temporary resized videos
    temp_dir = None
    videos_to_concat = video_paths

    if need_resize:
        print(f"\nResizing videos to match target resolution {target_width}x{target_height}...")
        temp_dir = Path(tempfile.mkdtemp())
        videos_to_concat = []

        for i, video in enumerate(video_paths):
            width, height = get_video_resolution(video)
            if width == target_width and height == target_height:
                # No resize needed, use original
                videos_to_concat.append(video)
                print(f"  Video {i+1}: No resize needed")
            else:
                # Resize to target resolution
                resized_path = temp_dir / f"resized_{i}_{video.name}"
                resize_cmd = [
                    ffmpeg_path,
                    "-i", str(video),
                    "-vf", f"scale={target_width}:{target_height}",
                    "-c:a", "copy",  # Copy audio without re-encoding
                    str(resized_path)
                ]
                print(f"  Video {i+1}: Resizing from {width}x{height}...")
                subprocess.run(resize_cmd, check=True, capture_output=True)
                videos_to_concat.append(resized_path)

    # Create a temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_file = Path(f.name)
        for video in videos_to_concat:
            f.write(f"file '{video.resolve()}'\n")

    try:
        cmd = [
            ffmpeg_path, "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path)
        ]

        print(f"\nConcatenating {len(video_paths)} videos...")
        subprocess.run(cmd, check=True)
        print(f"Videos concatenated successfully")
    finally:
        list_file.unlink()
        # Clean up temporary resized videos
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


def extract_frames(video_path: Path, output_dir: Path, fps: float = 2.0):
    """Extract frames from video using ffmpeg."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = get_ffmpeg_path()

    cmd = [
        ffmpeg_path, "-i", str(video_path),
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
    parser.add_argument("inputs", nargs='+', type=Path, help="Input video file(s) (.mp4) or image directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for video (default: 2)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    args = parser.parse_args()

    # The last argument is the output directory, the rest are inputs
    # We need to handle this since we use nargs='+'
    if len(args.inputs) < 1:
        print("Error: At least one input and an output directory required")
        return 1

    # Separate inputs and output
    all_paths = args.inputs + [args.output]
    output_dir = all_paths[-1]
    input_paths = all_paths[:-1]

    # Validate inputs exist
    for input_path in input_paths:
        if not input_path.exists():
            print(f"Error: Input not found: {input_path}")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if first input is a directory (assume all inputs are same type)
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

        if len(video_files) == 1:
            print(f"Input is a video file: {video_files[0]}")
            video_to_process = video_files[0]
        else:
            print(f"Input is {len(video_files)} video files")
            # Concatenate videos
            concat_output = output_dir / "concatenated.mp4"
            concat_videos(video_files, concat_output)
            video_to_process = concat_output

        images_dir = extract_frames(video_to_process, output_dir, args.fps)

    # Run COLMAP
    sparse_dir = run_colmap(output_dir, images_dir, args.gpu)

    print(f"\nReconstruction complete!")
    print(f"Results saved to: {sparse_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
