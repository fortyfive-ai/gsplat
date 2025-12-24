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
  - Use --skip-colmap to test video processing without running COLMAP
"""

import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np


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


def get_video_frame_count(video_path: Path, fps: float) -> int:
    """Calculate how many frames will be extracted from a video at given fps."""
    ffprobe_path = get_ffprobe_path()

    # Get video duration and fps
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "csv=s=,:p=0",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip().split(',')

    try:
        duration = float(output[0])
    except (ValueError, IndexError):
        # Fallback: try format duration
        cmd = [
            ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=s=,:p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

    # Calculate expected frame count at target fps
    frame_count = int(duration * fps)
    return frame_count


def get_video_brightness(video_path: Path) -> float:
    """Calculate average brightness of a video by sampling frames.

    Returns:
        float: Average brightness value (0-255)
    """
    import tempfile
    import shutil

    ffmpeg_path = get_ffmpeg_path()

    # Create temporary directory for frame samples
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Extract 5 sample frames
        cmd = [
            ffmpeg_path,
            "-i", str(video_path),
            "-vf", "select='not(mod(n\\,100))'",
            "-vsync", "vfr",
            "-frames:v", "5",
            "-q:v", "2",
            str(temp_dir / "sample_%03d.jpg")
        ]

        subprocess.run(cmd, capture_output=True, check=False)

        # Calculate brightness from extracted frames using PIL
        sample_files = list(temp_dir.glob("sample_*.jpg"))

        if sample_files:
            brightness_values = []
            for img_file in sample_files:
                img = Image.open(img_file).convert('L')
                pixels = np.array(img)
                brightness_values.append(pixels.mean())
                img.close()

            avg_brightness = sum(brightness_values) / len(brightness_values) * 2
            return avg_brightness

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # Fallback: return neutral brightness
    return 128.0

def get_video_color_balance(video_path: Path) -> tuple[float, float, float]:
    """Calculate average color balance (R, G, B means) of a video by sampling frames.

    Returns:
        tuple: (avg_red, avg_green, avg_blue) values (0-255)
    """
    import tempfile
    import shutil

    ffmpeg_path = get_ffmpeg_path()

    # Create temporary directory for frame samples
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Extract 5 sample frames
        cmd = [
            ffmpeg_path,
            "-i", str(video_path),
            "-vf", "select='not(mod(n\\,100))'",
            "-vsync", "vfr",
            "-frames:v", "5",
            "-q:v", "2",
            str(temp_dir / "sample_%03d.jpg")
        ]

        subprocess.run(cmd, capture_output=True, check=False)

        # Calculate color balance from extracted frames
        sample_files = list(temp_dir.glob("sample_*.jpg"))

        if sample_files:
            r_values, g_values, b_values = [], [], []
            for img_file in sample_files:
                img = Image.open(img_file).convert('RGB')
                pixels = np.array(img)
                r_values.append(pixels[:, :, 0].mean())
                g_values.append(pixels[:, :, 1].mean())
                b_values.append(pixels[:, :, 2].mean())
                img.close()

            avg_r = sum(r_values) / len(r_values)
            avg_g = sum(g_values) / len(g_values)
            avg_b = sum(b_values) / len(b_values)
            return avg_r, avg_g, avg_b

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # Fallback: return neutral color balance
    return 128.0, 128.0, 128.0


def concat_videos(video_paths: list[Path], output_path: Path):
    """Concatenate multiple videos into a single video using ffmpeg.

    If videos have different resolutions, they will be resized to match
    the resolution of the first video before concatenation.

    Special handling: nav_rear videos will have brightness and white balance
    matched to the first video using FFmpeg color correction filters.
    """
    ffmpeg_path = get_ffmpeg_path()

    # Get resolution of the first video (target resolution)
    target_width, target_height = get_video_resolution(video_paths[0])
    print(f"Target resolution: {target_width}x{target_height} (from first video)")

    # Get brightness and color balance of the first video for matching
    print(f"\nAnalyzing reference video (video 1)...")
    reference_brightness = get_video_brightness(video_paths[0])
    reference_r, reference_g, reference_b = get_video_color_balance(video_paths[0])
    print(f"Reference brightness: {reference_brightness:.2f}")
    print(f"Reference color balance: R={reference_r:.2f}, G={reference_g:.2f}, B={reference_b:.2f}")

    # Check if all videos have the same resolution and identify nav_rear videos
    need_resize = False
    has_nav_rear = False
    for i, video in enumerate(video_paths):
        width, height = get_video_resolution(video)
        is_nav_rear = "nav_rear" in video.stem.lower()
        if is_nav_rear:
            has_nav_rear = True
        status = " (nav_rear - will match brightness)" if is_nav_rear else ""
        print(f"Video {i+1}: {video.name} - {width}x{height}{status}")
        if width != target_width or height != target_height:
            need_resize = True

    # Need processing if resize required OR nav_rear videos present
    need_processing = need_resize or has_nav_rear
    temp_dir = None
    videos_to_concat = video_paths

    if need_processing:
        if need_resize:
            print(f"\nProcessing videos to match target resolution {target_width}x{target_height}...")

        temp_dir = Path(tempfile.mkdtemp())
        videos_to_concat = []

        for i, video in enumerate(video_paths):
            width, height = get_video_resolution(video)
            processed_path = temp_dir / f"processed_{i}_{video.name}"
            video_name = video.stem.lower()
            is_nav_rear = "nav_rear" in video_name

            # Build video filter chain
            vf_filters = []
            brightness_adjust = 0.0  # Initialize for logging

            # Add scaling if needed
            if width != target_width or height != target_height:
                vf_filters.append(f"scale={target_width}:{target_height}")

            # Add brightness and white balance matching for nav_rear videos
            if is_nav_rear and i > 0:  # Don't process first video
                # Get brightness and color balance of this video
                current_brightness = get_video_brightness(video)
                current_r, current_g, current_b = get_video_color_balance(video)

                # Calculate brightness adjustment
                # Use eq filter to adjust brightness
                brightness_diff = reference_brightness - current_brightness
                brightness_adjust = brightness_diff / 255.0
                brightness_adjust = max(-0.5, min(0.5, brightness_adjust))

                # Calculate white balance adjustments using a more conservative approach
                # Calculate the ratio of each channel to match reference
                if current_r > 0 and current_g > 0 and current_b > 0:
                    # Calculate direct gain ratios
                    r_gain = reference_r / current_r
                    g_gain = reference_g / current_g
                    b_gain = reference_b / current_b
                else:
                    r_gain = g_gain = b_gain = 1.0

                print(f"  Video {i+1} color correction:")
                print(f"    Brightness: {current_brightness:.2f} -> {reference_brightness:.2f} (adjust={brightness_adjust:.4f})")
                print(f"    Color balance: R={current_r:.2f}, G={current_g:.2f}, B={current_b:.2f}")
                print(f"    Target balance: R={reference_r:.2f}, G={reference_g:.2f}, B={reference_b:.2f}")
                print(f"    Color gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")

                # Apply brightness and color corrections
                # Use eq filter for brightness adjustment and colorchannelmixer for white balance
                # The colorchannelmixer is better for white balance than gamma adjustments
                color_filter = f"colorchannelmixer=rr={r_gain:.3f}:gg={g_gain:.3f}:bb={b_gain:.3f}"
                brightness_filter = f"eq=brightness={brightness_adjust:.4f}"

                # Apply both filters
                vf_filters.append(brightness_filter)
                vf_filters.append(color_filter)

            # Build the command
            process_cmd = [
                ffmpeg_path,
                "-i", str(video),
            ]

            if vf_filters:
                process_cmd.extend(["-vf", ",".join(vf_filters)])

            process_cmd.extend([
                "-c:v", "libx264",  # Use consistent video codec
                "-preset", "medium",
                "-crf", "18",  # High quality
                "-an",  # No audio (not needed for COLMAP)
                str(processed_path)
            ])

            if i == 0:
                print(f"  Video {i+1}: Processing reference video...")
            elif is_nav_rear:
                if width != target_width or height != target_height:
                    print(f"  Video {i+1} (nav_rear): Resizing from {width}x{height} and applying color correction...")
                else:
                    print(f"  Video {i+1} (nav_rear): Applying color correction...")
            elif width != target_width or height != target_height:
                print(f"  Video {i+1}: Resizing from {width}x{height}...")
            else:
                print(f"  Video {i+1}: Copying...")

            subprocess.run(process_cmd, check=True, capture_output=True)
            videos_to_concat.append(processed_path)

    # Create a temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_file = Path(f.name)
        for video in videos_to_concat:
            f.write(f"file '{video.resolve()}'\n")

    try:
        cmd = [
            ffmpeg_path, "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",  # Now safe to copy since all videos have same codec
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
    """Extract frames from video using ffmpeg.

    Returns:
        tuple: (images_dir, num_frames)
    """
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
    return images_dir, num_frames


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


def setup_masks(output_dir: Path, images_dir: Path, video_paths: list[Path] = None,
                video_frame_counts: list[int] = None, fps: float = 2.0):
    """Setup masks directory with camera-specific masks or empty masks.

    Args:
        output_dir: Output directory where masks/ will be created
        images_dir: Directory containing the extracted images
        video_paths: List of video paths to extract camera names from
        video_frame_counts: List of frame counts per video (if known)
        fps: Frame rate used for extraction
    """
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Path to proxie masks directory
    proxie_masks_dir = Path("/Users/yanbinghan/Fortyfive Dropbox/Yanbing Han/third_party/cobot-fortyfive/cobot_datasets/proxie_masks")

    # Get list of images to create masks for
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    if not image_files:
        print("Warning: No images found to create masks for")
        return masks_dir

    # Get image dimensions from first image
    first_image = Image.open(image_files[0])
    img_width, img_height = first_image.size
    first_image.close()

    # Camera identifiers to look for
    camera_identifiers = ["nav_front", "nav_rear", "nav_left", "nav_right",
                        "blindspot_front", "blindspot_rear"]

    # If we have video paths, calculate frame ranges for each video
    if video_paths and len(video_paths) > 1:
        print(f"\nProcessing masks for {len(video_paths)} videos...")

        # Calculate or use provided frame counts
        if video_frame_counts is None:
            print("Calculating frame counts for each video...")
            video_frame_counts = []
            for video_path in video_paths:
                frame_count = get_video_frame_count(video_path, fps)
                video_frame_counts.append(frame_count)

        # Build frame ranges and camera masks for each video
        frame_start = 1  # Frames are numbered starting from 1
        loaded_masks = {}  # Cache loaded masks

        for idx, (video_path, frame_count) in enumerate(zip(video_paths, video_frame_counts)):
            video_name = video_path.stem.lower()
            frame_end = frame_start + frame_count - 1

            # Find which camera this video is from
            camera_mask_path = None
            camera_id = None
            for cam_id in camera_identifiers:
                if cam_id in video_name:
                    camera_id = cam_id
                    mask_file = proxie_masks_dir / f"{cam_id}_mask.png"
                    if mask_file.exists():
                        camera_mask_path = mask_file
                    break

            # Load or create mask for this camera
            if camera_mask_path and camera_mask_path not in loaded_masks:
                camera_mask = Image.open(camera_mask_path).convert('L')
                if camera_mask.size != (img_width, img_height):
                    camera_mask = camera_mask.resize((img_width, img_height), Image.LANCZOS)
                loaded_masks[camera_mask_path] = camera_mask
                print(f"  Video {idx+1} ({video_path.name}): frames {frame_start}-{frame_end} -> {cam_id}_mask")
            elif camera_mask_path:
                print(f"  Video {idx+1} ({video_path.name}): frames {frame_start}-{frame_end} -> {cam_id}_mask (cached)")
            else:
                if camera_id:
                    print(f"  Video {idx+1} ({video_path.name}): frames {frame_start}-{frame_end} -> empty mask (no mask found for {camera_id})")
                else:
                    print(f"  Video {idx+1} ({video_path.name}): frames {frame_start}-{frame_end} -> empty mask (unknown camera)")

            # Create masks for this frame range
            for frame_num in range(frame_start, min(frame_end + 1, len(image_files) + 1)):
                mask_name = f"frame_{frame_num:04d}.png"
                mask_path = masks_dir / mask_name

                if camera_mask_path and camera_mask_path in loaded_masks:
                    loaded_masks[camera_mask_path].save(mask_path)
                else:
                    # Create empty (all white) mask
                    empty_mask = Image.new('L', (img_width, img_height), 255)
                    empty_mask.save(mask_path)
                    empty_mask.close()

            frame_start = frame_end + 1

        # Close all loaded masks
        for mask in loaded_masks.values():
            mask.close()

        print(f"Created {len(image_files)} masks for {len(video_paths)} videos")

    elif video_paths and len(video_paths) == 1:
        # Single video - use simple logic
        video_name = video_paths[0].stem.lower()
        camera_mask_path = None

        for cam_id in camera_identifiers:
            if cam_id in video_name:
                mask_file = proxie_masks_dir / f"{cam_id}_mask.png"
                if mask_file.exists():
                    camera_mask_path = mask_file
                    print(f"Found camera mask for '{cam_id}': {mask_file}")
                    break

        if camera_mask_path:
            camera_mask = Image.open(camera_mask_path).convert('L')
            if camera_mask.size != (img_width, img_height):
                print(f"Resizing mask from {camera_mask.size} to {img_width}x{img_height}")
                camera_mask = camera_mask.resize((img_width, img_height), Image.LANCZOS)

            print(f"Creating {len(image_files)} masks using camera mask...")
            for img_file in image_files:
                mask_name = img_file.stem + ".png"
                mask_path = masks_dir / mask_name
                camera_mask.save(mask_path)

            camera_mask.close()
            print(f"Created {len(image_files)} camera-specific masks")
        else:
            print(f"No camera mask found, creating {len(image_files)} empty (all-white) masks...")
            empty_mask = Image.new('L', (img_width, img_height), 255)
            for img_file in image_files:
                mask_name = img_file.stem + ".png"
                mask_path = masks_dir / mask_name
                empty_mask.save(mask_path)
            empty_mask.close()
            print(f"Created {len(image_files)} empty masks")
    else:
        # No video paths provided - create empty masks
        print(f"No video information, creating {len(image_files)} empty (all-white) masks...")
        empty_mask = Image.new('L', (img_width, img_height), 255)
        for img_file in image_files:
            mask_name = img_file.stem + ".png"
            mask_path = masks_dir / mask_name
            empty_mask.save(mask_path)
        empty_mask.close()
        print(f"Created {len(image_files)} empty masks")

    return masks_dir


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
            video_to_process = video_files[0]
        else:
            print(f"Input is {len(video_files)} video files")
            # Concatenate videos
            concat_output = output_dir / "concatenated.mp4"
            concat_videos(video_files, concat_output)
            video_to_process = concat_output

        images_dir, _ = extract_frames(video_to_process, output_dir, args.fps)

    # Setup masks
    print("\nSetting up masks...")
    masks_dir = setup_masks(output_dir, images_dir, video_paths_for_masks, fps=args.fps)

    # Run COLMAP unless skipped
    if args.skip_colmap:
        print("\nSkipping COLMAP reconstruction (--skip-colmap flag set)")
        print(f"Images saved to: {images_dir}")
        print(f"Masks saved to: {masks_dir}")
    else:
        sparse_dir = run_colmap(output_dir, images_dir, args.gpu)
        print(f"\nReconstruction complete!")
        print(f"Results saved to: {sparse_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
