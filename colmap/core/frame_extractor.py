"""Frame extraction utilities for videos."""

import subprocess
import shutil
import tempfile
from pathlib import Path

from ..utils.ffmpeg_utils import get_ffmpeg_path
from .video_processor import get_video_brightness, get_video_color_balance


def extract_frames(video_path: Path, output_dir: Path, fps: float = 2.0, video_prefix: str = "frame") -> tuple[Path, int]:
    """Extract frames from video using ffmpeg.

    Args:
        video_path: Path to the video file
        output_dir: Output directory
        fps: Frame extraction rate
        video_prefix: Prefix for frame names (e.g., "video1", "video2")

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
        str(images_dir / f"{video_prefix}_%04d.jpg")
    ]

    print(f"Extracting frames at {fps} fps with prefix '{video_prefix}'...")
    subprocess.run(cmd, check=True)

    num_frames = len(list(images_dir.glob(f"{video_prefix}_*.jpg")))
    print(f"Extracted {num_frames} frames")
    return images_dir, num_frames


def extract_frames_from_multiple_videos(video_paths: list[Path], output_dir: Path, fps: float = 2.0) -> tuple[Path, list[int]]:
    """Extract frames from multiple videos, preserving video source in frame names.

    Applies color correction to nav_rear videos to match the first video.

    Args:
        video_paths: List of video file paths
        output_dir: Output directory
        fps: Frame extraction rate

    Returns:
        tuple: (images_dir, list of frame counts per video)
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Get brightness and white balance info from first video for matching
    print(f"\nAnalyzing reference video (video 1)...")
    reference_brightness = get_video_brightness(video_paths[0])
    reference_r, reference_g, reference_b = get_video_color_balance(video_paths[0])
    print(f"Reference brightness: {reference_brightness:.2f}")
    print(f"Reference color balance: R={reference_r:.2f}, G={reference_g:.2f}, B={reference_b:.2f}")

    frame_counts = []
    temp_dir = None

    for idx, video_path in enumerate(video_paths):
        video_name = video_path.stem.lower()
        is_nav_rear = "nav_rear" in video_name

        # Create a clean prefix from the video stem (remove special characters)
        # Format: video1_nav_front, video2_nav_rear, etc.
        video_prefix = f"video{idx+1}_{video_name.replace('_d455_color_image_compressed', '').replace('_', '')}"

        # Check if this video needs color correction
        video_to_process = video_path

        if is_nav_rear and idx > 0:
            # Apply color correction before extracting frames
            if temp_dir is None:
                temp_dir = Path(tempfile.mkdtemp())

            corrected_video = temp_dir / f"corrected_{idx}_{video_path.name}"

            # Get current video stats
            current_brightness = get_video_brightness(video_path)
            current_r, current_g, current_b = get_video_color_balance(video_path)

            # Calculate corrections
            brightness_diff = reference_brightness - current_brightness
            brightness_adjust = brightness_diff / 255.0
            brightness_adjust = max(-0.5, min(0.5, brightness_adjust))

            if current_r > 0 and current_g > 0 and current_b > 0:
                r_gain = reference_r / current_r
                g_gain = reference_g / current_g
                b_gain = reference_b / current_b
            else:
                r_gain = g_gain = b_gain = 1.0

            print(f"\n  Video {idx+1} ({video_path.name}) color correction:")
            print(f"    Brightness: {current_brightness:.2f} -> {reference_brightness:.2f} (adjust={brightness_adjust:.4f})")
            print(f"    Color gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")

            # Apply corrections
            ffmpeg_path = get_ffmpeg_path()
            color_filter = f"colorchannelmixer=rr={r_gain:.3f}:gg={g_gain:.3f}:bb={b_gain:.3f}"
            brightness_filter = f"eq=brightness={brightness_adjust:.4f}"

            cmd = [
                ffmpeg_path,
                "-i", str(video_path),
                "-vf", f"{brightness_filter},{color_filter}",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-an",
                str(corrected_video)
            ]

            print(f"    Applying color correction...")
            subprocess.run(cmd, check=True, capture_output=True)
            video_to_process = corrected_video

        print(f"\nExtracting frames from video {idx+1}/{len(video_paths)}: {video_path.name}")
        _, num_frames = extract_frames(video_to_process, output_dir, fps, video_prefix)
        frame_counts.append(num_frames)

    # Clean up temp directory
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)

    total_frames = sum(frame_counts)
    print(f"\nTotal frames extracted: {total_frames} from {len(video_paths)} videos")

    return images_dir, frame_counts


def setup_images_from_directory(input_dir: Path, output_dir: Path) -> Path:
    """Setup images directory from an existing image directory.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory

    Returns:
        Path: images_dir path, or None if no images found
    """
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
