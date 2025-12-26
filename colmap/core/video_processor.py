"""Video processing utilities for color correction and concatenation."""

import subprocess
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from ..utils.ffmpeg_utils import get_ffmpeg_path, get_video_resolution


def get_video_brightness(video_path: Path) -> float:
    """Calculate average brightness of a video by sampling frames.

    Args:
        video_path: Path to video file

    Returns:
        float: Average brightness value (0-255)
    """
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

    Args:
        video_path: Path to video file

    Returns:
        tuple: (avg_red, avg_green, avg_blue) values (0-255)
    """
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

    Args:
        video_paths: List of video paths to concatenate
        output_path: Output path for concatenated video
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

            # Add scaling if needed
            if width != target_width or height != target_height:
                vf_filters.append(f"scale={target_width}:{target_height}")

            # Add brightness and white balance matching for nav_rear videos
            if is_nav_rear and i > 0:  # Don't process first video
                # Get brightness and color balance of this video
                current_brightness = get_video_brightness(video)
                current_r, current_g, current_b = get_video_color_balance(video)

                # Calculate brightness adjustment
                brightness_diff = reference_brightness - current_brightness
                brightness_adjust = brightness_diff / 255.0
                brightness_adjust = max(-0.5, min(0.5, brightness_adjust))

                # Calculate white balance adjustments
                if current_r > 0 and current_g > 0 and current_b > 0:
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
