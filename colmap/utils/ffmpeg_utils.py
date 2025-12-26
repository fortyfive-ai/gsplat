"""FFmpeg utility functions for video analysis and processing."""

import subprocess
from pathlib import Path


def get_ffmpeg_path() -> str:
    """Get ffmpeg executable path.

    Returns:
        str: Path to ffmpeg executable
    """
    ffmpeg_path = "/home/yanbinghan/bin/ffmpeg"
    if not Path(ffmpeg_path).exists():
        ffmpeg_path = "ffmpeg"
    return ffmpeg_path


def get_ffprobe_path() -> str:
    """Get ffprobe executable path.

    Returns:
        str: Path to ffprobe executable
    """
    ffprobe_path = "/home/yanbinghan/bin/ffprobe"
    if not Path(ffprobe_path).exists():
        ffprobe_path = "ffprobe"
    return ffprobe_path


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """Get video resolution using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        tuple: (width, height) in pixels
    """
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
    """Calculate how many frames will be extracted from a video at given fps.

    Args:
        video_path: Path to video file
        fps: Target frame rate for extraction

    Returns:
        int: Number of frames that will be extracted
    """
    ffprobe_path = get_ffprobe_path()

    # Get video duration
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
