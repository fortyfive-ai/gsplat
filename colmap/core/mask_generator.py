"""Mask generation utilities for camera-specific masking."""

import os
from pathlib import Path
from PIL import Image

from ..utils.ffmpeg_utils import get_video_frame_count


def setup_masks(output_dir: Path, images_dir: Path, video_paths: list[Path] = None,
                video_frame_counts: list[int] = None, fps: float = 2.0, use_video_prefixes: bool = False) -> Path:
    """Setup masks directory with camera-specific masks or empty masks.

    Args:
        output_dir: Output directory where masks/ will be created
        images_dir: Directory containing the extracted images
        video_paths: List of video paths to extract camera names from
        video_frame_counts: List of frame counts per video (if known)
        fps: Frame rate used for extraction
        use_video_prefixes: If True, frames are named with video prefixes (video1_*, video2_*, etc.)

    Returns:
        Path: Path to masks directory
    """
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Path to proxie masks directory
    dropbox_path = os.environ.get("DROPBOX", "")
    if dropbox_path:
        proxie_masks_dir = Path(dropbox_path) / "cobot_datasets" / "proxie_masks"
    else:
        proxie_masks_dir = Path.home() / "Fortyfive Dropbox" / "Yanbing Han" / "third_party" / "cobot-fortyfive" / "cobot_datasets" / "proxie_masks"

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
    camera_identifiers = ["nav_front", "nav_rear", "nav_left", "nav_right", "blindspot_front", "blindspot_rear"]

    # If we have video paths with prefixes, match frames to videos by prefix
    if use_video_prefixes and video_paths:
        print(f"\nProcessing masks for {len(video_paths)} videos with prefixes...")

        loaded_masks = {}  # Cache loaded masks

        for idx, video_path in enumerate(video_paths):
            video_name = video_path.stem.lower()

            # Create the same prefix format used in extraction
            video_prefix = f"video{idx+1}_{video_name.replace('_d455_color_image_compressed', '').replace('_', '')}"

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
                print(f"  Video {idx+1} ({video_path.name}): prefix '{video_prefix}' -> {camera_id}_mask")
            elif camera_mask_path:
                print(f"  Video {idx+1} ({video_path.name}): prefix '{video_prefix}' -> {camera_id}_mask (cached)")
            else:
                if camera_id:
                    print(f"  Video {idx+1} ({video_path.name}): prefix '{video_prefix}' -> empty mask (no mask found for {camera_id})")
                else:
                    print(f"  Video {idx+1} ({video_path.name}): prefix '{video_prefix}' -> empty mask (unknown camera)")

            # Find all frames for this video prefix
            video_frames = sorted(images_dir.glob(f"{video_prefix}_*.jpg"))

            # Create masks for each frame from this video
            for img_file in video_frames:
                mask_name = img_file.stem + ".png"
                mask_path = masks_dir / mask_name

                if camera_mask_path and camera_mask_path in loaded_masks:
                    loaded_masks[camera_mask_path].save(mask_path)
                else:
                    # Create empty (all white) mask
                    empty_mask = Image.new('L', (img_width, img_height), 255)
                    empty_mask.save(mask_path)
                    empty_mask.close()

        # Close all loaded masks
        for mask in loaded_masks.values():
            mask.close()

        print(f"Created {len(image_files)} masks for {len(video_paths)} videos")

    # If we have video paths, calculate frame ranges for each video (old method for backward compatibility)
    elif video_paths and len(video_paths) > 1:
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
                print(f"  Video {idx+1} ({video_path.name}): frames {frame_start}-{frame_end} -> {camera_id}_mask")
            elif camera_mask_path:
                print(f"  Video {idx+1} ({video_path.name}): frames {frame_start}-{frame_end} -> {camera_id}_mask (cached)")
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
