#!/usr/bin/env python3
"""
Image directories to COLMAP reconstruction pipeline.
Usage:
  python image_to_colmap.py --inputs /path/to/images1 /path/to/images2 --output output_dir [--gpu 0]
  python image_to_colmap.py --inputs /path/to/images --output output_dir [--gpu 0]
  python image_to_colmap.py --inputs /path/to/images1 /path/to/images2 --output output_dir --original-fps 30 --target-fps 4 [--gpu 0]

Features:
  - Supports multiple image directories
  - Images from each directory are prefixed with video identifiers (video1_, video2_, etc.)
  - Camera-specific masks are automatically matched based on directory names
  - FPS downsampling: specify --original-fps and --target-fps to downsample image sequences
  - Use --skip-colmap to test image processing without running COLMAP
"""

import json
import shutil
from pathlib import Path
from typing import List
from params_proto import proto

from .core.mask_generator import setup_masks
from .core.feature_matching import run_colmap


def find_depth_directory(image_dir: Path) -> Path:
    """Find corresponding depth directory for an image directory.

    Args:
        image_dir: Path to image directory (e.g., .../images/nav_front_d455_color_image_compressed)

    Returns:
        Path to depth directory or None if not found
    """
    # Navigate up to find the depth directory
    # Expected structure: .../demcap/XXX/images/nav_front_d455_color_image_compressed
    #                     .../demcap/XXX/depth/nav_front_d455_aligned_depth_to_color_image_raw_compressedDepth

    parent = image_dir.parent  # .../images
    base_dir = parent.parent   # .../demcap/XXX
    depth_base = base_dir / "depth"

    if not depth_base.exists():
        return None

    # Extract camera identifier from image directory name
    # e.g., "nav_front_d455_color_image_compressed" -> "nav_front_d455"
    dir_name = image_dir.name
    camera_id = dir_name.replace('_color_image_compressed', '')

    # Look for depth directory with matching camera identifier
    depth_dir = depth_base / f"{camera_id}_aligned_depth_to_color_image_raw_compressedDepth"

    if depth_dir.exists():
        return depth_dir

    return None


def find_closest_depth_image(color_timestamp: str, depth_images: List[Path], tolerance_ns: int = 50_000_000) -> Path:
    """Find the depth image with the closest timestamp to the color image.

    Args:
        color_timestamp: Timestamp string from color image filename (nanoseconds)
        depth_images: List of depth image paths
        tolerance_ns: Maximum allowed time difference in nanoseconds (default 50ms)

    Returns:
        Path to closest depth image or None if no match within tolerance
    """
    color_ts = int(color_timestamp)

    # Find depth image with minimum time difference
    closest_depth = None
    min_diff = float('inf')

    for depth_img in depth_images:
        depth_ts = int(depth_img.stem)  # filename without extension
        diff = abs(depth_ts - color_ts)

        if diff < min_diff:
            min_diff = diff
            closest_depth = depth_img

    # Only return if within tolerance
    if min_diff <= tolerance_ns:
        return closest_depth

    return None


def setup_images_from_multiple_directories(input_dirs: List[Path], output_dir: Path,
                                          original_fps: float = None, target_fps: float = None) -> tuple[Path, dict]:
    """Setup images from multiple directories with prefixes.

    Args:
        input_dirs: List of input directories containing images
        output_dir: Output directory
        original_fps: Original FPS of the image sequence (for downsampling)
        target_fps: Target FPS for downsampling

    Returns:
        tuple: (images_dir, image_counts_per_dir)
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    depths_dir = output_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing images in output
    for f in images_dir.glob("*.jpg"):
        f.unlink()
    for f in images_dir.glob("*.png"):
        f.unlink()

    # Clean existing depths in output
    for f in depths_dir.glob("*.png"):
        f.unlink()

    # Calculate sample rate for FPS downsampling
    sample_rate = 1
    if original_fps is not None and target_fps is not None:
        sample_rate = int(original_fps / target_fps)
        print(f"\nFPS downsampling enabled: {original_fps} fps -> {target_fps} fps (sample every {sample_rate} frames)")

    image_counts = {}

    for idx, input_dir in enumerate(input_dirs):
        # Find images in input directory
        input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")) + \
                       list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG"))

        if not input_images:
            print(f"Warning: No images found in {input_dir}")
            continue

        # Sort images to ensure consistent ordering
        input_images = sorted(input_images)

        # Find corresponding depth directory
        depth_dir = find_depth_directory(input_dir)
        depth_images = []
        if depth_dir:
            depth_images = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.PNG")))
            print(f"\nFound depth directory: {depth_dir.name}")
            print(f"  Found {len(depth_images)} depth images")
        else:
            print(f"\nWarning: No depth directory found for {input_dir.name}")

        # Create a clean prefix from the directory name
        dir_name = input_dir.name.lower()
        # Remove common suffixes and clean the name (same logic as mask_generator.py)
        clean_name = dir_name.replace('_d455_color_image_compressed', '').replace('_images', '').replace('_', '')
        prefix = f"video{idx+1}_{clean_name}"

        print(f"\nProcessing directory {idx+1}/{len(input_dirs)}: {input_dir.name}")
        print(f"  Found {len(input_images)} images")
        print(f"  Using prefix: {prefix}")

        # Copy or link images with prefix, applying FPS downsampling if enabled
        copied_count = 0
        depth_copied_count = 0
        output_idx = 1
        for img_idx, img in enumerate(input_images):
            # Skip images based on sample rate
            if img_idx % sample_rate != 0:
                continue

            # Create new filename with prefix and sequential numbering
            new_name = f"{prefix}_{output_idx:04d}{img.suffix}"
            link_path = images_dir / new_name

            if link_path.exists():
                link_path.unlink()

            # Use symlink for efficiency (or copy if symlink fails)
            try:
                link_path.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, link_path)

            copied_count += 1

            # Process corresponding depth image if available
            if depth_images:
                # Extract timestamp from color image filename
                color_timestamp = img.stem  # filename without extension

                # Find closest depth image
                depth_img = find_closest_depth_image(color_timestamp, depth_images)

                if depth_img:
                    # Create depth filename matching the color image naming
                    depth_name = f"{prefix}_{output_idx:04d}.png"
                    depth_link_path = depths_dir / depth_name

                    if depth_link_path.exists():
                        depth_link_path.unlink()

                    try:
                        depth_link_path.symlink_to(depth_img.resolve())
                    except OSError:
                        shutil.copy2(depth_img, depth_link_path)

                    depth_copied_count += 1

            output_idx += 1

        image_counts[prefix] = copied_count
        print(f"  Processed {copied_count} images (sampled from {len(input_images)})")
        if depth_images:
            print(f"  Processed {depth_copied_count} depth images")

    total_images = sum(image_counts.values())
    print(f"\nTotal images: {total_images} from {len(input_dirs)} directories")

    return images_dir, image_counts


def setup_images_from_single_directory(input_dir: Path, output_dir: Path,
                                      original_fps: float = None, target_fps: float = None) -> Path:
    """Setup images from a single directory.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory
        original_fps: Original FPS of the image sequence (for downsampling)
        target_fps: Target FPS for downsampling

    Returns:
        Path: images_dir path, or None if no images found
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    depths_dir = output_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing images in output
    for f in images_dir.glob("*.jpg"):
        f.unlink()
    for f in images_dir.glob("*.png"):
        f.unlink()

    # Clean existing depths in output
    for f in depths_dir.glob("*.png"):
        f.unlink()

    # Find images in input directory
    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")) + \
                   list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG"))

    if not input_images:
        print(f"Error: No images found in {input_dir}")
        return None

    # Sort images to ensure consistent ordering
    input_images = sorted(input_images)

    # Find corresponding depth directory
    depth_dir = find_depth_directory(input_dir)
    depth_images = []
    if depth_dir:
        depth_images = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.PNG")))
        print(f"Found depth directory: {depth_dir.name}")
        print(f"  Found {len(depth_images)} depth images")
    else:
        print(f"Warning: No depth directory found for {input_dir.name}")

    # Calculate sample rate for FPS downsampling
    sample_rate = 1
    if original_fps is not None and target_fps is not None:
        sample_rate = int(original_fps / target_fps)
        print(f"FPS downsampling enabled: {original_fps} fps -> {target_fps} fps (sample every {sample_rate} frames)")

    # Create symlinks to original images, applying FPS downsampling if enabled
    print(f"Linking images from {input_dir}...")
    linked_count = 0
    depth_linked_count = 0
    for img_idx, img in enumerate(input_images):
        # Skip images based on sample rate
        if img_idx % sample_rate != 0:
            continue

        link_path = images_dir / img.name
        if link_path.exists():
            link_path.unlink()
        try:
            link_path.symlink_to(img.resolve())
        except OSError:
            shutil.copy2(img, link_path)
        linked_count += 1

        # Process corresponding depth image if available
        if depth_images:
            # Extract timestamp from color image filename
            color_timestamp = img.stem  # filename without extension

            # Find closest depth image
            depth_img = find_closest_depth_image(color_timestamp, depth_images)

            if depth_img:
                # Use the same filename as the color image but with .png extension
                depth_name = f"{img.stem}.png"
                depth_link_path = depths_dir / depth_name

                if depth_link_path.exists():
                    depth_link_path.unlink()

                try:
                    depth_link_path.symlink_to(depth_img.resolve())
                except OSError:
                    shutil.copy2(depth_img, depth_link_path)

                depth_linked_count += 1

    print(f"Linked {linked_count} images (sampled from {len(input_images)})")
    if depth_images:
        print(f"Linked {depth_linked_count} depth images")
    return images_dir


@proto.cli
def main(
    inputs: str,  # Input image directory/directories (comma-separated if multiple)
    output: str,  # Output directory
    gpu: int = 0,  # GPU index
    skip_colmap: bool = False,  # Skip COLMAP reconstruction (useful for testing image processing)
    set_config: str = None,  # JSON config file defining set groupings (required)
    original_fps: float = None,  # Original FPS of the image sequence (for downsampling)
    target_fps: float = None,  # Target FPS for downsampling
    camera_intrinsics: str = None,  # Path to camera_intrinsics.json file for fixed intrinsics
):
    """Image directories to COLMAP reconstruction"""
    # Validate required arguments
    if inputs is None or output is None:
        print("Error: inputs and output are required arguments")
        print("Usage: python image_to_colmap.py --inputs dir1,dir2,dir3 --output output_dir")
        return 1

    # Parse comma-separated input directories
    input_dirs = [d.strip() for d in inputs.split(',')]

    # Convert string inputs to Path objects
    input_paths = [Path(p) for p in input_dirs]
    output_dir = Path(output)

    if len(input_paths) < 1:
        print("Error: At least one input directory required")
        return 1

    # Validate inputs exist and are directories
    for input_path in input_paths:
        if not input_path.exists():
            print(f"Error: Input not found: {input_path}")
            return 1
        if not input_path.is_dir():
            print(f"Error: Input is not a directory: {input_path}")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup images
    use_prefixes = len(input_paths) > 1
    image_counts = None

    if use_prefixes:
        print(f"Processing {len(input_paths)} image directories...")
        images_dir, image_counts = setup_images_from_multiple_directories(
            input_paths, output_dir, original_fps, target_fps
        )
    else:
        print(f"Processing single image directory: {input_paths[0]}")
        images_dir = setup_images_from_single_directory(
            input_paths[0], output_dir, original_fps, target_fps
        )
        if images_dir is None:
            return 1

    # Setup masks
    print("\nSetting up masks...")
    # Convert input directories to "video paths" for mask detection
    # The mask generator will extract camera identifiers from directory names
    pseudo_video_paths = [Path(str(p) + ".mp4") for p in input_paths]
    masks_dir = setup_masks(output_dir, images_dir, video_paths=pseudo_video_paths,
                           video_frame_counts=None, fps=None,
                           use_video_prefixes=use_prefixes)

    # Run COLMAP unless skipped
    if skip_colmap:
        print("\nSkipping COLMAP reconstruction (--skip-colmap flag set)")
        print(f"Images saved to: {images_dir}")
        print(f"Masks saved to: {masks_dir}")
        depths_dir = output_dir / "depths"
        if depths_dir.exists() and any(depths_dir.iterdir()):
            print(f"Depths saved to: {depths_dir}")
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
        group_by = config.get("group_by", "camera")
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
        depths_dir = output_dir / "depths"
        if depths_dir.exists() and any(depths_dir.iterdir()):
            print(f"Depths saved to: {depths_dir}")
        return 0


if __name__ == "__main__":
    main()
