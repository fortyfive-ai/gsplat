"""Image grouping and set creation for COLMAP matching strategies."""

from pathlib import Path


def group_images_by_pattern(image_files: list[Path], group_by: str = "video") -> dict[str, list[str]]:
    """Group images based on naming patterns.

    Args:
        image_files: List of image file paths
        group_by: Grouping strategy
            - "video": Group by video number (video1_*, video2_*, etc.)
            - "camera": Group by camera name (nav_front, nav_rear, etc.)
            - "video_camera": Group by video+camera combination
            - Custom prefix length: "prefix:N" where N is number of underscore-separated parts

    Returns:
        Dictionary mapping group keys to lists of image names
    """
    groups = {}

    for img_file in image_files:
        stem = img_file.stem
        parts = stem.split('_')

        if group_by == "video":
            # Group by video number only (video1, video2, etc.)
            if parts[0].startswith('video'):
                group_key = parts[0]
            else:
                group_key = "unknown"

        elif group_by == "camera":
            # Group by camera name (assumes format: video{N}_{camera}_{index})
            if len(parts) >= 2 and parts[0].startswith('video'):
                # Extract camera name (everything except video number and index)
                group_key = '_'.join(parts[1:-1])
            else:
                group_key = "unknown"

        elif group_by == "video_camera":
            # Group by video+camera combination
            if len(parts) >= 2 and parts[0].startswith('video'):
                group_key = '_'.join(parts[:-1])  # Everything except index
            else:
                group_key = "unknown"

        elif group_by.startswith("prefix:"):
            # Custom: use first N parts as group key
            try:
                n = int(group_by.split(':')[1])
                group_key = '_'.join(parts[:n])
            except (ValueError, IndexError):
                group_key = "unknown"
        else:
            group_key = "unknown"

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(img_file.name)

    return groups


def create_matching_sets(groups: dict[str, list[str]], set_definitions: dict = None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Organize groups into sets for matching.

    Args:
        groups: Dictionary of image groups from group_images_by_pattern
        set_definitions: Optional manual set definitions
            New format (recommended):
            {
                "set1": {
                    "groups": ["video1_navfront", "video1_navrear"],
                    "matching_mode": "sequential",  # "sequential", "exhaustive", or "none"
                    "time_ordered": true,
                    "match_with": ["set2"]  # Optional: specify which sets to inter-match with
                },
                "set2": {
                    "groups": ["video2_navfront", "video2_navrear"],
                    "matching_mode": "exhaustive",
                    "time_ordered": false,
                    "match_with": []  # Optional: empty list means no inter-set matching
                }
            }
            Note: If "match_with" is not specified, the set will match with all other sets (default behavior)

            Legacy format (still supported):
            {
                "set1": {
                    "groups": ["video1_navfront", "video1_navrear"],
                    "sequential": true,
                    "time_seq": true
                }
            }

            Simple format (defaults to sequential matching, not time-ordered):
            {
                "set1": ["video1_navfront", "video1_navrear"],
                "set2": ["video2_navfront", "video2_navrear"]
            }

            If None, each group becomes its own set with sequential matching.

    Returns:
        Tuple of (sets dictionary, set_config dictionary)
        - sets: Dictionary mapping set names to lists of image names
        - set_config: Dictionary mapping set names to config dicts with keys:
            - "matching_mode": "sequential", "exhaustive", or "none"
            - "time_ordered": bool
            - "match_with": list of set names to inter-match with (None = match with all)
    """
    if set_definitions:
        # Manual set definitions
        sets = {}
        set_config_out = {}

        for set_name, set_def in set_definitions.items():
            # Parse config - support new format, legacy format, and simple list
            if isinstance(set_def, dict) and "groups" in set_def:
                group_keys = set_def["groups"]

                # New format: matching_mode + time_ordered
                if "matching_mode" in set_def:
                    matching_mode = set_def["matching_mode"]
                    time_ordered = set_def.get("time_ordered", False)
                    match_with = set_def.get("match_with", None)  # None = match with all
                # Legacy format: sequential + time_seq + intra_exhaustive
                else:
                    sequential = set_def.get("sequential", True)
                    intra_exhaustive = set_def.get("intra_exhaustive", False)

                    if sequential:
                        matching_mode = "sequential"
                    elif intra_exhaustive:
                        matching_mode = "exhaustive"
                    else:
                        matching_mode = "none"

                    time_ordered = set_def.get("time_seq", False)
                    match_with = None  # Legacy format always matches with all
            else:
                # Simple list format - default to sequential matching, not time-ordered
                group_keys = set_def
                matching_mode = "sequential"
                time_ordered = False
                match_with = None

            set_config_out[set_name] = {
                "matching_mode": matching_mode,
                "time_ordered": time_ordered,
                "match_with": match_with
            }

            # Collect images from groups
            if time_ordered:
                # Time-based sequencing: interleave frames by frame number
                group_images = {}
                for group_key in group_keys:
                    if group_key in groups:
                        group_images[group_key] = sorted(groups[group_key])
                    else:
                        print(f"Warning: Group '{group_key}' not found in images")

                # Interleave by frame index: cam1_0001, cam2_0001, cam3_0001, cam1_0002, ...
                interleaved = []
                max_frames = max(len(imgs) for imgs in group_images.values()) if group_images else 0

                for frame_idx in range(max_frames):
                    for group_key in group_keys:
                        if group_key in group_images and frame_idx < len(group_images[group_key]):
                            interleaved.append(group_images[group_key][frame_idx])

                sets[set_name] = interleaved
            else:
                # Normal: concatenate all groups
                sets[set_name] = []
                for group_key in group_keys:
                    if group_key in groups:
                        sets[set_name].extend(groups[group_key])
                    else:
                        print(f"Warning: Group '{group_key}' not found in images")

        return sets, set_config_out
    else:
        # Auto: each group is its own set with sequential matching, not time-ordered
        sets = {f"set_{k}": v for k, v in groups.items()}
        set_config_out = {
            set_name: {"matching_mode": "sequential", "time_ordered": False, "match_with": None}
            for set_name in sets.keys()
        }
        return sets, set_config_out
