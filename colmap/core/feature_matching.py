"""Feature extraction and matching using COLMAP and hloc."""

import os
import tempfile
from pathlib import Path
import re

from .image_grouping import group_images_by_pattern, create_matching_sets


def _normalize_camera_name(name: str) -> str:
    """Normalize camera name for matching by removing common suffixes and prefixes.

    Examples:
        nav_front_d455_color_camera_info -> nav_front
        blindspot_rear_d455_aligned_depth_to_color_camera_info -> blindspot_rear
        navfront -> nav_front
        navrear -> nav_rear
    """
    # Remove common suffixes
    for suffix in ['_d455_color_camera_info', '_d455_aligned_depth_to_color_camera_info',
                   '_color_camera_info', '_camera_info', '_d455']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    # Normalize patterns like "navfront" to "nav_front"
    # Handle common camera position patterns
    replacements = {
        'navfront': 'nav_front',
        'navrear': 'nav_rear',
        'navleft': 'nav_left',
        'navright': 'nav_right',
        'blindspotfront': 'blindspot_front',
        'blindspotrear': 'blindspot_rear',
    }

    name_lower = name.lower()
    for pattern, replacement in replacements.items():
        if pattern in name_lower:
            name = name_lower.replace(pattern, replacement)
            break

    return name


def _extract_camera_name(image_filename: str, camera_intrinsics: dict = None) -> str:
    """Extract camera name from image filename and match with intrinsics keys.

    The function normalizes both the image filename and intrinsics keys to handle
    variations like "navfront" vs "nav_front", and matches based on normalized names.

    For example:
        - Image "video1_navfront_0001.jpg" normalizes to "nav_front"
        - Intrinsic key "nav_front_d455_color_camera_info" normalizes to "nav_front"
        - They match!

    Args:
        image_filename: The image filename to extract camera name from
        camera_intrinsics: Dictionary of camera intrinsics (keys are camera names)

    Returns:
        Camera name (the original key from camera_intrinsics), or "default" if no match found
    """
    # Remove extension and extract the camera part from filename
    name = image_filename.rsplit('.', 1)[0]

    # Extract pattern like "video1_navfront_0001" -> "navfront"
    match = re.match(r'(?:video\d+_)?([a-z_]+)_\d+', name)
    if match:
        extracted_name = match.group(1)
    else:
        extracted_name = name

    # Normalize the extracted name
    normalized_extracted = _normalize_camera_name(extracted_name)

    # If we have intrinsics dict, try to find best match
    if camera_intrinsics is not None:
        # Normalize all intrinsic keys and find matches
        for camera_key in camera_intrinsics.keys():
            normalized_key = _normalize_camera_name(camera_key)

            # Check for exact match after normalization
            if normalized_extracted == normalized_key:
                return camera_key  # Return the original key from the dict

            # Also check if normalized key is a substring (for partial matches)
            if normalized_key in normalized_extracted or normalized_extracted in normalized_key:
                return camera_key

    return "default"


def _prepopulate_database_with_cameras(database_path: Path, images_dir: Path, camera_intrinsics: dict):
    """Pre-populate COLMAP database with cameras and images before feature extraction.

    This creates the database structure with fixed intrinsics so that feature extraction
    can use the correct camera parameters from the start.

    Args:
        database_path: Path to COLMAP database (will be created)
        images_dir: Directory containing images
        camera_intrinsics: Dictionary of camera intrinsics. Supports two formats:
            1. ROS camera_info format: {"K": [fx,0,cx,0,fy,cy,0,0,1], "D": [k1,k2,p1,p2,...], "width": W, "height": H}
            2. Direct format: {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": W, "height": H, "model": "PINHOLE"}
    """
    import sqlite3

    # Get all image files
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files:
        print("Error: No images found")
        return

    # Group images by camera name
    image_to_camera = {}  # Maps image filename to camera name
    camera_image_counts = {}  # Count images per camera

    for image_file in image_files:
        image_name = image_file.name
        camera_name = _extract_camera_name(image_name, camera_intrinsics)
        image_to_camera[image_name] = camera_name
        camera_image_counts[camera_name] = camera_image_counts.get(camera_name, 0) + 1

    # Get unique camera names
    unique_cameras = set(image_to_camera.values())
    print(f"    Found {len(unique_cameras)} unique camera types:")
    for cam_name in sorted(unique_cameras):
        print(f"      - {cam_name}: {camera_image_counts[cam_name]} images")

    # Create database and tables
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Create cameras table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        )
    """)

    # Create images table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )
    """)

    # Create other required tables for COLMAP
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB
        )
    """)

    conn.commit()

    # Create or update cameras with fixed intrinsics
    camera_id_map = {}  # Maps camera_name to camera_id
    print(f"\n    Setting fixed intrinsics for each camera:")

    # Camera model IDs (from COLMAP)
    MODEL_IDS = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10
    }

    camera_id_counter = 1
    for camera_name in sorted(unique_cameras):
        if camera_name not in camera_intrinsics:
            print(f"      ⚠ {camera_name}: No intrinsics found in JSON, will use COLMAP auto-calibration")
            continue

        intrinsics = camera_intrinsics[camera_name]

        # Parse ROS camera_info format
        # K is a 3x3 matrix stored as [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        # D is distortion coefficients array
        if "K" in intrinsics and "D" in intrinsics:
            K = intrinsics["K"]
            D = intrinsics["D"]
            width = intrinsics["width"]
            height = intrinsics["height"]
            distortion_model = intrinsics.get("distortion_model", "plumb_bob")

            # Extract camera parameters from K matrix
            fx = K[0]  # K[0, 0]
            fy = K[4]  # K[1, 1]
            cx = K[2]  # K[0, 2]
            cy = K[5]  # K[1, 2]

            # Determine COLMAP model based on distortion
            # ROS typically uses "plumb_bob" model which maps to OPENCV in COLMAP
            if len(D) >= 4 and any(d != 0 for d in D):
                # Has distortion - use OPENCV model
                model_id = MODEL_IDS["OPENCV"]
                model_name = "OPENCV"
                # OPENCV params: [fx, fy, cx, cy, k1, k2, p1, p2]
                params = [
                    fx, fy, cx, cy,
                    D[0] if len(D) > 0 else 0.0,  # k1
                    D[1] if len(D) > 1 else 0.0,  # k2
                    D[2] if len(D) > 2 else 0.0,  # p1
                    D[3] if len(D) > 3 else 0.0,  # p2
                ]
            else:
                # No significant distortion - use PINHOLE model
                model_id = MODEL_IDS["PINHOLE"]
                model_name = "PINHOLE"
                params = [fx, fy, cx, cy]

        # Legacy format support (direct fx, fy, cx, cy keys)
        elif "fx" in intrinsics and "fy" in intrinsics:
            width = intrinsics["width"]
            height = intrinsics["height"]
            fx = intrinsics["fx"]
            fy = intrinsics["fy"]
            cx = intrinsics["cx"]
            cy = intrinsics["cy"]

            model_name = intrinsics.get("model", "PINHOLE")
            if model_name == "PINHOLE":
                model_id = MODEL_IDS["PINHOLE"]
                params = [fx, fy, cx, cy]
            elif model_name == "OPENCV":
                model_id = MODEL_IDS["OPENCV"]
                params = [
                    fx, fy, cx, cy,
                    intrinsics.get("k1", 0.0),
                    intrinsics.get("k2", 0.0),
                    intrinsics.get("p1", 0.0),
                    intrinsics.get("p2", 0.0)
                ]
            else:
                print(f"      ⚠ {camera_name}: Unsupported camera model '{model_name}', skipping")
                continue
        else:
            print(f"      ⚠ {camera_name}: Invalid intrinsics format (missing K matrix or fx/fy), skipping")
            continue
        params_blob = bytes()
        for p in params:
            import struct
            params_blob += struct.pack('d', p)  # double precision

        cursor.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id_counter, model_id, width, height, params_blob, 1)
        )

        camera_id_map[camera_name] = camera_id_counter

        # Print detailed information for this camera
        print(f"      ✓ {camera_name} (camera_id={camera_id_counter})")
        print(f"          Model: {model_name}")
        print(f"          Resolution: {width}x{height}")
        print(f"          Focal length: fx={params[0]:.2f}, fy={params[1]:.2f}")
        print(f"          Principal point: cx={params[2]:.2f}, cy={params[3]:.2f}")
        if model_name == "OPENCV" and len(params) > 4:
            print(f"          Distortion: k1={params[4]:.6f}, k2={params[5]:.6f}, p1={params[6]:.6f}, p2={params[7]:.6f}")
        print(f"          Assigned to {camera_image_counts[camera_name]} images")

        camera_id_counter += 1

    # Insert images into the database with correct camera assignments
    print(f"\n    Inserting images into database...")
    image_id_counter = 1
    inserted_count = 0

    for image_file in image_files:
        image_name = image_file.name
        camera_name = image_to_camera[image_name]

        # Get the camera_id for this image
        if camera_name in camera_id_map:
            camera_id = camera_id_map[camera_name]

            # Insert image with no prior pose (will be estimated)
            cursor.execute(
                "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
                (image_id_counter, image_name, camera_id)
            )
            inserted_count += 1
            image_id_counter += 1
        else:
            print(f"      ⚠ Skipping {image_name}: No camera intrinsics for {camera_name}")

    conn.commit()
    conn.close()
    print(f"    Summary: Inserted {inserted_count} images with {len(camera_id_map)} camera(s)")


def run_colmap(output_dir: Path, images_dir: Path, gpu_index: int = 0,
               group_by: str = "video_camera",
               set_definitions: dict = None,
               intra_set_overlap: int = 30,
               inter_set_sample_rate: int = 5,
               camera_intrinsics: dict = None) -> Path:
    """Run COLMAP reconstruction pipeline using pycolmap with set-based grouping.

    Args:
        output_dir: Output directory
        images_dir: Directory containing images
        gpu_index: GPU device index
        group_by: How to group images (from config)
        set_definitions: Manual grouping into sets for matching strategy (from config)
        intra_set_overlap: Overlap for sequential matching within sets (from config)
        inter_set_sample_rate: Sample every Nth frame for cross-set matching (from config)
        camera_intrinsics: Dictionary mapping camera names to their intrinsic parameters
                          Format: {"camera_name": {"fx": ..., "fy": ..., "cx": ..., "cy": ...,
                                  "width": ..., "height": ..., "model": "PINHOLE"}}

    Returns:
        Path: Path to sparse reconstruction directory
    """
    import pycolmap

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing database if present
    if database_path.exists():
        database_path.unlink()

    # Set device for GPU acceleration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = pycolmap.Device.auto
    print(f"\nUsing device: auto (will use GPU if available, otherwise CPU)")

    # If intrinsics provided, pre-populate the database with cameras before feature extraction
    if camera_intrinsics is not None:
        print("\nStep 0: Setting up database with fixed camera intrinsics...")
        _prepopulate_database_with_cameras(database_path, images_dir, camera_intrinsics)

    # Feature extraction
    print("\nStep 1: Extracting features...")
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = 8192
    extraction_options.use_gpu = (device == pycolmap.Device.cuda or device == pycolmap.Device.auto)

    # When intrinsics are pre-populated, skip automatic feature extraction
    # and just extract features using the existing database structure
    if camera_intrinsics is None:
        print("  Using automatic camera calibration")
        camera_mode = pycolmap.CameraMode.SINGLE

        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(images_dir),
            camera_mode=camera_mode,
            extraction_options=extraction_options,
            device=device,
        )
    else:
        print("  Extracting features with fixed camera intrinsics...")
        # The database already has cameras and images, just extract features
        # Use COLMAP's feature extractor directly with the existing database
        import subprocess

        cmd = [
            "/home/yanbing/.local/bin/colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "0",  # Allow multiple cameras
            "--ImageReader.camera_model", "PINHOLE",  # Will use existing cameras
            "--SiftExtraction.max_num_features", str(extraction_options.sift.max_num_features),
            "--FeatureExtraction.use_gpu", "1" if extraction_options.use_gpu else "0",
        ]

        if extraction_options.use_gpu:
            cmd.extend(["--FeatureExtraction.gpu_index", str(gpu_index)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error running feature_extractor:")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"feature_extractor failed with return code {result.returncode}")

    # Get all images
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    if not image_files:
        print("Error: No images found")
        return None

    # Group images
    print(f"\nGrouping images by: {group_by}")
    groups = group_images_by_pattern(image_files, group_by)

    for group_key, images in groups.items():
        print(f"  {group_key}: {len(images)} images")

    # Create sets
    sets, set_configs = create_matching_sets(groups, set_definitions)
    print(f"\nCreated {len(sets)} sets for matching:")
    for set_name, images in sets.items():
        config = set_configs[set_name]
        matching_mode = config["matching_mode"]
        time_ordered = config["time_ordered"]

        time_info = ", time-ordered" if time_ordered else ""
        print(f"  {set_name}: {len(images)} images (mode={matching_mode}{time_info})")

    # Feature matching with hybrid strategy
    print("\nStep 2: Matching features...")

    if len(sets) > 1:
        print("Using hybrid matching strategy:")

        # Create pairs file for custom matching
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            pairs_file = f.name

            # 1. Intra-set matching (sequential or exhaustive)
            sets_with_sequential = [name for name, cfg in set_configs.items() if cfg["matching_mode"] == "sequential"]
            sets_with_exhaustive = [name for name, cfg in set_configs.items() if cfg["matching_mode"] == "exhaustive"]

            if sets_with_sequential:
                print(f"  - Sequential matching within {len(sets_with_sequential)} set(s) (overlap={intra_set_overlap})")
                print("    Generating intra-set pairs (sequential)...")
                for set_name in sets_with_sequential:
                    set_images = sets[set_name]
                    sorted_images = sorted(set_images)
                    for i in range(len(sorted_images)):
                        for j in range(i + 1, min(i + intra_set_overlap + 1, len(sorted_images))):
                            f.write(f"{sorted_images[i]} {sorted_images[j]}\n")

            if sets_with_exhaustive:
                print(f"  - Exhaustive matching within {len(sets_with_exhaustive)} set(s)")
                print("    Generating intra-set pairs (exhaustive)...")
                for set_name in sets_with_exhaustive:
                    set_images = sets[set_name]
                    sorted_images = sorted(set_images)
                    pair_count = 0
                    for i in range(len(sorted_images)):
                        for j in range(i + 1, len(sorted_images)):
                            f.write(f"{sorted_images[i]} {sorted_images[j]}\n")
                            pair_count += 1
                    print(f"      {set_name}: {pair_count} pairs")

            # 2. Cross-set matching (sampled exhaustive)
            print(f"  - Exhaustive matching between sets (sampling every {inter_set_sample_rate} frames)")
            print("    Generating inter-set pairs (sampled exhaustive)...")
            set_names = list(sets.keys())
            inter_pair_count = 0

            for i in range(len(set_names)):
                for j in range(i + 1, len(set_names)):
                    set1_name = set_names[i]
                    set2_name = set_names[j]

                    # Check if these sets should match with each other
                    config1 = set_configs[set1_name]
                    config2 = set_configs[set2_name]

                    # match_with = None means match with all sets
                    # match_with = [] means match with no sets
                    # match_with = ["set_name"] means match only with specified sets
                    should_match = False

                    if config1["match_with"] is None and config2["match_with"] is None:
                        # Both default to matching with all
                        should_match = True
                    elif config1["match_with"] is None:
                        # set1 matches with all, check if set2 allows set1
                        should_match = set1_name in config2["match_with"]
                    elif config2["match_with"] is None:
                        # set2 matches with all, check if set1 allows set2
                        should_match = set2_name in config1["match_with"]
                    else:
                        # Both have explicit match_with lists
                        should_match = (set2_name in config1["match_with"]) or (set1_name in config2["match_with"])

                    if not should_match:
                        continue

                    set1_images = sorted(sets[set1_name])[::inter_set_sample_rate]
                    set2_images = sorted(sets[set2_name])[::inter_set_sample_rate]

                    # Exhaustive between sampled images from different sets
                    for img1 in set1_images:
                        for img2 in set2_images:
                            f.write(f"{img1} {img2}\n")
                            inter_pair_count += 1

            if inter_pair_count > 0:
                print(f"      Total inter-set pairs: {inter_pair_count}")

        # Run matching with custom pairs using COLMAP CLI
        # Note: pycolmap doesn't have a direct match_pairs() function,
        # so we use the COLMAP command-line tool instead
        print("  Running feature matching...")

        import subprocess

        # Use COLMAP matches_importer to match custom pairs
        gpu_str = str(gpu_index)
        use_gpu = "1" if (device == pycolmap.Device.cuda or device == pycolmap.Device.auto) else "0"

        cmd = [
            "/home/yanbing/.local/bin/colmap", "matches_importer",
            "--database_path", str(database_path),
            "--match_list_path", pairs_file,
            "--match_type", "pairs",
            "--FeatureMatching.use_gpu", use_gpu,
        ]

        if gpu_index >= 0:
            cmd.extend(["--FeatureMatching.gpu_index", gpu_str])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error running matches_importer:")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"matches_importer failed with return code {result.returncode}")

        os.unlink(pairs_file)
        print("  Feature matching complete")

    else:
        # Single set: check matching mode from config
        set_name = list(sets.keys())[0]
        matching_mode = set_configs[set_name]["matching_mode"]
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = (device == pycolmap.Device.cuda or device == pycolmap.Device.auto)
        if matching_mode == "exhaustive":
            print("Single set detected, using exhaustive matching")
            pycolmap.match_exhaustive(
                database_path=str(database_path),
                matching_options=matching_options,
                device=device,
            )
        else:
            print("Single set detected, using sequential matching")
            pairing_options = pycolmap.SequentialPairingOptions(overlap=intra_set_overlap)
            pycolmap.match_sequential(
                database_path=str(database_path),
                pairing_options=pairing_options,
                matching_options=matching_options,
                device=device,
            )

    # Sparse reconstruction with pycolmap
    print("\nStep 3: Running sparse reconstruction...")

    # Configure mapper options
    mapper_options = pycolmap.IncrementalPipelineOptions()

    # If using fixed intrinsics, prevent refinement during bundle adjustment
    if camera_intrinsics is not None:
        print("  Disabling intrinsic refinement (using fixed parameters)")
        mapper_options.ba_refine_focal_length = False
        mapper_options.ba_refine_principal_point = False
        mapper_options.ba_refine_extra_params = False

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
        options=mapper_options,
    )

    # Report results
    print("\nStep 4: Analyzing reconstruction...")
    for idx, reconstruction in reconstructions.items():
        print(f"\nModel {idx}:")
        print(f"  Registered images: {reconstruction.num_reg_images()}")
        print(f"  3D points: {reconstruction.num_points3D()}")

    return sparse_dir


def run_hloc(output_dir: Path, images_dir: Path) -> Path:
    """Run hloc reconstruction pipeline with SuperPoint+SuperGlue.

    Args:
        output_dir: Output directory
        images_dir: Directory containing images

    Returns:
        Path: Path to sparse reconstruction directory, or None if failed
    """
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
