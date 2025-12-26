"""Feature extraction and matching using COLMAP and hloc."""

import os
import tempfile
from pathlib import Path

from .image_grouping import group_images_by_pattern, create_matching_sets


def run_colmap(output_dir: Path, images_dir: Path, gpu_index: int = 0,
               group_by: str = "video_camera",
               set_definitions: dict = None,
               intra_set_overlap: int = 30,
               inter_set_sample_rate: int = 5) -> Path:
    """Run COLMAP reconstruction pipeline using pycolmap with set-based grouping.

    Args:
        output_dir: Output directory
        images_dir: Directory containing images
        gpu_index: GPU device index
        group_by: How to group images (from config)
        set_definitions: Manual grouping into sets for matching strategy (from config)
        intra_set_overlap: Overlap for sequential matching within sets (from config)
        inter_set_sample_rate: Sample every Nth frame for cross-set matching (from config)

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

    # Feature extraction
    print("\nStep 1: Extracting features...")
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = 8192
    extraction_options.use_gpu = (device == pycolmap.Device.cuda or device == pycolmap.Device.auto)

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_options,
        device=device,
    )

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
            "colmap", "matches_importer",
            "--database_path", str(database_path),
            "--match_list_path", pairs_file,
            "--match_type", "pairs",
            "--SiftMatching.use_gpu", use_gpu,
        ]

        if gpu_index >= 0:
            cmd.extend(["--SiftMatching.gpu_index", gpu_str])

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

        if matching_mode == "exhaustive":
            print("Single set detected, using exhaustive matching")
            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.use_gpu = (device == pycolmap.Device.cuda or device == pycolmap.Device.auto)

            pycolmap.match_exhaustive(
                database_path=str(database_path),
                matching_options=matching_options,
                device=device,
            )
        else:
            print("Single set detected, using sequential matching")
            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.use_gpu = (device == pycolmap.Device.cuda or device == pycolmap.Device.auto)
            pairing_options = pycolmap.SequentialPairingOptions(overlap=intra_set_overlap)

            pycolmap.match_sequential(
                database_path=str(database_path),
                pairing_options=pairing_options,
                matching_options=matching_options,
                device=device,
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
