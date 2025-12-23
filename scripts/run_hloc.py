from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

# Paths
workspace = Path("/home/yanbinghan/fortyfive/colmap_dataset/num234_0/merged")
image_dir = workspace / "images"
output_dir = workspace / "sparse_hloc"

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Output files
features = workspace / "features.h5"
matches = workspace / "matches.h5"
pairs = workspace / "pairs.txt"

# Clean up old files if they exist
for f in [features, matches, pairs]:
    if f.exists():
        f.unlink()
        print(f"Removed existing {f.name}")

print("Step 1: Generating exhaustive image pairs...")
image_names = [f.name for f in sorted(image_dir.glob("*.jpg"))]
pairs_from_exhaustive.main(pairs, image_list=image_names)
print(f"Pairs file created: {pairs}")

print("\nStep 2: Extracting SuperPoint features...")
extract_features.main(
    extract_features.confs["superpoint_max"],
    image_dir,
    feature_path=features
)
print("Feature extraction complete")

print("\nStep 3: Matching with SuperGlue...")
match_features.main(
    match_features.confs["superglue"],
    pairs,
    features,
    matches=matches
)
print("Feature matching complete")

print("\nStep 4: Running COLMAP reconstruction...")
model = reconstruction.main(
    output_dir,
    image_dir,
    pairs,
    features,
    matches
)

print(f"\nReconstruction complete!")
print(f"Registered images: {model.num_reg_images()}")
print(f"3D points: {model.num_points3D()}")
