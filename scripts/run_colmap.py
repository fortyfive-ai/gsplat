import pycolmap
from pathlib import Path

# Paths
workspace = Path("/home/yanbinghan/fortyfive/colmap_dataset/num234_0/merged")
image_dir = workspace / "images"
database_path = workspace / "database.db"
output_path = workspace / "sparse"

# Create output directory
output_path.mkdir(exist_ok=True)

# Remove old database if exists
if database_path.exists():
    database_path.unlink()
    print("Removed existing database")

print("Step 1: Extracting features...")
pycolmap.extract_features(database_path, image_dir)
print("Feature extraction complete")

print("Step 2: Matching features (sequential)...")
pycolmap.match_sequential(database_path)
print("Feature matching complete")

print("Step 3: Running incremental mapping...")
reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_path)

print(f"\nReconstruction complete!")
print(f"Number of reconstructions: {len(reconstructions)}")

for idx, reconstruction in reconstructions.items():
    print(f"\nReconstruction {idx}:")
    print(f"  Registered images: {reconstruction.num_reg_images()}")
    print(f"  3D points: {reconstruction.num_points3D()}")
