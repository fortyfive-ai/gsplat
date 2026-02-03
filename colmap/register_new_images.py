#!/usr/bin/env python3
"""
Register additional images into an existing COLMAP reconstruction.

This script safely adds new images to an existing COLMAP model by:
1. Creating a timestamped backup of the original reconstruction
2. Copying new images to the existing images directory
3. Extracting features from new images
4. Matching new images with existing images
5. Registering new images into the reconstruction

The original reconstruction is always preserved and can be restored from backup.

Usage:
    # Basic usage (searches subdirectories recursively by default)
    python -m colmap.register_new_images \
        --colmap_dir /path/to/existing/colmap \
        --new_images /path/to/new/images/dir

    # With specific camera parameters
    python -m colmap.register_new_images \
        --colmap_dir /path/to/existing/colmap \
        --new_images /path/to/new/images/dir \
        --camera_model PINHOLE \
        --camera_params fx,fy,cx,cy

    # With multiple new image directories
    python -m colmap.register_new_images \
        --colmap_dir /path/to/existing/colmap \
        --new_images /path/to/images1,/path/to/images2 \
        --camera_model PINHOLE \
        --camera_params fx,fy,cx,cy

    # Disable recursive search (only look for images directly in specified directory)
    python -m colmap.register_new_images \
        --colmap_dir /path/to/existing/colmap \
        --new_images /path/to/images \
        --no-recursive

    # Restore from backup if something goes wrong
    python -m colmap.register_new_images \
        --colmap_dir /path/to/existing/colmap \
        --restore_backup /path/to/backup_dir
"""

import argparse
import shutil
import sqlite3
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import numpy as np

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
    raise ImportError("pycolmap is required. Install with: pip install pycolmap")

# Import helper function from existing codebase
from .core.feature_matching import _find_colmap_binary


class IncrementalRegistration:
    """Register new images into an existing COLMAP reconstruction."""

    def __init__(
        self,
        colmap_dir: Path,
        camera_model: str = "PINHOLE",
        camera_params: Optional[List[float]] = None,
        gpu_index: int = 0,
    ):
        """
        Args:
            colmap_dir: Path to existing COLMAP reconstruction directory
            camera_model: Camera model for new images (PINHOLE, SIMPLE_PINHOLE, OPENCV, etc.)
            camera_params: Camera intrinsic parameters for new images
            gpu_index: GPU index to use for feature extraction and matching
        """
        self.colmap_dir = Path(colmap_dir)
        self.camera_model = camera_model
        self.camera_params = camera_params
        self.gpu_index = gpu_index

        # Paths
        self.sparse_dir = self._find_sparse_dir()
        self.images_dir = self.colmap_dir / "images"
        self.database_path = self.colmap_dir / "database.db"

        # Validate paths
        if not self.sparse_dir.exists():
            raise FileNotFoundError(f"Sparse reconstruction not found: {self.sparse_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")

        print(f"COLMAP directory: {self.colmap_dir}")
        print(f"Sparse reconstruction: {self.sparse_dir}")
        print(f"Images directory: {self.images_dir}")
        print(f"Database: {self.database_path}")

    def _find_sparse_dir(self) -> Path:
        """Find the sparse reconstruction directory."""
        # Try common locations
        candidates = [
            self.colmap_dir / "sparse" / "0",
            self.colmap_dir / "sparse",
            self.colmap_dir,
        ]

        for candidate in candidates:
            if (candidate / "cameras.bin").exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find sparse reconstruction in {self.colmap_dir}. "
            f"Expected to find cameras.bin in one of: {[str(c) for c in candidates]}"
        )

    def create_backup(self, new_image_names: List[str] = None) -> Path:
        """Create a timestamped backup of the reconstruction.

        Args:
            new_image_names: Optional list of new image filenames to track for restoration

        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.colmap_dir.parent / f"{self.colmap_dir.name}_backup_{timestamp}"

        print(f"\nCreating backup: {backup_dir}")

        # Backup sparse reconstruction
        backup_sparse = backup_dir / "sparse"
        backup_sparse.mkdir(parents=True, exist_ok=True)

        # Copy all files from sparse directory
        for file in self.sparse_dir.glob("*"):
            if file.is_file():
                shutil.copy2(file, backup_sparse / file.name)

        # Backup database
        if self.database_path.exists():
            shutil.copy2(self.database_path, backup_dir / "database.db")

        # Create backup info file
        info_file = backup_dir / "backup_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Backup created: {datetime.now()}\n")
            f.write(f"Original directory: {self.colmap_dir}\n")
            f.write(f"Sparse directory: {self.sparse_dir}\n")

        # Track new images for restoration
        if new_image_names:
            images_list_file = backup_dir / "new_images.txt"
            with open(images_list_file, 'w') as f:
                for img_name in new_image_names:
                    f.write(f"{img_name}\n")
            print(f"  Tracked {len(new_image_names)} new images for restoration")

        print(f"✓ Backup created successfully")
        print(f"  Location: {backup_dir}")
        return backup_dir

    @staticmethod
    def restore_backup(colmap_dir: Path, backup_dir: Path):
        """Restore reconstruction from backup.

        Args:
            colmap_dir: Target COLMAP directory to restore to
            backup_dir: Backup directory to restore from
        """
        print(f"\nRestoring from backup: {backup_dir}")

        backup_sparse = backup_dir / "sparse"
        backup_db = backup_dir / "database.db"
        new_images_list = backup_dir / "new_images.txt"

        if not backup_sparse.exists():
            raise FileNotFoundError(f"Backup sparse directory not found: {backup_sparse}")

        # Find target sparse directory
        target_sparse_candidates = [
            colmap_dir / "sparse" / "0",
            colmap_dir / "sparse",
        ]
        target_sparse = None
        for candidate in target_sparse_candidates:
            if candidate.exists() or candidate.parent.exists():
                target_sparse = candidate
                break

        if target_sparse is None:
            raise FileNotFoundError(f"Could not determine target sparse directory in {colmap_dir}")

        # Restore sparse reconstruction
        target_sparse.parent.mkdir(parents=True, exist_ok=True)
        target_sparse.mkdir(parents=True, exist_ok=True)

        for file in backup_sparse.glob("*"):
            if file.is_file():
                shutil.copy2(file, target_sparse / file.name)
                print(f"  Restored: {file.name}")

        # Restore database
        if backup_db.exists():
            target_db = colmap_dir / "database.db"
            shutil.copy2(backup_db, target_db)
            print(f"  Restored: database.db")

        # Remove new images that were added (if tracked)
        if new_images_list.exists():
            images_dir = colmap_dir / "images"
            with open(new_images_list, 'r') as f:
                new_images = [line.strip() for line in f if line.strip()]

            print(f"\nRemoving {len(new_images)} added images...")
            removed_count = 0
            for img_name in new_images:
                img_path = images_dir / img_name
                if img_path.exists():
                    img_path.unlink()
                    removed_count += 1

            print(f"  Removed {removed_count} images")

        print(f"\n✓ Restore complete!")

    def _get_images_to_copy(self, new_image_dirs: List[Path], recursive: bool = True) -> List[str]:
        """Determine which images will be copied (without actually copying).

        Args:
            new_image_dirs: List of directories containing new images
            recursive: If True, search subdirectories recursively

        Returns:
            List of image filenames that would be copied
        """
        images_to_copy = []

        for img_dir in new_image_dirs:
            if not img_dir.exists():
                continue

            # Find all images (with or without recursion)
            image_extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
            images = []

            if recursive:
                for ext in image_extensions:
                    images.extend(list(img_dir.rglob(ext)))
            else:
                for ext in image_extensions:
                    images.extend(list(img_dir.glob(ext)))

            if not images:
                continue

            # Determine unique names for images
            for img_path in sorted(images):
                relative_path = img_path.relative_to(img_dir)

                # Replace directory separators with underscores
                if len(relative_path.parts) > 1:
                    unique_name = "_".join(relative_path.parts)
                else:
                    unique_name = img_path.name

                target_path = self.images_dir / unique_name

                # Only include if it doesn't already exist
                if not target_path.exists():
                    images_to_copy.append(unique_name)

        return images_to_copy

    def copy_new_images(self, new_image_dirs: List[Path], recursive: bool = True) -> List[Path]:
        """Copy new images to the images directory.

        Args:
            new_image_dirs: List of directories containing new images
            recursive: If True, search subdirectories recursively

        Returns:
            List of paths to copied images (relative to images_dir)
        """
        print(f"\nCopying new images...")

        new_image_paths = []
        total_copied = 0

        for img_dir in new_image_dirs:
            if not img_dir.exists():
                print(f"Warning: Directory not found: {img_dir}")
                continue

            # Find all images (with or without recursion)
            image_extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
            images = []

            if recursive:
                # Search recursively in subdirectories
                for ext in image_extensions:
                    images.extend(list(img_dir.rglob(ext)))
            else:
                # Only search in the specified directory
                for ext in image_extensions:
                    images.extend(list(img_dir.glob(ext)))

            if not images:
                print(f"Warning: No images found in {img_dir}")
                continue

            print(f"  Found {len(images)} images in {img_dir} (recursive={recursive})")

            # Copy images with unique naming to avoid conflicts
            for img_path in sorted(images):
                # Create a unique name that preserves directory structure
                # For example: front/frame_0001.jpg -> front_frame_0001.jpg
                relative_path = img_path.relative_to(img_dir)

                # Replace directory separators with underscores
                if len(relative_path.parts) > 1:
                    # Has subdirectories
                    unique_name = "_".join(relative_path.parts)
                else:
                    # No subdirectories
                    unique_name = img_path.name

                target_path = self.images_dir / unique_name

                # Check if image already exists
                if target_path.exists():
                    print(f"  Warning: Image already exists, skipping: {unique_name}")
                    continue

                shutil.copy2(img_path, target_path)
                new_image_paths.append(unique_name)
                total_copied += 1

        print(f"✓ Copied {total_copied} new images")
        return new_image_paths

    def get_or_create_camera(self) -> int:
        """Get existing camera or create new one in database.

        Returns:
            Camera ID in the database
        """
        # Load existing reconstruction to check cameras
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))

        # If camera params not specified, use the first camera from reconstruction
        if self.camera_params is None:
            if len(reconstruction.cameras) == 0:
                raise ValueError("No cameras in reconstruction and no camera_params specified")

            camera_id = list(reconstruction.cameras.keys())[0]
            camera = reconstruction.cameras[camera_id]
            print(f"\nUsing existing camera {camera_id} from reconstruction:")
            print(f"  Model: {camera.model.name}")
            print(f"  Size: {camera.width} x {camera.height}")
            print(f"  Params: {camera.params}")
            return camera_id

        # Otherwise, check if a matching camera exists
        for cam_id, cam in reconstruction.cameras.items():
            if cam.model.name == self.camera_model:
                # Check if parameters match (with tolerance)
                if np.allclose(cam.params[:len(self.camera_params)], self.camera_params, rtol=1e-3):
                    print(f"\nFound matching camera {cam_id} in reconstruction")
                    return cam_id

        # Create new camera in database
        print(f"\nCreating new camera with model {self.camera_model}")
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()

        # Get next camera ID
        cursor.execute("SELECT MAX(camera_id) FROM cameras")
        result = cursor.fetchone()[0]
        new_camera_id = 1 if result is None else result + 1

        # Map model name to model ID
        model_name_to_id = {
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
            "THIN_PRISM_FISHEYE": 10,
        }
        model_id = model_name_to_id.get(self.camera_model, 1)  # Default to PINHOLE

        # Estimate image size (will be updated during feature extraction)
        width, height = 1920, 1080  # Default placeholder

        # Prepare params (pad to required length)
        num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
        n = num_params.get(model_id, 4)
        params = list(self.camera_params) + [0.0] * (n - len(self.camera_params))
        params = params[:n]

        # Insert into database
        cursor.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
            (new_camera_id, model_id, width, height, np.array(params).tobytes(), 1)
        )
        conn.commit()
        conn.close()

        print(f"  Created camera {new_camera_id}")
        return new_camera_id

    def extract_features_for_new_images(self, new_image_names: List[str]):
        """Extract SIFT features for new images.

        Args:
            new_image_names: List of new image filenames
        """
        print(f"\nExtracting features for {len(new_image_names)} new images...")

        # Configure feature extraction (matching the existing codebase pattern)
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift.max_num_features = 8192
        extraction_options.use_gpu = self.gpu_index >= 0

        device = pycolmap.Device.auto if self.gpu_index >= 0 else pycolmap.Device.cpu

        # Extract features only for new images
        print("  Running SIFT feature extraction...")
        pycolmap.extract_features(
            database_path=str(self.database_path),
            image_path=str(self.images_dir),
            extraction_options=extraction_options,
            device=device,
            image_names=new_image_names,
        )

        print(f"✓ Feature extraction complete")

    def match_new_images(self, new_image_names: List[str]):
        """Match new images with existing images in the reconstruction.

        Args:
            new_image_names: List of new image filenames
        """
        print(f"\nMatching new images with existing reconstruction...")

        # Get list of existing registered images
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))
        existing_image_names = [img.name for img in reconstruction.images.values()]

        print(f"  {len(existing_image_names)} existing images in reconstruction")
        print(f"  {len(new_image_names)} new images to match")

        # Create pairs file for matching (new images with all existing images)
        pairs_file = self.colmap_dir / "image_pairs_incremental.txt"
        with open(pairs_file, 'w') as f:
            for new_img in new_image_names:
                for existing_img in existing_image_names:
                    f.write(f"{new_img} {existing_img}\n")

        total_pairs = len(new_image_names) * len(existing_image_names)
        print(f"  Generated {total_pairs} image pairs for matching")

        # Run feature matching using COLMAP CLI (pycolmap doesn't have match_pairs)
        print("  Running feature matching...")

        colmap_bin = _find_colmap_binary()
        use_gpu = "1" if self.gpu_index >= 0 else "0"

        cmd = [
            colmap_bin, "matches_importer",
            "--database_path", str(self.database_path),
            "--match_list_path", str(pairs_file),
            "--match_type", "pairs",
            "--SiftMatching.use_gpu", use_gpu,
        ]

        if self.gpu_index >= 0:
            cmd.extend(["--SiftMatching.gpu_index", str(self.gpu_index)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error running matches_importer:")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"matches_importer failed with return code {result.returncode}")

        # Clean up pairs file
        pairs_file.unlink()

        print(f"✓ Feature matching complete")

    def register_images(self) -> pycolmap.Reconstruction:
        """Register new images into the reconstruction using image registration.

        Returns:
            Updated reconstruction
        """
        print(f"\nRegistering new images into reconstruction...")

        # Load existing reconstruction
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))

        initial_num_images = reconstruction.num_reg_images()
        initial_num_points = reconstruction.num_points3D()

        print(f"  Initial state: {initial_num_images} images, {initial_num_points} 3D points")

        # Configure registration options
        options = pycolmap.IncrementalMapperOptions()
        options.min_num_matches = 15
        options.ba_refine_focal_length = True
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = True

        # If camera params specified, disable refinement
        if self.camera_params is not None:
            options.ba_refine_focal_length = False
            options.ba_refine_principal_point = False
            options.ba_refine_extra_params = False
            print(f"  Using fixed camera intrinsics (no refinement)")

        # Register images
        print("  Running image registration...")
        num_registered = reconstruction.register_next_image(
            options=options,
            database_path=str(self.database_path),
        )

        # If register_next_image doesn't work, try full mapper
        if num_registered == 0:
            print("  register_next_image returned 0, trying incremental_mapping...")

            # Run incremental mapping (will skip already registered images)
            mapper_options = pycolmap.IncrementalPipelineOptions()
            mapper_options.min_num_matches = 15

            if self.camera_params is not None:
                mapper_options.ba_refine_focal_length = False
                mapper_options.ba_refine_principal_point = False
                mapper_options.ba_refine_extra_params = False

            # Create temporary output directory
            temp_output = self.colmap_dir / "sparse_temp"
            temp_output.mkdir(exist_ok=True)

            reconstructions = pycolmap.incremental_mapping(
                database_path=str(self.database_path),
                image_path=str(self.images_dir),
                output_path=str(temp_output),
                options=mapper_options,
                input_path=str(self.sparse_dir),  # Start from existing reconstruction
            )

            if len(reconstructions) > 0:
                # Use the first (largest) reconstruction
                reconstruction = reconstructions[0]

                # Write to original sparse directory
                reconstruction.write(str(self.sparse_dir))

                # Clean up temp directory
                shutil.rmtree(temp_output)
            else:
                print("  Warning: incremental_mapping returned no reconstructions")
                # Clean up temp directory
                shutil.rmtree(temp_output)
        else:
            # Save updated reconstruction
            reconstruction.write(str(self.sparse_dir))

        final_num_images = reconstruction.num_reg_images()
        final_num_points = reconstruction.num_points3D()

        print(f"\n✓ Registration complete!")
        print(f"  Final state: {final_num_images} images (+{final_num_images - initial_num_images}), "
              f"{final_num_points} 3D points (+{final_num_points - initial_num_points})")

        return reconstruction

    def run(self, new_image_dirs: List[Path], recursive: bool = True) -> Path:
        """Complete pipeline to register new images.

        Args:
            new_image_dirs: List of directories containing new images
            recursive: If True, search subdirectories recursively

        Returns:
            Path to backup directory
        """
        # Step 0: Determine which images will be added
        images_to_add = self._get_images_to_copy(new_image_dirs, recursive=recursive)

        if not images_to_add:
            print("\nNo new images to register!")
            print("All images already exist in the images directory.")
            return None

        # Step 1: Create backup (with list of images that will be added)
        backup_dir = self.create_backup(new_image_names=images_to_add)

        try:
            # Step 2: Copy new images
            new_image_names = self.copy_new_images(new_image_dirs, recursive=recursive)

            if not new_image_names:
                print("\nNo new images to register!")
                return backup_dir

            # Step 3: Get or create camera
            self.get_or_create_camera()

            # Step 4: Extract features
            self.extract_features_for_new_images(new_image_names)

            # Step 5: Match features
            self.match_new_images(new_image_names)

            # Step 6: Register images
            reconstruction = self.register_images()

            print(f"\n{'='*60}")
            print(f"SUCCESS! New images registered into reconstruction")
            print(f"{'='*60}")
            print(f"Backup saved to: {backup_dir}")
            print(f"Updated reconstruction: {self.sparse_dir}")
            print(f"\nTo restore from backup if needed:")
            print(f"  python -m colmap.register_new_images \\")
            print(f"    --colmap_dir {self.colmap_dir} \\")
            print(f"    --restore_backup {backup_dir}")

            return backup_dir

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR during registration!")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"\nYour original reconstruction is safe in: {backup_dir}")
            print(f"\nTo restore from backup:")
            print(f"  python -m colmap.register_new_images \\")
            print(f"    --colmap_dir {self.colmap_dir} \\")
            print(f"    --restore_backup {backup_dir}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Register additional images into an existing COLMAP reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register new images from a directory
  python -m colmap.register_new_images \\
      --colmap_dir /path/to/existing/colmap \\
      --new_images /path/to/new/images \\
      --camera_model PINHOLE \\
      --camera_params 1000,1000,960,540

  # Register from multiple directories
  python -m colmap.register_new_images \\
      --colmap_dir /path/to/existing/colmap \\
      --new_images /path/to/images1,/path/to/images2

  # Restore from backup
  python -m colmap.register_new_images \\
      --colmap_dir /path/to/colmap \\
      --restore_backup /path/to/backup_dir
        """
    )

    parser.add_argument(
        "--colmap_dir",
        type=str,
        required=True,
        help="Path to existing COLMAP reconstruction directory"
    )
    parser.add_argument(
        "--new_images",
        type=str,
        default=None,
        help="Path to directory/directories containing new images (comma-separated for multiple)"
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default=None,
        choices=["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"],
        help="Camera model for new images (if not specified, uses existing camera from reconstruction)"
    )
    parser.add_argument(
        "--camera_params",
        type=str,
        default=None,
        help="Camera parameters (comma-separated): fx,fy,cx,cy for PINHOLE, f,cx,cy for SIMPLE_PINHOLE"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (use -1 for CPU)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search for images recursively in subdirectories (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only search for images directly in specified directories (not in subdirectories)"
    )
    parser.add_argument(
        "--restore_backup",
        type=str,
        default=None,
        help="Restore reconstruction from backup directory"
    )

    args = parser.parse_args()

    colmap_dir = Path(args.colmap_dir)

    # Handle restore mode
    if args.restore_backup:
        backup_dir = Path(args.restore_backup)
        if not backup_dir.exists():
            print(f"Error: Backup directory not found: {backup_dir}")
            return 1

        IncrementalRegistration.restore_backup(colmap_dir, backup_dir)
        return 0

    # Handle registration mode
    if not args.new_images:
        print("Error: --new_images is required (or use --restore_backup to restore)")
        return 1

    # Parse comma-separated image directories
    image_dir_strings = [d.strip() for d in args.new_images.split(',')]
    new_image_dirs = [Path(d) for d in image_dir_strings]

    # Parse camera params
    camera_params = None
    if args.camera_params:
        camera_params = [float(x) for x in args.camera_params.split(",")]

    # Create registrar
    registrar = IncrementalRegistration(
        colmap_dir=colmap_dir,
        camera_model=args.camera_model or "PINHOLE",
        camera_params=camera_params,
        gpu_index=args.gpu,
    )

    # Run registration
    backup_dir = registrar.run(new_image_dirs, recursive=args.recursive)

    return 0


if __name__ == "__main__":
    exit(main())
