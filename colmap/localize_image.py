"""
Real-time camera pose estimation for a new image using pre-built COLMAP reconstruction.

This module provides fast and accurate localization by:
1. Building a feature database from COLMAP's 3D points
2. Matching new image features against the database
3. Solving PnP with RANSAC for robust pose estimation

Usage:
    python -m colmap.localize_image \
        --colmap_dir /path/to/colmap/sparse/0 \
        --image_path /path/to/new_image.jpg \
        --camera_model PINHOLE \
        --camera_params fx,fy,cx,cy
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
    print("Warning: pycolmap not installed. Some features may be limited.")


@dataclass
class CameraPose:
    """Estimated camera pose result."""
    R: np.ndarray  # 3x3 rotation matrix (camera-to-world)
    t: np.ndarray  # 3x1 translation (camera center in world)
    qvec: np.ndarray  # quaternion (w, x, y, z)
    num_inliers: int
    inlier_ratio: float
    estimation_time_ms: float
    success: bool


class COLMAPDatabase:
    """Load and index COLMAP reconstruction for fast feature matching."""

    def __init__(
        self,
        colmap_dir: str,
        database_path: Optional[str] = None,
        max_points: int = 100000,
        use_gpu: bool = True
    ):
        """
        Args:
            colmap_dir: Path to COLMAP sparse reconstruction (containing cameras.bin, images.bin, points3D.bin)
            database_path: Optional explicit path to database.db. If None, will auto-detect.
            max_points: Maximum number of 3D points to load (limits memory usage)
            use_gpu: Use GPU for feature extraction if available
        """
        self.colmap_dir = Path(colmap_dir)
        self.database_path = Path(database_path) if database_path else None
        self.max_points = max_points
        self.use_gpu = use_gpu

        # Load reconstruction
        self.cameras, self.images, self.points3D = self._load_reconstruction()

        # Build 3D point database with descriptors
        self.point_ids, self.point_xyz, self.point_descriptors = self._build_point_database(max_points)

        # Initialize feature matcher
        self.matcher = self._init_matcher()

        print(f"Loaded {len(self.points3D)} 3D points, {len(self.point_ids)} with descriptors")

    def _load_reconstruction(self) -> Tuple[Dict, Dict, Dict]:
        """Load COLMAP binary files."""
        cameras_path = self.colmap_dir / "cameras.bin"
        images_path = self.colmap_dir / "images.bin"
        points3D_path = self.colmap_dir / "points3D.bin"

        # Try pycolmap first (faster)
        if HAS_PYCOLMAP:
            reconstruction = pycolmap.Reconstruction(str(self.colmap_dir))
            # pycolmap already returns dicts keyed by ID
            cameras = dict(reconstruction.cameras)
            images = dict(reconstruction.images)
            points3D = dict(reconstruction.points3D)
            return cameras, images, points3D

        # Fallback to manual binary reading
        cameras = read_cameras_binary(str(cameras_path))
        images = read_images_binary(str(images_path))
        points3D = read_points3D_binary(str(points3D_path))
        return cameras, images, points3D

    def _build_point_database(self, max_points: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 3D points and their associated descriptors from the reconstruction.

        Args:
            max_points: Maximum number of 3D points to load (for memory efficiency)
        """
        # We need to get descriptors from the database.db file
        if self.database_path is not None:
            db_path = self.database_path
        else:
            # COLMAP structure: colmap_output/database.db and colmap_output/sparse/0/
            # So from sparse/0, we need to go up 2 levels
            db_path = self.colmap_dir.parent.parent / "database.db"

            # Also try one level up (in case colmap_dir points to sparse/ instead of sparse/0/)
            if not db_path.exists():
                db_path = self.colmap_dir.parent / "database.db"

        if not db_path.exists():
            print(f"Warning: database.db not found")
            print(f"  Tried: {self.colmap_dir.parent.parent / 'database.db'}")
            print(f"  Tried: {self.colmap_dir.parent / 'database.db'}")
            print("Will use image-based matching instead of direct 3D point matching")
            return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 128)

        print(f"Found database at: {db_path}")

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get image name to id mapping from database
        cursor.execute("SELECT image_id, name FROM images")
        db_images = {name: img_id for img_id, name in cursor.fetchall()}

        # Build mapping from COLMAP image_id to database image_id
        colmap_to_db = {}
        for img_id, img in self.images.items():
            if HAS_PYCOLMAP:
                img_name = img.name
            else:
                img_name = img['name']
            if img_name in db_images:
                colmap_to_db[img_id] = db_images[img_name]

        # First pass: group 3D points by their observing image (for efficient descriptor loading)
        # Only use points with higher track length (more reliable)
        print("Grouping 3D points by observing image...")
        image_to_points = {}  # db_img_id -> list of (pt_id, xyz, pt2d_idx)

        # Sort points by track length (more observations = more reliable)
        sorted_points = []
        for pt_id, pt in self.points3D.items():
            if HAS_PYCOLMAP:
                track_len = len(pt.track.elements)
            else:
                track_len = len(pt['track'])
            sorted_points.append((pt_id, pt, track_len))

        sorted_points.sort(key=lambda x: -x[2])  # Sort by track length descending

        # Limit to max_points
        if len(sorted_points) > max_points:
            print(f"Limiting to {max_points} points (from {len(sorted_points)} total)")
            sorted_points = sorted_points[:max_points]

        for pt_id, pt, _ in sorted_points:
            if HAS_PYCOLMAP:
                xyz = pt.xyz
                track_elements = pt.track.elements
            else:
                xyz = pt['xyz']
                track_elements = pt['track']

            if len(track_elements) > 0:
                if HAS_PYCOLMAP:
                    img_id, pt2d_idx = track_elements[0].image_id, track_elements[0].point2D_idx
                else:
                    img_id, pt2d_idx = track_elements[0]

                if img_id in colmap_to_db:
                    db_img_id = colmap_to_db[img_id]
                    if db_img_id not in image_to_points:
                        image_to_points[db_img_id] = []
                    image_to_points[db_img_id].append((pt_id, xyz, pt2d_idx))

        # Second pass: load descriptors image by image (memory efficient)
        print(f"Loading descriptors from {len(image_to_points)} images...")
        point_ids = []
        point_xyz = []
        point_descriptors = []

        for i, (db_img_id, pts) in enumerate(image_to_points.items()):
            if (i + 1) % 50 == 0:
                print(f"  Processing image {i+1}/{len(image_to_points)}...")

            cursor.execute(
                "SELECT data FROM descriptors WHERE image_id = ?",
                (db_img_id,)
            )
            result = cursor.fetchone()
            if result is None:
                continue

            descriptors = np.frombuffer(result[0], dtype=np.uint8)
            num_features = len(descriptors) // 128
            if num_features == 0:
                continue

            descriptors = descriptors.reshape(num_features, 128)

            # Extract all needed descriptors from this image
            for pt_id, xyz, pt2d_idx in pts:
                if pt2d_idx < len(descriptors):
                    point_ids.append(pt_id)
                    point_xyz.append(xyz)
                    point_descriptors.append(descriptors[pt2d_idx])

        conn.close()

        if len(point_ids) == 0:
            return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 128)

        print(f"Loaded {len(point_ids)} 3D points with descriptors")
        return (
            np.array(point_ids),
            np.array(point_xyz),
            np.array(point_descriptors, dtype=np.uint8)
        )

    def _init_matcher(self):
        """Initialize feature matcher (FLANN for speed)."""
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)


class ImageLocalizer:
    """Localize new images against a COLMAP reconstruction."""

    def __init__(
        self,
        colmap_database: COLMAPDatabase,
        camera_model: str = "PINHOLE",
        camera_params: Optional[List[float]] = None,
        use_gpu: bool = True,
        num_features: int = 8192,
    ):
        """
        Args:
            colmap_database: Pre-built COLMAP database
            camera_model: Camera model (PINHOLE, SIMPLE_PINHOLE, OPENCV, etc.)
            camera_params: Camera intrinsic parameters [fx, fy, cx, cy] or [f, cx, cy]
            use_gpu: Use GPU for feature extraction
            num_features: Maximum number of SIFT features to extract
        """
        self.db = colmap_database
        self.camera_model = camera_model
        self.camera_params = camera_params
        self.use_gpu = use_gpu
        self.num_features = num_features

        # Initialize pycolmap SIFT detector (matches COLMAP's feature extraction)
        options = pycolmap.FeatureExtractionOptions()
        options.sift.max_num_features = num_features
        self.sift = pycolmap.Sift(options)

        # Convert database descriptors to L1_ROOT normalized format (same as pycolmap)
        # COLMAP stores descriptors as uint8, we need to convert to normalized float32
        if len(self.db.point_descriptors) > 0:
            self.db_descriptors_normalized = self._normalize_descriptors(
                self.db.point_descriptors.astype(np.float32)
            )
            self._build_flann_index()

    def _normalize_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Apply L1_ROOT normalization to match COLMAP's descriptor format."""
        # L1 normalize
        l1_norms = np.sum(np.abs(descriptors), axis=1, keepdims=True)
        l1_norms = np.maximum(l1_norms, 1e-8)
        descriptors = descriptors / l1_norms
        # Square root (L1_ROOT)
        descriptors = np.sqrt(descriptors)
        return descriptors.astype(np.float32)

    def _build_flann_index(self):
        """Build FLANN index for fast nearest neighbor search."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Add normalized descriptors to index
        self.flann.add([self.db_descriptors_normalized])
        self.flann.train()

    def _extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract SIFT features from image using pycolmap (matches COLMAP's extraction)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # pycolmap.Sift.extract returns (keypoints, descriptors)
        # keypoints is Nx6 array: [x, y, a11, a12, a21, a22] (affine shape)
        # descriptors is Nx128 float32 array (L1_ROOT normalized)
        keypoints, descriptors = self.sift.extract(gray)

        if descriptors is None or len(descriptors) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)

        # Extract x, y coordinates from keypoints
        pts = keypoints[:, :2]  # First two columns are x, y
        return pts, descriptors

    def _match_features(
        self,
        descriptors: np.ndarray,
        ratio_threshold: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match 2D features to 3D points using ratio test."""
        if len(descriptors) == 0 or len(self.db.point_descriptors) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)

        # KNN matching with k=2 for ratio test
        matches = self.flann.knnMatch(
            descriptors.astype(np.float32),
            k=2
        )

        # Apply ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)

        # Keep only unique 3D point matches (best match per 3D point)
        # This handles cases where multiple 2D points match the same 3D point
        best_matches = {}
        for m in good_matches:
            train_idx = m.trainIdx
            if train_idx not in best_matches or m.distance < best_matches[train_idx].distance:
                best_matches[train_idx] = m

        unique_matches = list(best_matches.values())

        # Get matched indices
        query_indices = np.array([m.queryIdx for m in unique_matches])
        train_indices = np.array([m.trainIdx for m in unique_matches])

        return query_indices, train_indices

    def _get_camera_matrix(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Get camera intrinsic matrix."""
        h, w = image_shape[:2]

        if self.camera_params is not None:
            if self.camera_model == "SIMPLE_PINHOLE":
                f, cx, cy = self.camera_params[:3]
                fx = fy = f
            else:  # PINHOLE, OPENCV, etc.
                fx, fy, cx, cy = self.camera_params[:4]
        else:
            # Use a reasonable default (assume image center is principal point)
            fx = fy = max(w, h)
            cx, cy = w / 2, h / 2

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def _get_distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients."""
        if self.camera_params is None:
            return np.zeros(4)

        if self.camera_model == "OPENCV" and len(self.camera_params) >= 8:
            # k1, k2, p1, p2
            return np.array(self.camera_params[4:8])

        return np.zeros(4)

    def localize(
        self,
        image: np.ndarray,
        ransac_threshold: float = 8.0,
        min_inliers: int = 12,
    ) -> CameraPose:
        """
        Estimate camera pose for a new image.

        Args:
            image: Input image (BGR or grayscale)
            ransac_threshold: RANSAC inlier threshold in pixels
            min_inliers: Minimum number of inliers for successful localization

        Returns:
            CameraPose object with estimated pose and statistics
        """
        start_time = time.time()

        # Extract features
        keypoints, descriptors = self._extract_features(image)

        if len(keypoints) < min_inliers:
            return CameraPose(
                R=np.eye(3), t=np.zeros(3), qvec=np.array([1, 0, 0, 0]),
                num_inliers=0, inlier_ratio=0.0,
                estimation_time_ms=(time.time() - start_time) * 1000,
                success=False
            )

        # Match to 3D points
        query_idx, train_idx = self._match_features(descriptors)

        if len(query_idx) < min_inliers:
            return CameraPose(
                R=np.eye(3), t=np.zeros(3), qvec=np.array([1, 0, 0, 0]),
                num_inliers=len(query_idx), inlier_ratio=0.0,
                estimation_time_ms=(time.time() - start_time) * 1000,
                success=False
            )

        # Get 2D-3D correspondences
        points_2d = keypoints[query_idx]
        points_3d = self.db.point_xyz[train_idx]

        # Get camera parameters
        K = self._get_camera_matrix(image.shape)
        dist_coeffs = self._get_distortion_coeffs()

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
            reprojectionError=ransac_threshold,
            confidence=0.9999,
            iterationsCount=10000,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if not success or inliers is None or len(inliers) < min_inliers:
            return CameraPose(
                R=np.eye(3), t=np.zeros(3), qvec=np.array([1, 0, 0, 0]),
                num_inliers=0 if inliers is None else len(inliers),
                inlier_ratio=0.0,
                estimation_time_ms=elapsed_ms,
                success=False
            )

        # Refine with inliers only
        inlier_mask = inliers.flatten()
        _, rvec, tvec = cv2.solvePnP(
            points_3d[inlier_mask].astype(np.float64),
            points_2d[inlier_mask].astype(np.float64),
            K,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Convert to rotation matrix and camera center
        R_cv, _ = cv2.Rodrigues(rvec)

        # OpenCV gives world-to-camera, we want camera-to-world
        R_c2w = R_cv.T
        t_c2w = -R_cv.T @ tvec.flatten()

        # Convert to quaternion (w, x, y, z)
        qvec = rotation_matrix_to_quaternion(R_c2w)

        return CameraPose(
            R=R_c2w,
            t=t_c2w,
            qvec=qvec,
            num_inliers=len(inliers),
            inlier_ratio=len(inliers) / len(query_idx),
            estimation_time_ms=elapsed_ms,
            success=True
        )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


# Binary file readers (fallback if pycolmap not available)
def read_cameras_binary(path: str) -> Dict:
    """Read COLMAP cameras.bin file."""
    import struct
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            # Number of params depends on model
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
            n = num_params.get(model_id, 4)
            params = struct.unpack(f"<{n}d", f.read(8 * n))

            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_binary(path: str) -> Dict:
    """Read COLMAP images.bin file."""
    import struct
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read image name
            name = ""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name += char.decode("utf-8")

            # Read 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            points2D = []
            for _ in range(num_points2D):
                x, y = struct.unpack("<2d", f.read(16))
                point3D_id = struct.unpack("<q", f.read(8))[0]
                points2D.append((x, y, point3D_id))

            images[image_id] = {
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': name,
                'points2D': points2D
            }
    return images


def read_points3D_binary(path: str) -> Dict:
    """Read COLMAP points3D.bin file."""
    import struct
    points3D = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]

            track_length = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("<I", f.read(4))[0]
                point2D_idx = struct.unpack("<I", f.read(4))[0]
                track.append((image_id, point2D_idx))

            points3D[point3D_id] = {
                'xyz': np.array(xyz),
                'rgb': rgb,
                'error': error,
                'track': track
            }
    return points3D


def main():
    parser = argparse.ArgumentParser(
        description="Localize a new image using pre-built COLMAP reconstruction"
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        required=True,
        help="Path to COLMAP sparse reconstruction (containing cameras.bin, images.bin, points3D.bin)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the new image to localize"
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default="PINHOLE",
        choices=["SIMPLE_PINHOLE", "PINHOLE", "OPENCV"],
        help="Camera model"
    )
    parser.add_argument(
        "--camera_params",
        type=str,
        default=None,
        help="Camera intrinsic parameters (comma-separated): fx,fy,cx,cy for PINHOLE, f,cx,cy for SIMPLE_PINHOLE"
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=8.0,
        help="RANSAC reprojection error threshold in pixels"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Explicit path to database.db (optional, will auto-detect if not specified)"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=50000,
        help="Maximum number of 3D points to load (default: 50000, reduces memory usage)"
    )

    args = parser.parse_args()

    # Parse camera params
    camera_params = None
    if args.camera_params:
        camera_params = [float(x) for x in args.camera_params.split(",")]

    print(f"Loading COLMAP reconstruction from {args.colmap_dir}...")
    db = COLMAPDatabase(
        args.colmap_dir,
        database_path=args.database_path,
        max_points=args.max_points
    )

    # Print stored camera intrinsics
    print(f"\n=== Stored Camera Intrinsics ===")
    for cam_id, cam in db.cameras.items():
        if HAS_PYCOLMAP:
            model_name = cam.model.name
            width, height = cam.width, cam.height
            params = cam.params
        else:
            model_id_to_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                               3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                               6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
                               9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}
            model_name = model_id_to_name.get(cam['model_id'], f"UNKNOWN({cam['model_id']})")
            width, height = cam['width'], cam['height']
            params = cam['params']

        print(f"  Camera {cam_id}:")
        print(f"    Model: {model_name}")
        print(f"    Size: {width} x {height}")

        if model_name == "SIMPLE_PINHOLE":
            print(f"    f={params[0]:.2f}, cx={params[1]:.2f}, cy={params[2]:.2f}")
            print(f"    --camera_params {params[0]:.4f},{params[1]:.4f},{params[2]:.4f}")
        elif model_name == "PINHOLE":
            print(f"    fx={params[0]:.2f}, fy={params[1]:.2f}, cx={params[2]:.2f}, cy={params[3]:.2f}")
            print(f"    --camera_params {params[0]:.4f},{params[1]:.4f},{params[2]:.4f},{params[3]:.4f}")
        elif model_name in ["OPENCV", "SIMPLE_RADIAL", "RADIAL"]:
            print(f"    fx={params[0]:.2f}, fy={params[1]:.2f}, cx={params[2]:.2f}, cy={params[3]:.2f}")
            if len(params) > 4:
                print(f"    distortion: {params[4:]}")
            print(f"    --camera_params {','.join(f'{p:.4f}' for p in params)}")
        else:
            print(f"    params: {params}")
    print()

    print(f"Initializing localizer...")
    localizer = ImageLocalizer(
        db,
        camera_model=args.camera_model,
        camera_params=camera_params,
    )

    print(f"Loading image {args.image_path}...")
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image {args.image_path}")
        return

    print(f"Localizing...")
    pose = localizer.localize(image, ransac_threshold=args.ransac_threshold)

    if pose.success:
        print(f"\n✓ Localization successful!")
        print(f"  Time: {pose.estimation_time_ms:.1f} ms")
        print(f"  Inliers: {pose.num_inliers} ({pose.inlier_ratio*100:.1f}%)")
        print(f"  Camera center (world): [{pose.t[0]:.4f}, {pose.t[1]:.4f}, {pose.t[2]:.4f}]")
        print(f"  Quaternion (w,x,y,z): [{pose.qvec[0]:.4f}, {pose.qvec[1]:.4f}, {pose.qvec[2]:.4f}, {pose.qvec[3]:.4f}]")
        print(f"\nRotation matrix (camera-to-world):")
        print(pose.R)
    else:
        print(f"\n✗ Localization failed")
        print(f"  Time: {pose.estimation_time_ms:.1f} ms")
        print(f"  Matches found: {pose.num_inliers}")


if __name__ == "__main__":
    main()
