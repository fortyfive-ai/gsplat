#!/usr/bin/env python3
"""
Generate dense point clouds from COLMAP sparse reconstruction and metric depth maps.

This script solves the scale ambiguity between SfM reconstruction and metric depth:
1. Projects sparse 3D points from COLMAP to camera frames to get sparse depth
2. Compares sparse depth with metric depth from sensors to estimate scale
3. Unprojects dense metric depth maps to world coordinates using estimated scale

Usage:
  python dense_from_depth.py --colmap-dir /path/to/colmap/output --output dense_points.ply
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import struct
from params_proto import proto

# COLMAP binary reading utilities
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {
                "id": camera_id,
                "model": CAMERA_MODEL_NAMES[model_id],
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path_to_model_file):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            # Read image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                  tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def read_points3D_binary(path_to_model_file):
    """Read COLMAP points3D.bin file."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]

            # Read track
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))

            points3D[point3D_id] = {
                "id": point3D_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids": image_ids,
                "point2D_idxs": point2D_idxs,
            }
    return points3D


# Camera model constants
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

CAMERA_MODEL_NUM_PARAMS = {
    0: 3,  # SIMPLE_PINHOLE: f, cx, cy
    1: 4,  # PINHOLE: fx, fy, cx, cy
    2: 4,  # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,  # RADIAL: f, cx, cy, k1, k2
    4: 8,  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,  # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
    6: 12, # FULL_OPENCV
    7: 5,  # FOV
    8: 5,  # SIMPLE_RADIAL_FISHEYE
    9: 6,  # RADIAL_FISHEYE
    10: 12 # THIN_PRISM_FISHEYE
}


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def get_camera_matrix(camera):
    """Get camera intrinsic matrix from COLMAP camera parameters."""
    params = camera["params"]
    model = camera["model"]

    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
    elif model in ["SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in ["OPENCV", "OPENCV_FISHEYE"]:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def project_points_to_image(points3D_world, R, t, K):
    """Project 3D points in world coordinates to image coordinates.

    Args:
        points3D_world: Nx3 array of 3D points in world coordinates
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
        K: 3x3 camera intrinsic matrix

    Returns:
        points2D: Nx2 array of 2D image coordinates
        depths: N array of depths in camera frame
    """
    # Transform to camera coordinates
    points3D_cam = (R @ points3D_world.T).T + t
    depths = points3D_cam[:, 2]

    # Project to image
    points2D_hom = (K @ points3D_cam.T).T
    points2D = points2D_hom[:, :2] / points2D_hom[:, 2:3]

    return points2D, depths


def read_depth_image(depth_path: Path, depth_scale: float = 1000.0) -> np.ndarray:
    """Read depth image and convert to meters.

    Args:
        depth_path: Path to depth image (16-bit PNG)
        depth_scale: Scale factor to convert to meters (default 1000 for mm to m)

    Returns:
        Depth map in meters
    """
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Could not read depth image: {depth_path}")
    return depth.astype(np.float32) / depth_scale


def estimate_scale_for_image(
    sparse_points3D: np.ndarray,
    sparse_depths: np.ndarray,
    metric_depth: np.ndarray,
    points2D: np.ndarray,
    image_shape: Tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float = 10.0
) -> Tuple[float, int]:
    """Estimate scale factor between SfM depth and metric depth for one image.

    Args:
        sparse_points3D: Nx3 array of sparse 3D points in world coords
        sparse_depths: N array of SfM depths (in camera frame)
        metric_depth: HxW metric depth map
        points2D: Nx2 array of 2D projections
        image_shape: (height, width)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        scale: Estimated scale factor (sfm_depth / metric_depth)
        num_correspondences: Number of valid correspondences used
    """
    height, width = image_shape

    # Filter points within image bounds
    valid_mask = (
        (points2D[:, 0] >= 0) & (points2D[:, 0] < width) &
        (points2D[:, 1] >= 0) & (points2D[:, 1] < height) &
        (sparse_depths > 0)
    )

    valid_points2D = points2D[valid_mask]
    valid_sfm_depths = sparse_depths[valid_mask]

    if len(valid_points2D) == 0:
        return None, 0

    # Sample metric depth at sparse point locations
    metric_depths_at_points = []
    sfm_depths_filtered = []

    for i, pt in enumerate(valid_points2D):
        x, y = int(pt[0]), int(pt[1])
        metric_d = metric_depth[y, x]

        # Filter by valid depth range
        if min_depth < metric_d < max_depth:
            metric_depths_at_points.append(metric_d)
            sfm_depths_filtered.append(valid_sfm_depths[i])

    if len(metric_depths_at_points) < 10:  # Need minimum correspondences
        return None, 0

    metric_depths_at_points = np.array(metric_depths_at_points)
    sfm_depths_filtered = np.array(sfm_depths_filtered)

    # Compute scale: scale = sfm_depth / metric_depth
    scales = sfm_depths_filtered / metric_depths_at_points

    # Use median for robustness
    scale = np.median(scales)

    return scale, len(metric_depths_at_points)


def filter_depth_edges(depth_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Filter depth edges where there are large discontinuities (common sensor artifacts).

    Args:
        depth_map: HxW depth map in meters
        threshold: Maximum allowed depth difference with neighbors (in meters)

    Returns:
        HxW boolean mask where True indicates valid depth
    """
    # Compute depth gradients
    dy, dx = np.gradient(depth_map)
    gradient_mag = np.sqrt(dx**2 + dy**2)

    # Mark pixels with large gradients as invalid (likely edge contamination)
    valid = gradient_mag < threshold

    # Also mark zero depth as invalid
    valid = valid & (depth_map > 0)

    return valid


def unproject_depth_to_points(
    depth_map: np.ndarray,
    color_image: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    filter_edges: bool = False,
    edge_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject depth map to 3D points in world coordinates with RGB colors.

    Args:
        depth_map: HxW depth map in meters
        color_image: HxWx3 RGB color image
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
        scale: Scale factor to apply (metric to SfM scale)
        mask: Optional HxW boolean mask for valid pixels
        min_depth: Minimum valid depth in meters (filter contaminated near values)
        max_depth: Maximum valid depth in meters (filter contaminated far values)
        filter_edges: Whether to filter depth discontinuities (edge artifacts)
        edge_threshold: Threshold for depth gradient filtering (meters)

    Returns:
        Tuple of (Nx3 array of 3D points in world coordinates, Nx3 array of RGB colors)
    """
    height, width = depth_map.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Filter valid depth values:
    # 1. Depth > 0 (not empty)
    # 2. Depth is not NaN or Inf (contaminated)
    # 3. Depth within valid range [min_depth, max_depth]
    valid_depth = (
        (depth_map > 0) &
        np.isfinite(depth_map) &
        (depth_map >= min_depth) &
        (depth_map <= max_depth)
    )

    # 4. Filter edge artifacts if requested
    if filter_edges:
        edge_valid = filter_depth_edges(depth_map, edge_threshold)
        valid_depth = valid_depth & edge_valid

    # Apply user mask if provided
    if mask is not None:
        valid = mask & valid_depth
    else:
        valid = valid_depth

    u = u[valid]
    v = v[valid]
    depth = depth_map[valid]

    # Extract colors from the same pixel locations
    colors = color_image[valid]  # Nx3 RGB values

    # Scale depth to SfM scale
    depth_scaled = depth * scale

    # Unproject to camera coordinates
    K_inv = np.linalg.inv(K)
    pixels_hom = np.stack([u, v, np.ones_like(u)], axis=0)  # 3xN
    points_cam = K_inv @ pixels_hom  # 3xN
    points_cam = points_cam * depth_scaled  # Scale by depth

    # Transform to world coordinates
    # P_world = R^T @ (P_cam - t)
    R_inv = R.T
    points_world = (R_inv @ (points_cam - t.reshape(3, 1))).T  # Nx3

    return points_world, colors


def save_ply(points: np.ndarray, output_path: Path, colors: Optional[np.ndarray] = None):
    """Save points to PLY file in binary format.

    Args:
        points: Nx3 array of 3D points
        output_path: Output PLY file path
        colors: Optional Nx3 array of RGB colors (0-255)
    """
    # Write header in text mode
    with open(output_path, 'wb') as f:
        # Write PLY header
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {len(points)}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        if colors is not None:
            header += "property uchar red\n"
            header += "property uchar green\n"
            header += "property uchar blue\n"
        header += "end_header\n"

        f.write(header.encode('utf-8'))

        # Write binary data
        if colors is not None:
            # Create structured array with xyz (float32) and rgb (uint8)
            dtype = np.dtype([
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            vertex_data = np.zeros(len(points), dtype=dtype)
            vertex_data['x'] = points[:, 0]
            vertex_data['y'] = points[:, 1]
            vertex_data['z'] = points[:, 2]
            vertex_data['red'] = colors[:, 0].astype(np.uint8)
            vertex_data['green'] = colors[:, 1].astype(np.uint8)
            vertex_data['blue'] = colors[:, 2].astype(np.uint8)
        else:
            # Only xyz
            dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            vertex_data = np.zeros(len(points), dtype=dtype)
            vertex_data['x'] = points[:, 0]
            vertex_data['y'] = points[:, 1]
            vertex_data['z'] = points[:, 2]

        vertex_data.tofile(f)


@proto.cli
def main(
    colmap_dir: str = None,  # Path to COLMAP output directory
    output: str = None,  # Output PLY file path
    depths_subdir: str = "depths",  # Subdirectory name for depth maps (e.g., "depths", "depths_anything")
    depth_scale: float = 1000.0,  # Depth scale factor (default: 1000 for mm to m)
    min_depth: float = 0.1,  # Minimum valid depth in meters
    max_depth: float = 10.0,  # Maximum valid depth in meters
    subsample: int = 1,  # Subsample depth maps by this factor (1=no subsampling)
    filter_edges: bool = False,  # Filter depth edge artifacts
    edge_threshold: float = 0.5,  # Depth gradient threshold for edge filtering in meters
):
    """Generate dense point clouds from COLMAP and depth maps"""
    if colmap_dir is None or output is None:
        print("Error: colmap_dir and output are required arguments")
        print("Usage: python dense_from_depth.py --colmap-dir /path/to/colmap/output --output dense_points.ply")
        return 1

    colmap_dir = Path(colmap_dir)
    sparse_dir = colmap_dir / "sparse" / "0"
    depths_dir = colmap_dir / depths_subdir
    images_dir = colmap_dir / "images"
    masks_dir = colmap_dir / "masks"

    # Verify directories exist
    if not sparse_dir.exists():
        print(f"Error: Sparse directory not found: {sparse_dir}")
        return 1
    if not depths_dir.exists():
        print(f"Error: Depths directory not found: {depths_dir}")
        return 1

    print("Loading COLMAP reconstruction...")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    points3D = read_points3D_binary(sparse_dir / "points3D.bin")

    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Convert points3D to numpy array
    point_ids = list(points3D.keys())
    points3D_xyz = np.array([points3D[pid]["xyz"] for pid in point_ids])

    # Build a mapping from original image filenames (basename) to symlink names
    # This handles cases where COLMAP stores paths that resolve differently than symlink targets
    print("Building image path mapping...")
    basename_to_symlink = {}
    for img_file in images_dir.iterdir():
        if img_file.is_symlink() or img_file.is_file():
            # Map the basename to the symlink/file name
            # This works even if paths don't match exactly
            target = img_file.resolve()
            basename = target.name
            basename_to_symlink[basename] = img_file.name

    # Estimate scale for each image
    print("\nEstimating scale factors...")
    scales = {}
    scale_stats = []

    for img_id, img_data in tqdm(images.items()):
        img_name = img_data["name"]

        # Extract basename from COLMAP's stored path
        colmap_basename = Path(img_name).name

        # Look up the actual filename in our images directory
        if colmap_basename in basename_to_symlink:
            # Use the symlink filename for depth lookup
            actual_img_name = basename_to_symlink[colmap_basename]
        else:
            # Fall back to using the basename directly
            actual_img_name = colmap_basename

        depth_name = actual_img_name.rsplit('.', 1)[0] + '.png'
        depth_path = depths_dir / depth_name

        if not depth_path.exists():
            print(f"Warning: Depth map not found for {img_name}")
            continue

        # Read depth map
        metric_depth = read_depth_image(depth_path, depth_scale)

        # Get camera parameters
        camera = cameras[img_data["camera_id"]]
        K = get_camera_matrix(camera)
        R = qvec2rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])

        # Get sparse points observed by this image
        point3D_ids = img_data["point3D_ids"]
        valid_mask = point3D_ids != -1

        if not valid_mask.any():
            continue

        valid_point_ids = point3D_ids[valid_mask]
        sparse_pts = np.array([points3D[pid]["xyz"] for pid in valid_point_ids])

        # Project to image
        points2D, depths = project_points_to_image(sparse_pts, R, t, K)

        # Estimate scale
        scale, n_corr = estimate_scale_for_image(
            sparse_pts, depths, metric_depth, points2D,
            (camera["height"], camera["width"]),
            min_depth, max_depth
        )

        if scale is not None:
            scales[img_name] = scale
            scale_stats.append(scale)
            print(f"  {img_name}: scale={scale:.4f} ({n_corr} correspondences)")

    if not scales:
        print("Error: Could not estimate scale for any image")
        return 1

    # Compute global scale as median
    global_scale = np.median(scale_stats)
    scale_std = np.std(scale_stats)
    print(f"\nGlobal scale: {global_scale:.4f} ± {scale_std:.4f}")
    print(f"Scale range: [{np.min(scale_stats):.4f}, {np.max(scale_stats):.4f}]")

    # Generate dense point cloud
    print("\nGenerating dense point cloud...")
    all_points = []
    all_colors = []

    for img_id, img_data in tqdm(images.items()):
        img_name = img_data["name"]

        # Extract basename from COLMAP's stored path
        colmap_basename = Path(img_name).name

        # Look up the actual filename in our images directory
        if colmap_basename in basename_to_symlink:
            # Use the symlink filename for depth lookup
            actual_img_name = basename_to_symlink[colmap_basename]
            img_path = images_dir / actual_img_name
        else:
            # Fall back to using the basename directly
            actual_img_name = colmap_basename
            img_path = images_dir / img_name

        depth_name = actual_img_name.rsplit('.', 1)[0] + '.png'
        depth_path = depths_dir / depth_name

        if not depth_path.exists():
            continue

        # Read color image
        color_image = cv2.imread(str(img_path))
        if color_image is None:
            print(f"Warning: Could not read color image {img_path}, skipping")
            continue

        # Convert BGR to RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Read depth map
        metric_depth = read_depth_image(depth_path, depth_scale)

        # Ensure color and depth have same dimensions
        if color_image.shape[:2] != metric_depth.shape:
            print(f"Warning: Color {color_image.shape[:2]} and depth {metric_depth.shape} dimensions mismatch for {img_name}, resizing color")
            color_image = cv2.resize(color_image, (metric_depth.shape[1], metric_depth.shape[0]))

        # Read mask if available
        mask = None
        mask_path = masks_dir / depth_name if masks_dir.exists() else None
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0

        # Subsample if requested
        if subsample > 1:
            metric_depth = metric_depth[::subsample, ::subsample]
            color_image = color_image[::subsample, ::subsample]
            if mask is not None:
                mask = mask[::subsample, ::subsample]

        # Get camera parameters
        camera = cameras[img_data["camera_id"]]
        K = get_camera_matrix(camera)

        # Adjust K for subsampling
        if subsample > 1:
            K = K.copy()
            K[0, 0] /= subsample  # fx
            K[1, 1] /= subsample  # fy
            K[0, 2] /= subsample  # cx
            K[1, 2] /= subsample  # cy

        R = qvec2rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])

        # Use per-image scale if available, otherwise global scale
        scale = scales.get(img_name, global_scale)

        # Unproject to world coordinates with colors
        points_world, colors = unproject_depth_to_points(
            metric_depth, color_image, K, R, t, scale, mask,
            min_depth, max_depth, filter_edges, edge_threshold
        )
        all_points.append(points_world)
        all_colors.append(colors)

    # Concatenate all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    print(f"\nGenerated {len(all_points)} dense points")

    # Save to PLY
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(all_points, output_path, all_colors)
    print(f"Saved dense point cloud to: {output_path}")

    return 0


if __name__ == "__main__":
    main()