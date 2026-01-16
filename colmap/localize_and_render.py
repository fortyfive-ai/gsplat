"""
Localize a novel view image using COLMAP and render from 3DGS for comparison.

This script:
1. Takes a novel view image as input
2. Estimates camera pose using COLMAP reconstruction (via localize_image.py)
3. Renders the scene from 3DGS using the estimated pose
4. Displays the novel image and rendered image side by side

IMPORTANT: If you trained with `--normalize_world_space True` (the default),
you must pass the same `--data_dir` used during training so that the same
normalization transform can be computed and applied to the estimated pose.

Usage:
    python -m colmap.localize_and_render \
        --data_dir /path/to/training/data_dir \
        --ckpt_path /path/to/gaussian.splat \
        --image_path /path/to/novel_image.jpg

    # Or without normalization (if trained with --normalize_world_space False):
    python -m colmap.localize_and_render \
        --colmap_dir /path/to/colmap/sparse/0 \
        --ckpt_path /path/to/gaussian.splat \
        --image_path /path/to/novel_image.jpg \
        --no_normalize
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from colmap.localize_image import COLMAPDatabase, ImageLocalizer, HAS_PYCOLMAP


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    Get a similarity transform to normalize dataset from c2w (OpenCV convention) cameras.
    Copied from datasets/normalize.py for standalone usage.
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def align_principal_axes(point_cloud):
    """Align point cloud principal axes. Copied from datasets/normalize.py."""
    centroid = np.median(point_cloud, axis=0)
    translated_point_cloud = point_cloud - centroid
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation_matrix = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def transform_points(matrix, points):
    """Transform points using an SE(3) matrix."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_camera(matrix, c2w):
    """Transform a single camera-to-world matrix using an SE(3) matrix."""
    # Apply transform: new_c2w = matrix @ c2w
    new_c2w = matrix @ c2w
    # Normalize rotation (remove any scaling)
    scaling = np.linalg.norm(new_c2w[:3, 0])
    new_c2w[:3, :3] = new_c2w[:3, :3] / scaling
    return new_c2w


def compute_normalization_transform(colmap_dir: str, max_points: int = 100000) -> np.ndarray:
    """
    Compute the normalization transform from COLMAP reconstruction.
    This replicates the logic in datasets/colmap.py Parser.

    Args:
        colmap_dir: Path to COLMAP sparse reconstruction

    Returns:
        4x4 transformation matrix
    """
    import pycolmap

    # Load reconstruction using pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction(str(colmap_dir))

    # Extract camera-to-world matrices
    images = reconstruction.images
    c2w_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    for img_id, img in images.items():
        # Get rotation matrix and translation from image
        # pycolmap Image has .cam_from_world() method returning Rigid3d
        rigid = img.cam_from_world()
        rot = rigid.rotation.matrix()  # world-to-camera rotation matrix
        trans = rigid.translation.reshape(3, 1)  # world-to-camera translation
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        c2w = np.linalg.inv(w2c)
        c2w_mats.append(c2w)

    c2w_mats = np.stack(c2w_mats, axis=0)

    # Get 3D points
    points3D = reconstruction.points3D
    points_list = []
    for pt_id, pt in points3D.items():
        points_list.append(pt.xyz)
    points = np.array(points_list, dtype=np.float32)

    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    # Compute transforms (same as datasets/colmap.py)
    T1 = similarity_from_cameras(c2w_mats)

    # Transform cameras and points with T1
    c2w_transformed = []
    for c2w in c2w_mats:
        c2w_transformed.append(transform_camera(T1, c2w))
    c2w_transformed = np.stack(c2w_transformed, axis=0)

    points_transformed = transform_points(T1, points)

    T2 = align_principal_axes(points_transformed)

    # Apply T2
    c2w_final = []
    for c2w in c2w_transformed:
        c2w_final.append(transform_camera(T2, c2w))
    c2w_final = np.stack(c2w_final, axis=0)

    points_final = transform_points(T2, points_transformed)

    transform = T2 @ T1

    # Check for up-side-down fix
    if np.median(points_final[:, 2]) > np.mean(points_final[:, 2]):
        T3 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        transform = T3 @ transform

    return transform.astype(np.float32)


def load_splat_file(splat_path: str) -> dict:
    """Load a .splat file and return gaussian properties as numpy arrays.

    .splat format (32 bytes per gaussian):
    - Position: 3x Float32 (12 bytes)
    - Scale: 3x Float32 (12 bytes)
    - Color: 4x uint8 RGBA (4 bytes)
    - Rotation: 4x uint8 quaternion (4 bytes)

    Args:
        splat_path: Path to .splat file

    Returns:
        Dictionary with numpy arrays for means, scales, colors, opacities, quats
    """
    data = np.fromfile(splat_path, dtype=np.uint8)
    n_gaussians = len(data) // 32
    print(f"  Loading {n_gaussians:,} gaussians from .splat file")

    # Reshape to per-gaussian
    data = data.reshape(n_gaussians, 32)

    # Position (bytes 0-11): 3x float32
    means = data[:, 0:12].view(np.float32).reshape(n_gaussians, 3)

    # Scale (bytes 12-23): 3x float32
    scales = data[:, 12:24].view(np.float32).reshape(n_gaussians, 3)

    # Color RGBA (bytes 24-27): 4x uint8
    rgba = data[:, 24:28].astype(np.float32)
    colors = rgba[:, :3] / 255.0
    opacities = rgba[:, 3] / 255.0

    # Rotation (bytes 28-31): 4x uint8
    quats_raw = data[:, 28:32].astype(np.float32)
    quats = (quats_raw / 128.0) - 1.0

    # Normalize quaternions
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(quat_norms, 1e-8)

    return {
        'means': means.copy(),
        'scales': scales.copy(),
        'colors': colors.copy(),
        'opacities': opacities.copy(),
        'quats': quats.copy(),
        'n_gaussians': n_gaussians
    }


def load_gaussian_model(ckpt_path: str, device: torch.device):
    """Load pretrained Gaussian Splatting model from checkpoint.

    Supports both .pt (PyTorch checkpoint) and .splat file formats.

    Args:
        ckpt_path: Path to the .pt or .splat file
        device: torch device to load model to

    Returns:
        Dictionary containing gaussian parameters ready for rendering
    """
    ckpt_path = Path(ckpt_path)

    # Check file extension to determine format
    if ckpt_path.suffix.lower() == '.splat':
        # Load .splat file format
        data = load_splat_file(str(ckpt_path))

        means = torch.from_numpy(data['means']).float().to(device)
        scales = torch.from_numpy(data['scales']).float().to(device)
        quats = torch.from_numpy(data['quats']).float().to(device)
        opacities = torch.from_numpy(data['opacities']).float().to(device)
        colors = torch.from_numpy(data['colors']).float().to(device)  # [N, 3] direct RGB

        return {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
            "sh_degree": None,  # No SH for .splat files, use direct RGB
        }

    # Load .pt checkpoint format
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "splats" in ckpt:
        splats = ckpt["splats"]
    else:
        # Assume the checkpoint is the splats dict directly
        splats = ckpt

    # Extract and prepare gaussian parameters
    means = splats["means"]  # [N, 3]
    quats = F.normalize(splats["quats"], p=2, dim=-1)  # [N, 4] normalized
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"].squeeze(-1))  # [N]

    # Handle different opacity shapes
    if opacities.dim() > 1:
        opacities = opacities.squeeze()

    # Spherical harmonics coefficients
    sh0 = splats["sh0"]  # [N, 1, 3]
    shN = splats.get("shN", None)  # [N, K, 3] or None

    if shN is not None:
        colors = torch.cat([sh0, shN], dim=-2)  # [N, (degree+1)^2, 3]
    else:
        colors = sh0

    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
    }


def render_gaussian(
    gaussians: dict,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    near_plane: float = 0.01,
    far_plane: float = 1000.0,
):
    """Render image from Gaussian Splatting model.

    Args:
        gaussians: Dictionary with gaussian parameters
        viewmat: 4x4 world-to-camera transformation matrix
        K: 3x3 camera intrinsic matrix
        width: Output image width
        height: Output image height
        device: torch device
        near_plane: Near clipping plane
        far_plane: Far clipping plane

    Returns:
        Rendered RGB image as numpy array [H, W, 3] in range [0, 255]
    """
    from gsplat.rendering import rasterization

    # Add batch dimension
    viewmats = viewmat.unsqueeze(0)  # [1, 4, 4]
    Ks = K.unsqueeze(0)  # [1, 3, 3]

    with torch.no_grad():
        render_colors, render_alphas, meta = rasterization(
            means=gaussians["means"],
            quats=gaussians["quats"],
            scales=gaussians["scales"],
            opacities=gaussians["opacities"],
            colors=gaussians["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=gaussians["sh_degree"],
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB",
            packed=False,
        )

    # Extract RGB [H, W, 3]
    rgb = render_colors[0, ..., 0:3].clamp(0, 1)

    # Convert to numpy uint8
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

    return rgb_np


def pose_to_viewmat(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world pose to world-to-camera view matrix.

    Args:
        R_c2w: 3x3 camera-to-world rotation matrix
        t_c2w: 3x1 camera center in world coordinates

    Returns:
        4x4 world-to-camera transformation matrix
    """
    # World-to-camera rotation is transpose of camera-to-world
    R_w2c = R_c2w.T

    # World-to-camera translation
    t_w2c = -R_w2c @ t_c2w

    # Build 4x4 matrix
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R_w2c
    viewmat[:3, 3] = t_w2c

    return viewmat


def create_side_by_side(img1: np.ndarray, img2: np.ndarray, labels: tuple = ("Novel View", "3DGS Render")) -> np.ndarray:
    """Create side-by-side comparison image.

    Args:
        img1: First image (BGR or RGB)
        img2: Second image (BGR or RGB)
        labels: Tuple of labels for each image

    Returns:
        Combined image with labels
    """
    # Ensure same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    max_h = max(h1, h2)

    # Resize if needed to match heights
    if h1 != max_h:
        scale = max_h / h1
        img1 = cv2.resize(img1, (int(w1 * scale), max_h))
    if h2 != max_h:
        scale = max_h / h2
        img2 = cv2.resize(img2, (int(w2 * scale), max_h))

    # Add padding between images
    padding = 10
    separator = np.ones((max_h, padding, 3), dtype=np.uint8) * 128

    # Concatenate horizontally
    combined = np.concatenate([img1, separator, img2], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Label positions
    label1_pos = (10, 30)
    label2_pos = (img1.shape[1] + padding + 10, 30)

    # Draw background rectangles for labels
    for label, pos in [(labels[0], label1_pos), (labels[1], label2_pos)]:
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(combined, (pos[0] - 5, pos[1] - text_h - 5),
                     (pos[0] + text_w + 5, pos[1] + 5), bg_color, -1)
        cv2.putText(combined, label, pos, font, font_scale, color, thickness)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Localize novel view and render from 3DGS for comparison"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to training data directory (same as used for training). "
             "Required if trained with normalize_world_space=True (default). "
             "The colmap_dir will be auto-detected as data_dir/sparse/0."
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        default=None,
        help="Path to COLMAP sparse reconstruction. Auto-detected from data_dir if not provided."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to 3DGS checkpoint (.pt or .splat file)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the novel view image to localize"
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
        help="Camera intrinsic parameters (comma-separated): fx,fy,cx,cy for PINHOLE"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Explicit path to database.db (optional)"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=50000,
        help="Maximum number of 3D points to load for localization"
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=8.0,
        help="RANSAC reprojection error threshold in pixels"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the comparison image"
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Don't display the window (useful for headless environments)"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip world space normalization (use if trained with --normalize_world_space False)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.data_dir is None and args.colmap_dir is None:
        parser.error("Either --data_dir or --colmap_dir must be provided")

    # Auto-detect colmap_dir from data_dir
    if args.colmap_dir is None:
        colmap_dir = os.path.join(args.data_dir, "sparse/0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(args.data_dir, "sparse")
        if not os.path.exists(colmap_dir):
            parser.error(f"Could not find COLMAP sparse reconstruction in {args.data_dir}")
        args.colmap_dir = colmap_dir

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse camera params
    camera_params = None
    if args.camera_params:
        camera_params = [float(x) for x in args.camera_params.split(",")]

    # Load COLMAP database
    print(f"\n=== Loading COLMAP reconstruction from {args.colmap_dir} ===")
    colmap_db = COLMAPDatabase(
        args.colmap_dir,
        database_path=args.database_path,
        max_points=args.max_points,
    )

    # Get camera intrinsics from COLMAP if not provided
    if camera_params is None:
        print("\nNo camera params provided, using COLMAP intrinsics...")
        # Use first camera's intrinsics
        cam_id = list(colmap_db.cameras.keys())[0]
        cam = colmap_db.cameras[cam_id]

        if HAS_PYCOLMAP:
            params = cam.params
            model_name = cam.model.name
        else:
            params = cam['params']
            model_id_to_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                               3: "RADIAL", 4: "OPENCV"}
            model_name = model_id_to_name.get(cam['model_id'], "PINHOLE")

        if model_name == "SIMPLE_PINHOLE":
            camera_params = [params[0], params[0], params[1], params[2]]  # f, f, cx, cy
            args.camera_model = "SIMPLE_PINHOLE"
        elif model_name == "SIMPLE_RADIAL":
            # SIMPLE_RADIAL: [f, cx, cy, k1]
            camera_params = [params[0], params[0], params[1], params[2]]  # f, f, cx, cy
            args.camera_model = "PINHOLE"  # Treat as pinhole for PnP (ignore distortion)
        elif model_name == "RADIAL":
            # RADIAL: [f, cx, cy, k1, k2]
            camera_params = [params[0], params[0], params[1], params[2]]  # f, f, cx, cy
            args.camera_model = "PINHOLE"
        else:
            camera_params = list(params[:4])  # fx, fy, cx, cy

        print(f"  Camera model: {model_name}")
        print(f"  Using camera params: fx={camera_params[0]:.2f}, fy={camera_params[1]:.2f}, cx={camera_params[2]:.2f}, cy={camera_params[3]:.2f}")

    # Initialize localizer
    print(f"\n=== Initializing localizer ===")
    localizer = ImageLocalizer(
        colmap_db,
        camera_model=args.camera_model,
        camera_params=camera_params,
    )

    # Load novel view image
    print(f"\n=== Loading novel view image: {args.image_path} ===")
    novel_image = cv2.imread(args.image_path)
    if novel_image is None:
        print(f"Error: Could not load image {args.image_path}")
        return

    height, width = novel_image.shape[:2]
    print(f"  Image size: {width} x {height}")

    # Localize the image
    print(f"\n=== Localizing image ===")
    pose = localizer.localize(novel_image, ransac_threshold=args.ransac_threshold)

    if not pose.success:
        print(f"Localization failed!")
        print(f"  Matches found: {pose.num_inliers}")
        print(f"  Time: {pose.estimation_time_ms:.1f} ms")
        return

    print(f"Localization successful!")
    print(f"  Time: {pose.estimation_time_ms:.1f} ms")
    print(f"  Inliers: {pose.num_inliers} ({pose.inlier_ratio*100:.1f}%)")
    print(f"  Camera center (COLMAP): [{pose.t[0]:.4f}, {pose.t[1]:.4f}, {pose.t[2]:.4f}]")
    print(f"  Quaternion (w,x,y,z): [{pose.qvec[0]:.4f}, {pose.qvec[1]:.4f}, {pose.qvec[2]:.4f}, {pose.qvec[3]:.4f}]")

    # Compute normalization transform if needed
    norm_transform = None
    if not args.no_normalize:
        print(f"\n=== Computing normalization transform ===")
        print(f"  (This matches the transform applied during training)")
        try:
            norm_transform = compute_normalization_transform(args.colmap_dir)
            print(f"  Transform computed successfully")
        except Exception as e:
            print(f"  Warning: Could not compute normalization transform: {e}")
            print(f"  Proceeding without normalization (results may be incorrect)")

    # Load 3DGS model
    print(f"\n=== Loading 3DGS model from {args.ckpt_path} ===")
    gaussians = load_gaussian_model(args.ckpt_path, device)
    num_gaussians = gaussians["means"].shape[0]
    print(f"  Loaded {num_gaussians:,} gaussians")
    print(f"  SH degree: {gaussians['sh_degree']}")

    # Build camera-to-world matrix from estimated pose
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = pose.R
    c2w[:3, 3] = pose.t

    # Apply normalization transform if computed
    if norm_transform is not None:
        c2w = transform_camera(norm_transform, c2w)
        print(f"  Applied normalization transform to camera pose")
        print(f"  Camera center (normalized): [{c2w[0, 3]:.4f}, {c2w[1, 3]:.4f}, {c2w[2, 3]:.4f}]")

    # Convert camera-to-world to world-to-camera (view matrix)
    viewmat = np.linalg.inv(c2w).astype(np.float32)
    viewmat_torch = torch.from_numpy(viewmat).float().to(device)

    # Build camera intrinsic matrix
    fx, fy, cx, cy = camera_params[:4]
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    K_torch = torch.from_numpy(K).float().to(device)

    # Render from 3DGS
    print(f"\n=== Rendering from 3DGS ===")
    rendered_image = render_gaussian(
        gaussians,
        viewmat_torch,
        K_torch,
        width,
        height,
        device,
    )
    print(f"  Rendered image size: {rendered_image.shape}")

    # Convert rendered image from RGB to BGR for OpenCV
    rendered_bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)

    # Create side-by-side comparison
    comparison = create_side_by_side(novel_image, rendered_bgr)

    # Save if requested
    if args.output_path:
        cv2.imwrite(args.output_path, comparison)
        print(f"\n=== Saved comparison to {args.output_path} ===")

    # Display
    if not args.no_display:
        print(f"\n=== Displaying comparison (press 'q' or ESC to exit) ===")
        window_name = "Novel View vs 3DGS Render"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Resize window to fit screen if image is too large
        screen_width = 1920  # Adjust as needed
        if comparison.shape[1] > screen_width:
            scale = screen_width / comparison.shape[1]
            new_h = int(comparison.shape[0] * scale)
            cv2.resizeWindow(window_name, screen_width, new_h)

        cv2.imshow(window_name, comparison)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    main()
