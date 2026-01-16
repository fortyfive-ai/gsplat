#!/usr/bin/env python3
"""
Growing Gaussian Demo

This demo loads a pretrained .splat file, renders from camera poses to identify
visible gaussians, and incrementally adds them to a vuer visualization.

Each new camera pose adds the newly visible gaussians (not previously seen)
to the visualization, creating a "growing" gaussian effect.

Usage:
    python examples/growing_gaussian_demo.py --splat-path /path/to/point_cloud.splat
"""

import asyncio
import struct
import tempfile
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Splat, OrbitControls, PerspectiveCamera
import tkinter as tk
from tkinter import ttk


# Spherical harmonics constant for 0th order
SH_C0 = 0.28209479177387814


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
    with open(splat_path, 'rb') as f:
        data = f.read()

    n_gaussians = len(data) // 32
    print(f"Loading {n_gaussians} gaussians from {splat_path}")

    # Pre-allocate arrays
    means = np.zeros((n_gaussians, 3), dtype=np.float32)
    scales = np.zeros((n_gaussians, 3), dtype=np.float32)
    colors = np.zeros((n_gaussians, 3), dtype=np.float32)
    opacities = np.zeros(n_gaussians, dtype=np.float32)
    quats = np.zeros((n_gaussians, 4), dtype=np.float32)

    # Parse binary data
    for i in range(n_gaussians):
        offset = i * 32
        # Position (3x float32)
        means[i] = struct.unpack('<3f', data[offset:offset+12])
        offset += 12
        # Scale (3x float32)
        scales[i] = struct.unpack('<3f', data[offset:offset+12])
        offset += 12
        # Color RGBA (4x uint8)
        rgba = struct.unpack('<4B', data[offset:offset+4])
        colors[i] = np.array(rgba[:3]) / 255.0
        opacities[i] = rgba[3] / 255.0
        offset += 4
        # Quaternion (4x uint8)
        quat_raw = struct.unpack('<4B', data[offset:offset+4])
        # Dequantize: scale from 0-255 to -1 to 1
        quats[i] = np.array([(q / 128.0) - 1.0 for q in quat_raw])

    # Normalize quaternions
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(quat_norms, 1e-8)

    return {
        'means': means,
        'scales': scales,
        'colors': colors,
        'opacities': opacities,
        'quats': quats,
        'n_gaussians': n_gaussians
    }


def load_splat_file_vectorized(splat_path: str) -> dict:
    """Vectorized version of splat loading (faster for large files).

    Args:
        splat_path: Path to .splat file

    Returns:
        Dictionary with numpy arrays for means, scales, colors, opacities, quats
    """
    data = np.fromfile(splat_path, dtype=np.uint8)
    n_gaussians = len(data) // 32
    print(f"Loading {n_gaussians} gaussians from {splat_path}")

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
        'means': means.copy(),  # Copy to ensure contiguous
        'scales': scales.copy(),
        'colors': colors.copy(),
        'opacities': opacities.copy(),
        'quats': quats.copy(),
        'n_gaussians': n_gaussians
    }


def write_splat_file(splat_path: str, means: np.ndarray, scales: np.ndarray,
                     colors: np.ndarray, opacities: np.ndarray, quats: np.ndarray):
    """Write gaussian data to .splat file format.

    Args:
        splat_path: Output path
        means: (N, 3) positions
        scales: (N, 3) scales
        colors: (N, 3) RGB colors in [0, 1]
        opacities: (N,) opacities in [0, 1]
        quats: (N, 4) quaternions
    """
    n_gaussians = len(means)

    # Pre-allocate output buffer (32 bytes per gaussian)
    buffer = np.zeros((n_gaussians, 32), dtype=np.uint8)

    # Position (bytes 0-11): 3x float32
    buffer[:, 0:12] = means.astype(np.float32).view(np.uint8).reshape(n_gaussians, 12)

    # Scale (bytes 12-23): 3x float32
    buffer[:, 12:24] = scales.astype(np.float32).view(np.uint8).reshape(n_gaussians, 12)

    # Color RGBA (bytes 24-27): 4x uint8
    rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
    opacity = np.clip(opacities * 255, 0, 255).astype(np.uint8)
    buffer[:, 24:27] = rgb
    buffer[:, 27] = opacity

    # Rotation (bytes 28-31): 4x uint8
    # Normalize quaternions first
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats_norm = quats / np.maximum(quat_norms, 1e-8)
    # Quantize: scale from [-1, 1] to [0, 255]
    quats_uint8 = np.clip((quats_norm + 1.0) * 128, 0, 255).astype(np.uint8)
    buffer[:, 28:32] = quats_uint8

    # Write to file
    buffer.tofile(splat_path)
    print(f"Wrote {n_gaussians} gaussians to {splat_path}")


def get_visible_gaussian_ids(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> torch.Tensor:
    """Render from a camera pose and return visible gaussian IDs.

    Args:
        means: (N, 3) gaussian centers
        quats: (N, 4) quaternions (wxyz)
        scales: (N, 3) scales
        opacities: (N,) opacities
        colors: (N, 3) colors
        viewmat: (4, 4) world-to-camera transform
        K: (3, 3) camera intrinsics
        width: image width
        height: image height
        near_plane: near clipping plane
        far_plane: far clipping plane

    Returns:
        Tensor of unique gaussian IDs that are visible from this camera
    """
    from gsplat.rendering import rasterization

    # Add batch dimension for single camera
    viewmats = viewmat.unsqueeze(0)  # (1, 4, 4)
    Ks = K.unsqueeze(0)  # (1, 3, 3)

    # Render with packed mode to get gaussian IDs
    with torch.no_grad():
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=True,
            render_mode="RGB",
        )

    # Get unique gaussian IDs from the packed projection
    gaussian_ids = meta['gaussian_ids']
    unique_ids = torch.unique(gaussian_ids)

    return unique_ids


def create_camera_pose(
    position: Tuple[float, float, float],
    look_at: Tuple[float, float, float] = (0, 0, 0),
    up: Tuple[float, float, float] = (0, -1, 0),
) -> np.ndarray:
    """Create a 4x4 world-to-camera view matrix (OpenCV convention).

    In OpenCV/gsplat convention:
    - Camera looks along +Z axis
    - Y axis points down
    - X axis points right

    Args:
        position: Camera position in world coordinates
        look_at: Point to look at
        up: Up vector (default is -Y for OpenCV convention)

    Returns:
        4x4 view matrix (world-to-camera)
    """
    position = np.array(position, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Z axis points from position to look_at (forward direction)
    z_axis = look_at - position
    z_axis = z_axis / np.linalg.norm(z_axis)

    # X axis is perpendicular to Z and up
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y axis is perpendicular to Z and X
    y_axis = np.cross(z_axis, x_axis)

    # Build rotation matrix (world-to-camera)
    # Each row is an axis of the camera coordinate system expressed in world coordinates
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)

    # Translation: transform position to camera space
    t = -R @ position

    # Build 4x4 view matrix
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    return viewmat


def create_intrinsics(
    width: int,
    height: int,
    fov_deg: float = 60.0,
) -> np.ndarray:
    """Create camera intrinsic matrix.

    Args:
        width: Image width
        height: Image height
        fov_deg: Horizontal field of view in degrees

    Returns:
        3x3 intrinsic matrix K
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


class GrowingGaussianVisualizer:
    """Visualizer that incrementally adds gaussians based on camera views."""

    def __init__(
        self,
        splat_path: str,
        host: str = "0.0.0.0",
        port: int = 8012,
        render_width: int = 640,
        render_height: int = 480,
        fov_deg: float = 60.0,
    ):
        """Initialize the growing gaussian visualizer.

        Args:
            splat_path: Path to the pretrained .splat file
            host: Server host
            port: Server port
            render_width: Width for rendering (to detect visible gaussians)
            render_height: Height for rendering
            fov_deg: Field of view in degrees
        """
        self.splat_path = Path(splat_path)
        self.host = host
        self.port = port
        self.render_width = render_width
        self.render_height = render_height
        self.fov_deg = fov_deg

        # Load gaussian data
        self.gaussian_data = load_splat_file_vectorized(str(self.splat_path))

        # Convert to torch tensors (for rendering)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.means_torch = torch.from_numpy(self.gaussian_data['means']).to(self.device)
        self.scales_torch = torch.from_numpy(self.gaussian_data['scales']).to(self.device)
        self.quats_torch = torch.from_numpy(self.gaussian_data['quats']).to(self.device)
        self.opacities_torch = torch.from_numpy(self.gaussian_data['opacities']).to(self.device)
        self.colors_torch = torch.from_numpy(self.gaussian_data['colors']).to(self.device)

        # Camera intrinsics
        self.K = create_intrinsics(render_width, render_height, fov_deg)
        self.K_torch = torch.from_numpy(self.K).to(self.device)

        # Track which gaussians have been added (boolean mask for speed)
        self.seen_mask = np.zeros(self.gaussian_data['n_gaussians'], dtype=bool)
        self.splat_count = 0

        # Temp directory for generated splat files
        self.temp_dir = tempfile.mkdtemp(prefix="growing_gaussian_")
        print(f"Temp directory: {self.temp_dir}")

        print(f"{'='*60}")
        print("Growing Gaussian Visualizer")
        print(f"Splat file: {self.splat_path}")
        print(f"Total gaussians: {self.gaussian_data['n_gaussians']}")
        print(f"Device: {self.device}")
        print(f"Render size: {render_width}x{render_height}")
        print(f"FOV: {fov_deg}°")

    def add_camera_pose(
        self,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float] = (0, 0, 0),
    ) -> Tuple[str, int]:
        """Process a camera pose and return path to new splat file with newly visible gaussians.

        Args:
            position: Camera position in world coordinates
            look_at: Point to look at

        Returns:
            Tuple of (path to new splat file, number of new gaussians)
        """
        # Create view matrix
        viewmat = create_camera_pose(position, look_at)
        viewmat_torch = torch.from_numpy(viewmat).to(self.device)

        # Get visible gaussian IDs
        visible_ids = get_visible_gaussian_ids(
            means=self.means_torch,
            quats=self.quats_torch,
            scales=self.scales_torch,
            opacities=self.opacities_torch,
            colors=self.colors_torch,
            viewmat=viewmat_torch,
            K=self.K_torch,
            width=self.render_width,
            height=self.render_height,
        )

        # Use numpy arrays and boolean mask for fast set operations
        visible_ids_np = visible_ids.cpu().numpy()
        n_visible = len(visible_ids_np)

        # Find new IDs using boolean mask (much faster than Python sets)
        is_new = ~self.seen_mask[visible_ids_np]
        new_ids_np = visible_ids_np[is_new]
        n_new = len(new_ids_np)

        print(f"Camera at {position}: visible={n_visible}, new={n_new}")

        if n_new == 0:
            return None, 0

        # Update seen mask
        self.seen_mask[new_ids_np] = True

        # Extract new gaussians using numpy advanced indexing
        new_means = self.gaussian_data['means'][new_ids_np]
        new_scales = self.gaussian_data['scales'][new_ids_np]
        new_colors = self.gaussian_data['colors'][new_ids_np]
        new_opacities = self.gaussian_data['opacities'][new_ids_np]
        new_quats = self.gaussian_data['quats'][new_ids_np]

        # Write to new splat file
        self.splat_count += 1
        splat_filename = f"splat_part_{self.splat_count:04d}.splat"
        splat_path = Path(self.temp_dir) / splat_filename

        write_splat_file(
            str(splat_path),
            new_means, new_scales, new_colors, new_opacities, new_quats
        )

        return str(splat_path), n_new

    def visualize(self):
        """Launch the visualization server with separate tkinter UI window."""
        app = Vuer(static_root=self.temp_dir)

        # Get scene bounds (convert to Python floats)
        means = self.gaussian_data['means']
        scene_min = [float(x) for x in means.min(axis=0)]
        scene_max = [float(x) for x in means.max(axis=0)]
        scene_center = [float(x) for x in means.mean(axis=0)]
        scene_range = float(max(scene_max[i] - scene_min[i] for i in range(3)) * 1.5)

        print(f"\n{'='*60}")
        print("Starting Visualization Server")
        print(f"URL: http://localhost:{self.port}")
        print(f"Scene center: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})")
        print(f"Scene range: {scene_range:.2f}")
        print(f"{'='*60}\n")

        # Command queue for communication between tkinter and async loop
        cmd_queue = queue.Queue()

        # Vuer session reference (set when connected)
        vuer_session = [None]

        def create_ui():
            """Create tkinter UI in separate thread."""
            root = tk.Tk()
            root.title("Growing Gaussian Controls")
            root.geometry("320x500")
            root.configure(bg='#2b2b2b')

            style = ttk.Style()
            style.theme_use('clam')
            style.configure('TLabel', background='#2b2b2b', foreground='white')
            style.configure('TButton', padding=5)
            style.configure('TScale', background='#2b2b2b')
            style.configure('TFrame', background='#2b2b2b')

            # Status frame
            status_frame = ttk.Frame(root)
            status_frame.pack(fill='x', padx=10, pady=10)

            status_var = tk.StringVar(value=f"Gaussians: 0 / {self.gaussian_data['n_gaussians']:,}")
            status_label = ttk.Label(status_frame, textvariable=status_var, font=('Arial', 10, 'bold'))
            status_label.pack()

            pct_var = tk.StringVar(value="Coverage: 0.0%")
            pct_label = ttk.Label(status_frame, textvariable=pct_var)
            pct_label.pack()

            # Orbit controls frame
            orbit_frame = ttk.LabelFrame(root, text="Orbit Camera", padding=10)
            orbit_frame.pack(fill='x', padx=10, pady=5)

            # Angle slider
            ttk.Label(orbit_frame, text="Angle (degrees):").pack(anchor='w')
            angle_var = tk.DoubleVar(value=0)
            angle_slider = ttk.Scale(orbit_frame, from_=0, to=360, variable=angle_var, orient='horizontal')
            angle_slider.pack(fill='x')
            angle_label = ttk.Label(orbit_frame, text="0°")
            angle_label.pack(anchor='e')

            def update_angle_label(*args):
                angle_label.config(text=f"{angle_var.get():.0f}°")
            angle_var.trace('w', update_angle_label)

            # Radius slider (start closer for partial coverage)
            ttk.Label(orbit_frame, text="Radius:").pack(anchor='w')
            radius_var = tk.DoubleVar(value=3.0)  # Close radius for < 50% coverage
            radius_slider = ttk.Scale(orbit_frame, from_=1, to=scene_range, variable=radius_var, orient='horizontal')
            radius_slider.pack(fill='x')

            # Height slider
            ttk.Label(orbit_frame, text="Height:").pack(anchor='w')
            height_var = tk.DoubleVar(value=2.0)  # Low height for partial coverage
            height_slider = ttk.Scale(orbit_frame, from_=scene_min[1]-5, to=scene_max[1]+10, variable=height_var, orient='horizontal')
            height_slider.pack(fill='x')

            # Add orbit camera button
            def add_orbit_camera():
                angle = angle_var.get()
                radius = radius_var.get()
                height = height_var.get()
                cmd_queue.put(('orbit', angle, radius, height))

            ttk.Button(orbit_frame, text="Add Orbit Camera", command=add_orbit_camera).pack(fill='x', pady=5)

            # Auto capture frame
            auto_frame = ttk.LabelFrame(root, text="Auto Capture", padding=10)
            auto_frame.pack(fill='x', padx=10, pady=5)

            def spiral_8():
                cmd_queue.put(('spiral', 8, radius_var.get(), height_var.get()))
            def spiral_16():
                cmd_queue.put(('spiral', 16, radius_var.get(), height_var.get()))
            def spiral_32():
                cmd_queue.put(('spiral', 32, radius_var.get(), height_var.get()))

            ttk.Button(auto_frame, text="Spiral (8 cameras)", command=spiral_8).pack(fill='x', pady=2)
            ttk.Button(auto_frame, text="Spiral (16 cameras)", command=spiral_16).pack(fill='x', pady=2)
            ttk.Button(auto_frame, text="Full Coverage (32 cameras)", command=spiral_32).pack(fill='x', pady=2)

            # Reset button
            def reset_view():
                cmd_queue.put(('reset',))
            ttk.Button(root, text="Reset (Clear All)", command=reset_view).pack(fill='x', padx=10, pady=10)

            # Update status periodically
            def update_status():
                seen = int(self.seen_mask.sum())
                total = self.gaussian_data['n_gaussians']
                pct = 100 * seen / total
                status_var.set(f"Gaussians: {seen:,} / {total:,}")
                pct_var.set(f"Coverage: {pct:.1f}% | Parts: {self.splat_count}")
                root.after(200, update_status)

            update_status()
            root.mainloop()

        # Start tkinter in separate thread
        ui_thread = threading.Thread(target=create_ui, daemon=True)
        ui_thread.start()

        @app.spawn(start=True)
        async def main(session: VuerSession):
            vuer_session[0] = session

            # Camera initial position (close view for < 50% initial coverage)
            cam_init_pos = [0.0, 2.0, 2.0]

            # Set up initial scene
            session.set @ DefaultScene(
                rawChildren=[
                    PerspectiveCamera(
                        key="main-camera",
                        makeDefault=True,
                        fov=50,
                        near=0.1,
                        far=1000,
                        position=cam_init_pos,
                    ),
                ],
                bgChildren=[
                    OrbitControls(key="orbit-controls", target=scene_center),
                ],
                show_helper=False,
                grid=True,
                up=[0, 1, 0],
                background="#1a1a1a",
            )

            # Helper to add splat
            def add_splat_from_pose(position, look_at):
                position = tuple(float(x) for x in position)
                look_at = tuple(float(x) for x in look_at)

                splat_path, n_new = self.add_camera_pose(position, look_at)
                if splat_path:
                    splat_url = f"http://localhost:{self.port}/static/{Path(splat_path).name}"
                    session.upsert @ Splat(
                        key=f"splat_{self.splat_count}",
                        src=splat_url,
                        scale=1,
                    )
                    print(f"Added {n_new} new gaussians (total: {self.seen_mask.sum():,})")
                else:
                    print("No new gaussians visible from this pose")

            # Process commands from UI
            while True:
                await asyncio.sleep(0.05)

                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    continue

                if cmd[0] == 'orbit':
                    _, angle, radius, height = cmd
                    angle_rad = np.deg2rad(angle)
                    x = scene_center[0] + radius * np.cos(angle_rad)
                    z = scene_center[2] + radius * np.sin(angle_rad)
                    position = (float(x), float(height), float(z))
                    add_splat_from_pose(position, scene_center)

                elif cmd[0] == 'spiral':
                    _, n_poses, radius, height = cmd
                    print(f"Adding {n_poses} cameras in spiral pattern...")
                    for i in range(n_poses):
                        angle_rad = 2 * np.pi * i / n_poses
                        r = radius * (0.8 + 0.4 * i / n_poses)
                        h = height + (scene_range / 4) * (i / n_poses - 0.5)
                        x = scene_center[0] + r * np.cos(angle_rad)
                        z = scene_center[2] + r * np.sin(angle_rad)
                        position = (float(x), float(h), float(z))
                        add_splat_from_pose(position, scene_center)
                        await asyncio.sleep(0.05)
                    print("Spiral complete!")

                elif cmd[0] == 'reset':
                    # Reset seen mask and remove all splats
                    self.seen_mask[:] = False
                    for i in range(1, self.splat_count + 1):
                        session.remove @ f"splat_{i}"
                    self.splat_count = 0
                    print("Reset complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Growing Gaussian Demo")
    parser.add_argument(
        "--splat-path",
        type=str,
        default="/home/yanbinghan/Downloads/point_cloud_29999.splat",
        help="Path to the pretrained .splat file"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8012, help="Server port")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of view (degrees)")

    args = parser.parse_args()

    visualizer = GrowingGaussianVisualizer(
        splat_path=args.splat_path,
        host=args.host,
        port=args.port,
        render_width=args.width,
        render_height=args.height,
        fov_deg=args.fov,
    )

    visualizer.visualize()


if __name__ == "__main__":
    main()
