"""Camera utilities for NeRF."""

import numpy as np
from typing import Tuple

def get_camera_rays(height: int, width: int, intrinsics: np.ndarray, 
                   pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate camera rays for all pixels.
    
    Args:
        height: Image height
        width: Image width
        intrinsics: [3, 3] camera intrinsic matrix
        pose: [4, 4] camera-to-world transformation matrix
        
    Returns:
        rays_o: [H, W, 3] ray origins
        rays_d: [H, W, 3] ray directions
    """
    # Create pixel coordinates
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Convert to normalized device coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Ray directions in camera coordinate system
    directions = np.stack([
        (i - cx) / fx,
        -(j - cy) / fy,  # Negative because image y-axis points down
        -np.ones_like(i)  # Negative z direction
    ], axis=-1)
    
    # Transform ray directions to world coordinate system
    rays_d = np.sum(directions[..., None, :] * pose[:3, :3], axis=-1)
    
    # Ray origins (camera center in world coordinates)
    rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
    
    return rays_o.astype(np.float32), rays_d.astype(np.float32)

def normalize_rays(rays_d: np.ndarray) -> np.ndarray:
    """Normalize ray directions.
    
    Args:
        rays_d: [..., 3] ray directions
        
    Returns:
        rays_d_norm: Normalized ray directions
    """
    return rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

def get_ray_points(rays_o: np.ndarray, rays_d: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """Get 3D points along rays.
    
    Args:
        rays_o: [..., 3] ray origins
        rays_d: [..., 3] ray directions
        t_vals: [..., N] parameter values along rays
        
    Returns:
        points: [..., N, 3] 3D points along rays
    """
    return rays_o[..., None, :] + t_vals[..., :, None] * rays_d[..., None, :]

def poses_avg(poses: np.ndarray) -> np.ndarray:
    """Compute average pose from multiple poses.
    
    Args:
        poses: [N, 4, 4] camera poses
        
    Returns:
        pose_avg: [4, 4] average pose
    """
    center = poses[:, :3, 3].mean(axis=0)
    vec2 = normalize_vector(poses[:, :3, 2].sum(axis=0))
    vec1_avg = poses[:, :3, 1].mean(axis=0)
    vec0 = normalize_vector(np.cross(vec1_avg, vec2))
    vec1 = normalize_vector(np.cross(vec2, vec0))
    
    pose_avg = np.eye(4)
    pose_avg[:3, :3] = np.stack([vec0, vec1, vec2], axis=1)
    pose_avg[:3, 3] = center
    
    return pose_avg

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)
