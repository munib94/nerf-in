"""RGB-D dataset handling for NeRF-In."""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image

from utils.camera_utils import get_camera_rays
from utils.image_utils import load_image, load_depth

class RGBDDataset:
    """RGB-D dataset for NeRF-In training."""
    
    def __init__(self, data_path: str, split: str = 'train', 
                 image_height: int = 480, image_width: int = 640,
                 near_plane: float = 0.1, far_plane: float = 10.0):
        self.data_path = Path(data_path)
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.near_plane = near_plane
        self.far_plane = far_plane
        
        # Load dataset metadata
        self.images = []
        self.depths = []
        self.poses = []
        self.intrinsics = None
        self.inpaint_masks = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset files and metadata."""
        split_file = self.data_path / f"{self.split}.txt"
        
        # Load file list
        if split_file.exists():
            with open(split_file, 'r') as f:
                file_list = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use all images in directory
            image_dir = self.data_path / "images"
            file_list = sorted([f.stem for f in image_dir.glob("*.png")])
        
        # Load images, depths, and poses
        for filename in file_list:
            # RGB image
            img_path = self.data_path / "images" / f"{filename}.png"
            if img_path.exists():
                self.images.append(str(img_path))
            
            # Depth image
            depth_path = self.data_path / "depth" / f"{filename}.png"
            if depth_path.exists():
                self.depths.append(str(depth_path))
            else:
                self.depths.append(None)
            
            # Camera pose
            pose_path = self.data_path / "poses" / f"{filename}.txt"
            if pose_path.exists():
                pose = np.loadtxt(pose_path).astype(np.float32)
                self.poses.append(pose)
            
            # Inpainting mask
            mask_path = self.data_path / "masks" / f"{filename}.png"
            if mask_path.exists():
                self.inpaint_masks.append(str(mask_path))
            else:
                self.inpaint_masks.append(None)
        
        # Load camera intrinsics
        intrinsics_path = self.data_path / "intrinsics.txt"
        if intrinsics_path.exists():
            self.intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)
        else:
            # Default intrinsics (may need adjustment)
            focal_length = max(self.image_width, self.image_height)
            self.intrinsics = np.array([
                [focal_length, 0, self.image_width / 2],
                [0, focal_length, self.image_height / 2],
                [0, 0, 1]
            ], dtype=np.float32)
        
        print(f"Loaded {len(self.images)} images for {self.split} split")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary containing sample data
        """
        # Load RGB image
        image = load_image(self.images[idx], target_size=(self.image_height, self.image_width))
        
        # Load depth image
        depth = None
        if self.depths[idx] is not None:
            depth = load_depth(self.depths[idx], target_size=(self.image_height, self.image_width))
        
        # Get camera pose
        pose = self.poses[idx] if idx < len(self.poses) else np.eye(4, dtype=np.float32)
        
        # Load inpainting mask
        inpaint_mask = None
        if self.inpaint_masks[idx] is not None:
            mask_img = load_image(self.inpaint_masks[idx], target_size=(self.image_height, self.image_width))
            inpaint_mask = (mask_img[..., 0] > 0.5).astype(np.float32)  # Binary mask
        
        # Generate rays for this view
        rays_o, rays_d = get_camera_rays(self.image_height, self.image_width, 
                                       self.intrinsics, pose)
        
        sample = {
            'image': image,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'pose': pose,
            'intrinsics': self.intrinsics,
            'idx': idx
        }
        
        if depth is not None:
            sample['depth'] = depth
        
        if inpaint_mask is not None:
            sample['inpaint_mask'] = inpaint_mask
        
        return sample
    
    def get_rays_batch(self, batch_size: int, idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get a batch of rays for training.
        
        Args:
            batch_size: Number of rays to sample
            idx: Optional specific image index
            
        Returns:
            ray_batch: Dictionary containing ray batch data
        """
        if idx is None:
            idx = np.random.randint(0, len(self))
        
        sample = self[idx]
        image = sample['image']
        rays_o = sample['rays_o']
        rays_d = sample['rays_d']
        
        # Flatten spatial dimensions
        H, W = image.shape[:2]
        image_flat = image.reshape(-1, 3)
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # Sample random rays
        ray_indices = np.random.choice(H * W, batch_size, replace=False)
        
        ray_batch = {
            'ray_origins': rays_o_flat[ray_indices],
            'ray_directions': rays_d_flat[ray_indices],
            'rgb_gt': image_flat[ray_indices],
            'image_idx': np.full((batch_size,), idx, dtype=np.int32)
        }
        
        # Add depth if available
        if 'depth' in sample:
            depth_flat = sample['depth'].reshape(-1)
            ray_batch['depth_gt'] = depth_flat[ray_indices]
        
        # Add inpainting mask if available
        if 'inpaint_mask' in sample:
            mask_flat = sample['inpaint_mask'].reshape(-1)
            ray_batch['inpaint_mask'] = mask_flat[ray_indices]
        
        return ray_batch
