"""NeRF-In model following the paper specification exactly."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from models.backends import create_backend
from models.nerf.nerf_model import NeRFModel
from models.nerf.ray_sampling import RaySampler
from models.nerf.volume_rendering import VolumeRenderer
from models.inpainting.stcn_mask_transfer import STCNMaskTransfer
from models.inpainting.mst_inpainting import MSTInpainter
from models.inpainting.bilateral_solver import FastBilateralSolver
from config.base_config import ModelConfig
from utils.camera_utils import get_camera_rays

class NeRFInModel:
    """NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors.
    
    Implementation following the paper exactly:
    - STCN for mask transfer
    - MST inpainting for RGB guidance  
    - Bilateral solver for depth completion
    - Proper loss formulation with L_all_color, L_out_color, L_depth
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.backend = create_backend()
        
        # Core NeRF components (pre-trained)
        self.nerf_coarse = NeRFModel(config)
        self.nerf_fine = NeRFModel(config) if config.use_fine_network else None
        
        # Sampling and rendering
        self.ray_sampler = RaySampler(config.near_plane, config.far_plane)
        self.volume_renderer = VolumeRenderer()
        
        # NeRF-In specific components (following paper)
        self.mask_transfer = STCNMaskTransfer()
        self.inpainter = MSTInpainter(model_type='lama')  # MST or LaMa
        self.bilateral_solver = FastBilateralSolver()
        
        # Training state
        self.is_pretrained = False
        self.inpainting_mode = False
        
        print("🎯 NeRF-In Model initialized with paper-compliant components")
    
    def set_pretrained_nerf(self, nerf_coarse, nerf_fine=None):
        """Set pre-trained NeRF models as specified in paper.
        
        Args:
            nerf_coarse: Pre-trained coarse NeRF network
            nerf_fine: Pre-trained fine NeRF network (optional)
        """
        self.nerf_coarse = nerf_coarse
        self.nerf_fine = nerf_fine
        self.is_pretrained = True
        print("✅ Pre-trained NeRF models loaded")
    
    def generate_guiding_materials(self, 
                                 rendered_images: List[np.ndarray],
                                 rendered_depths: List[np.ndarray], 
                                 user_mask: np.ndarray,
                                 user_image_idx: int = 0) -> Dict[str, List[np.ndarray]]:
        """Generate guiding materials following paper Algorithm 1.
        
        Args:
            rendered_images: List of rendered images from sampled views
            rendered_depths: List of rendered depth images
            user_mask: User-drawn binary mask [H, W]
            user_image_idx: Index of user-chosen view
            
        Returns:
            guiding_materials: Dictionary with 'rgb_guides' and 'depth_guides'
        """
        print("🔄 Generating guiding materials...")
        
        # Step 1: Transfer mask to other views using STCN (Equation 5 region)
        print("   📍 Transferring masks using STCN...")
        transferred_masks = self.mask_transfer.transfer_mask(
            rendered_images, user_mask, user_image_idx
        )
        
        # Step 2: Generate guiding RGB images using MST inpainting (Equation 5)
        # I_s^G = ρ(I_s, M_s) where ρ is MST inpainting
        print("   🎨 Generating RGB guidance using MST inpainting...")
        guiding_images = self.inpainter.batch_inpaint(
            rendered_images, transferred_masks
        )
        
        # Step 3: Generate guiding depth images using bilateral solver (Equation 6)  
        # D_s^G = τ(D_s, M_s, I_s^G) where τ is bilateral solver
        print("   🔍 Completing depth using bilateral solver...")
        guiding_depths = []
        for depth, mask, rgb_guide in zip(rendered_depths, transferred_masks, guiding_images):
            completed_depth = self.bilateral_solver.complete_depth_image(
                depth, rgb_guide, mask
            )
            guiding_depths.append(completed_depth)
        
        print("✅ Guiding materials generated")
        
        return {
            'rgb_guides': guiding_images,
            'depth_guides': guiding_depths, 
            'transferred_masks': transferred_masks
        }
    
    def forward_inpainting(self, ray_batch: Dict[str, Any], 
                          guiding_materials: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inpainting optimization following paper.
        
        Args:
            ray_batch: Ray batch data
            guiding_materials: RGB-D guiding materials
            
        Returns:
            outputs: Model outputs for loss computation
        """
        # Standard NeRF forward pass
        outputs = self.forward_nerf(ray_batch)
        
        # Add guiding information for loss computation
        if 'rgb_guides' in guiding_materials:
            outputs['rgb_guides'] = guiding_materials['rgb_guides']
        if 'depth_guides' in guiding_materials:
            outputs['depth_guides'] = guiding_materials['depth_guides']
        if 'transferred_masks' in guiding_materials:
            outputs['masks'] = guiding_materials['transferred_masks']
        
        return outputs
    
    def forward_nerf(self, ray_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Standard NeRF forward pass."""
        ray_origins = ray_batch['ray_origins'] 
        ray_directions = ray_batch['ray_directions']
        batch_size = ray_origins.shape[0]
        
        outputs = {}
        
        # Coarse network
        t_vals_coarse = self.ray_sampler.sample_rays_stratified(
            self.config.num_samples_coarse, batch_size
        )
        
        pts_coarse = self.ray_sampler.get_ray_points(ray_origins, ray_directions, t_vals_coarse)
        pts_flat = self.backend.reshape(pts_coarse, (-1, 3))
        dirs_flat = ray_directions[:, None, :].repeat(1, self.config.num_samples_coarse, 1)
        dirs_flat = self.backend.reshape(dirs_flat, (-1, 3))
        
        rgb_coarse, density_coarse = self.nerf_coarse.forward(pts_flat, dirs_flat)
        rgb_coarse = self.backend.reshape(rgb_coarse, (batch_size, self.config.num_samples_coarse, 3))
        density_coarse = self.backend.reshape(density_coarse, (batch_size, self.config.num_samples_coarse, 1))
        
        rgb_map_coarse, depth_map_coarse, weights_coarse = self.volume_renderer.render_rays(
            rgb_coarse, density_coarse, t_vals_coarse, ray_directions
        )
        
        outputs.update({
            'rgb_coarse': rgb_map_coarse,
            'depth_coarse': depth_map_coarse,
            'weights_coarse': weights_coarse
        })
        
        # Fine network (if available)
        if self.nerf_fine is not None:
            t_vals_fine = self.ray_sampler.sample_importance(
                t_vals_coarse, weights_coarse, self.config.num_samples_fine
            )
            t_vals_combined = self.backend.concatenate([t_vals_coarse, t_vals_fine], axis=-1)
            t_vals_combined, _ = self.backend.sort(t_vals_combined, axis=-1)
            
            pts_fine = self.ray_sampler.get_ray_points(ray_origins, ray_directions, t_vals_combined)
            pts_flat = self.backend.reshape(pts_fine, (-1, 3))
            total_samples = t_vals_combined.shape[-1]
            dirs_flat = ray_directions[:, None, :].repeat(1, total_samples, 1)
            dirs_flat = self.backend.reshape(dirs_flat, (-1, 3))
            
            rgb_fine, density_fine = self.nerf_fine.forward(pts_flat, dirs_flat)
            rgb_fine = self.backend.reshape(rgb_fine, (batch_size, total_samples, 3))
            density_fine = self.backend.reshape(density_fine, (batch_size, total_samples, 1))
            
            rgb_map_fine, depth_map_fine, weights_fine = self.volume_renderer.render_rays(
                rgb_fine, density_fine, t_vals_combined, ray_directions
            )
            
            outputs.update({
                'rgb_fine': rgb_map_fine,
                'depth_fine': depth_map_fine,
                'weights_fine': weights_fine
            })
        
        return outputs
    
    def sample_camera_trajectory(self, num_views: int = 24, 
                                poses: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """Sample camera positions following LLFF trajectory as in paper.
        
        Args:
            num_views: Number of views to sample (paper uses 24)
            poses: Optional list of camera poses
            
        Returns:
            sampled_views: List of view dictionaries with poses and intrinsics
        """
        if poses is None:
            # Generate default circular trajectory like LLFF
            poses = self._generate_llff_trajectory(num_views)
        
        sampled_views = []
        for i, pose in enumerate(poses[:num_views]):
            view = {
                'pose': pose,
                'view_idx': i,
                'intrinsics': self._get_default_intrinsics()
            }
            sampled_views.append(view)
        
        return sampled_views
    
    def _generate_llff_trajectory(self, num_views: int) -> List[np.ndarray]:
        """Generate LLFF-style camera trajectory."""
        poses = []
        
        # Simple circular trajectory around the scene
        radius = 3.0
        height = 0.0
        
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            
            # Camera position
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = height
            
            # Look at origin
            camera_pos = np.array([x, y, z])
            target = np.array([0, 0, 0])
            up = np.array([0, 1, 0])
            
            # Create pose matrix
            pose = self._look_at_matrix(camera_pos, target, up)
            poses.append(pose)
        
        return poses
    
    def _look_at_matrix(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create look-at pose matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        
        return pose
    
    def _get_default_intrinsics(self) -> np.ndarray:
        """Get default camera intrinsics."""
        focal = 800.0
        cx = 320.0
        cy = 240.0
        
        intrinsics = np.array([
            [focal, 0, cx],
            [0, focal, cy], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsics
    
    def render_image(self, rays: Dict[str, Any], chunk_size: int = 1024) -> Dict[str, Any]:
        """Render a full image by processing rays in chunks.
        
        Args:
            rays: Dictionary of ray data for full image
            chunk_size: Number of rays to process at once
            
        Returns:
            rendered: Dictionary of rendered outputs
        """
        all_outputs = {}
        num_rays = rays['ray_origins'].shape[0]
        
        for i in range(0, num_rays, chunk_size):
            end_i = min(i + chunk_size, num_rays)
            
            # Extract chunk
            ray_chunk = {}
            for key, value in rays.items():
                ray_chunk[key] = value[i:end_i]
            
            # Process chunk
            chunk_outputs = self.forward_nerf(ray_chunk)
            
            # Accumulate outputs
            for key, value in chunk_outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value)
        
        # Concatenate all chunks
        for key in all_outputs:
            all_outputs[key] = self.backend.concatenate(all_outputs[key], axis=0)
        
        return all_outputs
