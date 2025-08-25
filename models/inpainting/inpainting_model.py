"""Main NeRF-In inpainting model."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from models.backends import create_backend
from models.nerf.nerf_model import NeRFModel
from models.nerf.ray_sampling import RaySampler
from models.nerf.volume_rendering import VolumeRenderer
from config.base_config import ModelConfig, TrainingConfig

class NeRFInModel:
    """NeRF-In: NeRF Inpainting with RGB-D Priors."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.backend = create_backend()
        
        # Core NeRF components
        self.nerf_coarse = NeRFModel(config)
        self.nerf_fine = NeRFModel(config) if config.use_fine_network else None
        
        # Sampling and rendering
        self.ray_sampler = RaySampler(config.near_plane, config.far_plane)
        self.volume_renderer = VolumeRenderer()
        
        # Inpainting specific components
        self.use_rgbd_prior = config.use_rgbd_prior
        self.depth_weight = config.depth_weight
        self.consistency_weight = config.consistency_weight
    
    def forward(self, ray_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass through NeRF-In model.
        
        Args:
            ray_batch: Dictionary containing:
                - ray_origins: [N, 3] ray origins
                - ray_directions: [N, 3] ray directions  
                - rgb_gt: [N, 3] ground truth RGB (if available)
                - depth_gt: [N] ground truth depth (if available)
                - inpaint_mask: [N] binary mask for inpainting regions
                
        Returns:
            outputs: Dictionary containing rendered outputs
        """
        ray_origins = ray_batch['ray_origins']
        ray_directions = ray_batch['ray_directions']
        batch_size = ray_origins.shape[0]
        
        outputs = {}
        
        # Coarse sampling and rendering
        t_vals_coarse = self.ray_sampler.sample_rays_stratified(
            self.config.num_samples_coarse, batch_size
        )
        
        # Get 3D points along rays
        pts_coarse = self.ray_sampler.get_ray_points(ray_origins, ray_directions, t_vals_coarse)
        
        # Flatten for network input
        pts_flat = self.backend.reshape(pts_coarse, (-1, 3))
        dirs_flat = ray_directions[:, None, :].repeat(1, self.config.num_samples_coarse, 1)
        dirs_flat = self.backend.reshape(dirs_flat, (-1, 3))
        
        # Forward through coarse network
        rgb_coarse, density_coarse = self.nerf_coarse.forward(pts_flat, dirs_flat)
        
        # Reshape back
        rgb_coarse = self.backend.reshape(rgb_coarse, (batch_size, self.config.num_samples_coarse, 3))
        density_coarse = self.backend.reshape(density_coarse, (batch_size, self.config.num_samples_coarse, 1))
        
        # Volume rendering for coarse network
        rgb_map_coarse, depth_map_coarse, weights_coarse = self.volume_renderer.render_rays(
            rgb_coarse, density_coarse, t_vals_coarse, ray_directions
        )
        
        outputs['rgb_coarse'] = rgb_map_coarse
        outputs['depth_coarse'] = depth_map_coarse
        outputs['weights_coarse'] = weights_coarse
        
        # Fine sampling and rendering (if enabled)
        if self.nerf_fine is not None and self.config.use_importance_sampling:
            # Importance sampling based on coarse weights
            t_vals_fine = self.ray_sampler.sample_importance(
                t_vals_coarse, weights_coarse, self.config.num_samples_fine
            )
            
            # Combine coarse and fine samples
            t_vals_combined = self.backend.concatenate([t_vals_coarse, t_vals_fine], axis=-1)
            t_vals_combined, _ = self.backend.sort(t_vals_combined, axis=-1)
            
            # Get fine points
            pts_fine = self.ray_sampler.get_ray_points(ray_origins, ray_directions, t_vals_combined)
            
            # Flatten for network input
            pts_flat = self.backend.reshape(pts_fine, (-1, 3))
            dirs_flat = ray_directions[:, None, :].repeat(1, t_vals_combined.shape[-1], 1)
            dirs_flat = self.backend.reshape(dirs_flat, (-1, 3))
            
            # Forward through fine network
            rgb_fine, density_fine = self.nerf_fine.forward(pts_flat, dirs_flat)
            
            # Reshape back
            total_samples = t_vals_combined.shape[-1]
            rgb_fine = self.backend.reshape(rgb_fine, (batch_size, total_samples, 3))
            density_fine = self.backend.reshape(density_fine, (batch_size, total_samples, 1))
            
            # Volume rendering for fine network
            rgb_map_fine, depth_map_fine, weights_fine = self.volume_renderer.render_rays(
                rgb_fine, density_fine, t_vals_combined, ray_directions
            )
            
            outputs['rgb_fine'] = rgb_map_fine
            outputs['depth_fine'] = depth_map_fine
            outputs['weights_fine'] = weights_fine
        
        # Apply RGB-D priors for inpainting regions
        if self.use_rgbd_prior and 'inpaint_mask' in ray_batch:
            outputs = self._apply_rgbd_priors(outputs, ray_batch)
        
        return outputs
    
    def _apply_rgbd_priors(self, outputs: Dict[str, Any], ray_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply RGB-D priors for inpainting regions.
        
        Args:
            outputs: Current model outputs
            ray_batch: Input ray batch with priors
            
        Returns:
            outputs: Updated outputs with prior integration
        """
        inpaint_mask = ray_batch.get('inpaint_mask', None)
        depth_prior = ray_batch.get('depth_prior', None)
        
        if inpaint_mask is None:
            return outputs
        
        # Apply depth prior constraint in inpainting regions
        if depth_prior is not None and 'depth_fine' in outputs:
            depth_rendered = outputs['depth_fine']
            
            # Blend rendered depth with prior in inpainting regions
            depth_blended = depth_rendered * (1 - inpaint_mask) + depth_prior * inpaint_mask
            outputs['depth_prior_blended'] = depth_blended
        
        # Store inpainting mask for loss computation
        outputs['inpaint_mask'] = inpaint_mask
        
        return outputs
    
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
            chunk_outputs = self.forward(ray_chunk)
            
            # Accumulate outputs
            for key, value in chunk_outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value)
        
        # Concatenate all chunks
        for key in all_outputs:
            all_outputs[key] = self.backend.concatenate(all_outputs[key], axis=0)
        
        return all_outputs
