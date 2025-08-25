"""Volume rendering implementation for NeRF."""

import numpy as np
from models.backends import create_backend

class VolumeRenderer:
    """Handles volume rendering for NeRF."""
    
    def __init__(self):
        self.backend = create_backend()
    
    def render_rays(self, rgb, density, t_vals, ray_directions, white_background=False):
        """Render rays using volume rendering equation.
        
        Args:
            rgb: [batch_size, num_samples, 3] RGB values
            density: [batch_size, num_samples, 1] density values
            t_vals: [batch_size, num_samples] sample distances
            ray_directions: [batch_size, 3] ray directions
            white_background: Whether to use white background
            
        Returns:
            rgb_map: [batch_size, 3] rendered RGB
            depth_map: [batch_size] rendered depth
            weights: [batch_size, num_samples] rendering weights
        """
        # Calculate distances between samples
        delta = t_vals[..., 1:] - t_vals[..., :-1]
        
        # Add small value for last sample (infinity approximation)
        inf_delta = self.backend.ones((*delta.shape[:-1], 1)) * 1e10
        delta = self.backend.concatenate([delta, inf_delta], axis=-1)
        
        # Multiply by ray direction norm for proper distances
        ray_norms = self.backend.norm(ray_directions, axis=-1, keepdim=True)
        delta = delta * ray_norms[..., None]
        
        # Calculate alpha values
        alpha = 1.0 - self.backend.exp(-density[..., 0] * delta)
        
        # Calculate transmittance (cumulative product of (1-alpha))
        ones = self.backend.ones((*alpha.shape[:-1], 1))
        transmittance = self.backend.cumprod(
            self.backend.concatenate([ones, 1.0 - alpha + 1e-10], axis=-1), axis=-1
        )[..., :-1]
        
        # Calculate weights
        weights = alpha * transmittance
        
        # Render RGB
        rgb_map = self.backend.sum(weights[..., None] * rgb, axis=-2)
        
        # Render depth
        depth_map = self.backend.sum(weights * t_vals, axis=-1)
        
        # Handle background
        if white_background:
            acc_map = self.backend.sum(weights, axis=-1)
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        
        return rgb_map, depth_map, weights
    
    def render_depth(self, depth_vals, weights):
        """Render depth map from depth values and weights.
        
        Args:
            depth_vals: [batch_size, num_samples] depth values
            weights: [batch_size, num_samples] rendering weights
            
        Returns:
            depth_map: [batch_size] rendered depth
        """
        return self.backend.sum(weights * depth_vals, axis=-1)
