"""Ray sampling utilities for NeRF."""

import numpy as np
from typing import Tuple
from models.backends import create_backend

class RaySampler:
    """Handles ray sampling for NeRF training and inference."""
    
    def __init__(self, near: float = 0.1, far: float = 10.0):
        self.near = near
        self.far = far
        self.backend = create_backend()
    
    def sample_rays_uniform(self, num_samples: int, batch_size: int) -> Tuple:
        """Sample points uniformly along rays.
        
        Args:
            num_samples: Number of samples per ray
            batch_size: Number of rays
            
        Returns:
            t_vals: [batch_size, num_samples] sample distances
            pts: [batch_size, num_samples, 3] 3D points
        """
        # Uniform sampling between near and far
        t_vals = self.backend.create_tensor(
            np.linspace(self.near, self.far, num_samples)[None, :].repeat(batch_size, axis=0)
        )
        
        return t_vals
    
    def sample_rays_stratified(self, num_samples: int, batch_size: int) -> Tuple:
        """Sample points using stratified sampling.
        
        Args:
            num_samples: Number of samples per ray
            batch_size: Number of rays
            
        Returns:
            t_vals: [batch_size, num_samples] sample distances
        """
        # Create bins
        bins = np.linspace(self.near, self.far, num_samples + 1)
        lower = bins[:-1]
        upper = bins[1:]
        
        # Stratified sampling within each bin
        u = np.random.uniform(0, 1, (batch_size, num_samples))
        t_vals = lower[None, :] + (upper - lower)[None, :] * u
        
        return self.backend.create_tensor(t_vals)
    
    def sample_importance(self, t_vals_coarse, weights_coarse, num_samples_fine: int):
        """Importance sampling based on coarse weights.
        
        Args:
            t_vals_coarse: Coarse sample positions
            weights_coarse: Weights from coarse network
            num_samples_fine: Number of fine samples
            
        Returns:
            t_vals_fine: Fine sample positions
        """
        # Convert weights to probability distribution
        weights = weights_coarse + 1e-5  # Prevent divide by zero
        pdf = weights / self.backend.sum(weights, axis=-1, keepdim=True)
        cdf = self.backend.cumsum(pdf, axis=-1)
        cdf = self.backend.concatenate([self.backend.zeros_like(cdf[..., :1]), cdf], axis=-1)
        
        # Inverse transform sampling
        u = self.backend.random((cdf.shape[0], num_samples_fine))
        
        # Find indices for sampling
        indices = self.backend.searchsorted(cdf, u, side='right')
        below = self.backend.maximum(self.backend.zeros_like(indices), indices - 1)
        above = self.backend.minimum((cdf.shape[-1] - 1) * self.backend.ones_like(indices), indices)
        
        # Linear interpolation
        t_vals_fine = t_vals_coarse[..., below] + (t_vals_coarse[..., above] - t_vals_coarse[..., below]) * u
        
        return t_vals_fine
    
    def get_ray_points(self, ray_origins, ray_directions, t_vals):
        """Get 3D points along rays.
        
        Args:
            ray_origins: [batch_size, 3] ray origins
            ray_directions: [batch_size, 3] ray directions
            t_vals: [batch_size, num_samples] sample distances
            
        Returns:
            pts: [batch_size, num_samples, 3] 3D points
        """
        # pts = origin + t * direction
        pts = ray_origins[..., None, :] + t_vals[..., :, None] * ray_directions[..., None, :]
        return pts
