"""Rendering losses for NeRF-In."""

import numpy as np
from typing import Dict, Any
from models.backends import create_backend

class RenderingLoss:
    """Handles rendering-related losses."""
    
    def __init__(self, rgb_weight: float = 1.0, depth_weight: float = 0.1):
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.backend = create_backend()
    
    def compute_rgb_loss(self, rgb_pred: Any, rgb_gt: Any, mask: Any = None) -> Any:
        """Compute RGB reconstruction loss.
        
        Args:
            rgb_pred: Predicted RGB values
            rgb_gt: Ground truth RGB values  
            mask: Optional mask for selective loss computation
            
        Returns:
            loss: RGB loss value
        """
        mse_loss = self.backend.mse_loss(rgb_pred, rgb_gt)
        
        if mask is not None:
            # Apply mask weighting
            mse_loss = mse_loss * mask
            mse_loss = self.backend.sum(mse_loss) / (self.backend.sum(mask) + 1e-8)
        
        return mse_loss
    
    def compute_depth_loss(self, depth_pred: Any, depth_gt: Any, mask: Any = None) -> Any:
        """Compute depth reconstruction loss.
        
        Args:
            depth_pred: Predicted depth values
            depth_gt: Ground truth depth values
            mask: Optional mask for selective loss computation
            
        Returns:
            loss: Depth loss value
        """
        # Use L1 loss for depth (more robust to outliers)
        l1_loss = self.backend.l1_loss(depth_pred, depth_gt)
        
        if mask is not None:
            l1_loss = l1_loss * mask
            l1_loss = self.backend.sum(l1_loss) / (self.backend.sum(mask) + 1e-8)
        
        return l1_loss
    
    def compute_total_loss(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Compute total rendering loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            
        Returns:
            losses: Dictionary of individual and total losses
        """
        losses = {}
        total_loss = 0
        
        # RGB loss for coarse network
        if 'rgb_coarse' in outputs and 'rgb_gt' in targets:
            rgb_loss_coarse = self.compute_rgb_loss(
                outputs['rgb_coarse'], targets['rgb_gt']
            )
            losses['rgb_loss_coarse'] = rgb_loss_coarse
            total_loss += self.rgb_weight * rgb_loss_coarse
        
        # RGB loss for fine network  
        if 'rgb_fine' in outputs and 'rgb_gt' in targets:
            rgb_loss_fine = self.compute_rgb_loss(
                outputs['rgb_fine'], targets['rgb_gt']
            )
            losses['rgb_loss_fine'] = rgb_loss_fine
            total_loss += self.rgb_weight * rgb_loss_fine
        
        # Depth loss
        depth_key = 'depth_fine' if 'depth_fine' in outputs else 'depth_coarse'
        if depth_key in outputs and 'depth_gt' in targets:
            depth_loss = self.compute_depth_loss(
                outputs[depth_key], targets['depth_gt']
            )
            losses['depth_loss'] = depth_loss
            total_loss += self.depth_weight * depth_loss
        
        losses['total_loss'] = total_loss
        return losses
