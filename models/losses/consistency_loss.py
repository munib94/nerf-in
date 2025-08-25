"""Multi-view consistency losses for NeRF-In."""

import numpy as np
from typing import Dict, Any, List
from models.backends import create_backend

class ConsistencyLoss:
    """Multi-view consistency losses for inpainting."""
    
    def __init__(self, consistency_weight: float = 0.01):
        self.consistency_weight = consistency_weight
        self.backend = create_backend()
    
    def compute_multi_view_consistency(self, outputs_list: List[Dict[str, Any]], 
                                     camera_poses: List[Any]) -> Any:
        """Compute multi-view consistency loss.
        
        Args:
            outputs_list: List of outputs from different viewpoints
            camera_poses: List of camera poses for each view
            
        Returns:
            loss: Multi-view consistency loss
        """
        if len(outputs_list) < 2:
            return self.backend.zeros((1,))
        
        consistency_loss = 0
        num_pairs = 0
        
        # Compare all pairs of views
        for i in range(len(outputs_list)):
            for j in range(i + 1, len(outputs_list)):
                # Get RGB outputs from both views
                rgb_i = outputs_list[i].get('rgb_fine', outputs_list[i].get('rgb_coarse'))
                rgb_j = outputs_list[j].get('rgb_fine', outputs_list[j].get('rgb_coarse'))
                
                if rgb_i is not None and rgb_j is not None:
                    # Simple consistency loss (can be made more sophisticated)
                    pair_loss = self.backend.mse_loss(rgb_i, rgb_j)
                    consistency_loss += pair_loss
                    num_pairs += 1
        
        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs
        
        return consistency_loss * self.consistency_weight
    
    def compute_temporal_consistency(self, outputs_prev: Dict[str, Any], 
                                   outputs_curr: Dict[str, Any]) -> Any:
        """Compute temporal consistency loss between frames.
        
        Args:
            outputs_prev: Outputs from previous frame
            outputs_curr: Outputs from current frame
            
        Returns:
            loss: Temporal consistency loss
        """
        rgb_prev = outputs_prev.get('rgb_fine', outputs_prev.get('rgb_coarse'))
        rgb_curr = outputs_curr.get('rgb_fine', outputs_curr.get('rgb_coarse'))
        
        if rgb_prev is None or rgb_curr is None:
            return self.backend.zeros((1,))
        
        temporal_loss = self.backend.mse_loss(rgb_prev, rgb_curr)
        return temporal_loss * self.consistency_weight
