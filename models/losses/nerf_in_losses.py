"""NeRF-In specific losses following the paper exactly."""

import numpy as np
from typing import Dict, Any, List
from models.backends import create_backend

class NeRFInLoss:
    """NeRF-In losses following paper Equations 8, 9, 10, 11."""
    
    def __init__(self, config):
        self.config = config
        self.backend = create_backend()
        
        # Loss weights from paper
        self.rgb_weight = getattr(config.training, 'rgb_weight', 1.0)
        self.depth_weight = getattr(config.training, 'depth_weight', 0.1)
    
    def compute_color_guiding_loss(self, outputs: Dict[str, Any], 
                                 targets: Dict[str, Any],
                                 view_assignments: Dict[str, List[int]]) -> Dict[str, Any]:
        """Compute color-guiding loss following paper Equation 8.
        
        L_color(Θ) = L_all_color(Θ) + L_out_color(Θ)
        
        Args:
            outputs: Model outputs
            targets: Target values with guiding images
            view_assignments: Which views belong to O_all vs O_out
            
        Returns:
            color_losses: Dictionary of color loss components
        """
        losses = {}
        
        # L_all_color: loss on entire image for user-chosen view (Equation 9)
        # L_all_color(Θ) = Σ_{o∈O_all} ||F^image_Θ(o) - I^G_o||
        if 'O_all' in view_assignments:
            l_all_color = 0
            count = 0
            
            for view_idx in view_assignments['O_all']:
                if view_idx < len(outputs.get('rgb_fine', outputs.get('rgb_coarse', []))):
                    rgb_pred = outputs.get('rgb_fine', outputs['rgb_coarse'])[view_idx]
                    rgb_guide = targets['rgb_guides'][view_idx]
                    
                    # MSE loss on entire image
                    l_all_color += self.backend.mse_loss(rgb_pred, rgb_guide)
                    count += 1
            
            if count > 0:
                l_all_color = l_all_color / count
            losses['L_all_color'] = l_all_color
        
        # L_out_color: loss outside mask for sampled views (Equation 10)  
        # L_out_color(Θ) = Σ_{o∈O_out} (F^image_Θ(o) - I^G_o) ⊙ M_o
        if 'O_out' in view_assignments:
            l_out_color = 0
            count = 0
            
            for view_idx in view_assignments['O_out']:
                if (view_idx < len(outputs.get('rgb_fine', outputs.get('rgb_coarse', []))) and
                    view_idx < len(targets.get('masks', []))):
                    
                    rgb_pred = outputs.get('rgb_fine', outputs['rgb_coarse'])[view_idx]
                    rgb_guide = targets['rgb_guides'][view_idx]
                    mask = targets['masks'][view_idx]  # M_o
                    
                    # Apply mask: only compute loss outside inpainting regions
                    outside_mask = 1.0 - mask  # Invert mask for outside regions
                    masked_loss = self.backend.mse_loss(rgb_pred, rgb_guide) * outside_mask
                    
                    l_out_color += self.backend.sum(masked_loss) / (self.backend.sum(outside_mask) + 1e-8)
                    count += 1
            
            if count > 0:
                l_out_color = l_out_color / count
            losses['L_out_color'] = l_out_color
        
        # Total color loss
        total_color_loss = losses.get('L_all_color', 0) + losses.get('L_out_color', 0)
        losses['L_color_total'] = total_color_loss
        
        return losses
    
    def compute_depth_guiding_loss(self, outputs: Dict[str, Any], 
                                 targets: Dict[str, Any]) -> Dict[str, Any]:
        """Compute depth-guiding loss following paper Equation 11.
        
        L_depth(Θ) = Σ_{o_s∈O} ||D^f(o_s) - D^G(o_s)||²₂ + ||D^c(o_s) - D^G(o_s)||²₂
        
        Args:
            outputs: Model outputs with rendered depths
            targets: Target values with guiding depths
            
        Returns:
            depth_losses: Dictionary of depth loss components  
        """
        losses = {}
        
        if 'depth_guides' not in targets:
            losses['L_depth'] = self.backend.zeros((1,))
            return losses
        
        total_depth_loss = 0
        count = 0
        
        # Sum over all sampled views O
        num_views = len(targets['depth_guides'])
        
        for view_idx in range(num_views):
            if view_idx < len(outputs.get('depth_fine', outputs.get('depth_coarse', []))):
                depth_guide = targets['depth_guides'][view_idx]  # D^G(o_s)
                
                # Fine volume predicted depth D^f(o_s)
                if 'depth_fine' in outputs:
                    depth_fine = outputs['depth_fine'][view_idx]
                    fine_loss = self.backend.mse_loss(depth_fine, depth_guide)
                    total_depth_loss += fine_loss
                    count += 1
                
                # Coarse volume predicted depth D^c(o_s) 
                if 'depth_coarse' in outputs:
                    depth_coarse = outputs['depth_coarse'][view_idx]
                    coarse_loss = self.backend.mse_loss(depth_coarse, depth_guide)
                    total_depth_loss += coarse_loss
                    count += 1
        
        if count > 0:
            total_depth_loss = total_depth_loss / count
        
        losses['L_depth'] = total_depth_loss
        return losses
    
    def compute_nerf_in_total_loss(self, outputs: Dict[str, Any], 
                                  targets: Dict[str, Any],
                                  view_assignments: Dict[str, List[int]]) -> Dict[str, Any]:
        """Compute total NeRF-In loss following paper Equation 7.
        
        Θ̃ := arg min_Θ L_color(Θ) + L_depth(Θ)
        
        Args:
            outputs: Model outputs
            targets: Targets with guiding materials
            view_assignments: View assignments for loss computation
            
        Returns:
            all_losses: Dictionary with all loss components and total
        """
        all_losses = {}
        
        # Compute color-guiding loss (Equation 8)
        color_losses = self.compute_color_guiding_loss(outputs, targets, view_assignments)
        all_losses.update(color_losses)
        
        # Compute depth-guiding loss (Equation 11)
        depth_losses = self.compute_depth_guiding_loss(outputs, targets)
        all_losses.update(depth_losses)
        
        # Total NeRF-In loss (Equation 7)
        total_loss = (self.rgb_weight * color_losses.get('L_color_total', 0) + 
                     self.depth_weight * depth_losses.get('L_depth', 0))
        
        all_losses['total_loss'] = total_loss
        all_losses['weighted_color_loss'] = self.rgb_weight * color_losses.get('L_color_total', 0)
        all_losses['weighted_depth_loss'] = self.depth_weight * depth_losses.get('L_depth', 0)
        
        return all_losses
    
    def create_view_assignments(self, num_views: int, user_view_idx: int = 0) -> Dict[str, List[int]]:
        """Create view assignments following paper methodology.
        
        Paper specifies:
        - O_all = {o_u} (only user-chosen view)
        - O_out = O (all sampled views)
        
        Args:
            num_views: Total number of sampled views
            user_view_idx: Index of user-chosen view
            
        Returns:
            assignments: Dictionary mapping 'O_all' and 'O_out' to view indices
        """
        assignments = {
            'O_all': [user_view_idx],  # Only user-chosen view
            'O_out': list(range(num_views))  # All sampled views
        }
        
        return assignments
