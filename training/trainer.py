"""Main training loop for NeRF-In."""

import os
import time
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from models.backends import create_backend
from models.inpainting.inpainting_model import NeRFInModel
from models.losses.rendering_loss import RenderingLoss
from models.losses.consistency_loss import ConsistencyLoss
from data.loaders.data_utils import create_data_loader
from config.base_config import BaseConfig
from utils.logging_utils import setup_logging
from evaluation.metrics import compute_metrics

class NeRFInTrainer:
    """Main trainer for NeRF-In model."""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.backend = create_backend()
        
        # Setup logging
        self.logger = setup_logging(config.log_dir)
        
        # Initialize model and losses
        self.model = NeRFInModel(config.model)
        self.rendering_loss = RenderingLoss(
            config.training.rgb_weight, 
            config.training.depth_weight
        )
        self.consistency_loss = ConsistencyLoss(config.training.consistency_weight)
        
        # Setup optimizer (backend-specific)
        self._setup_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        
    def _setup_optimizer(self):
        """Setup optimizer based on backend."""
        if self.backend.backend_name == "pytorch":
            import torch.optim as optim
            # Get all parameters from both networks
            params = []
            if hasattr(self.model.nerf_coarse, 'parameters'):
                params.extend(self.model.nerf_coarse.parameters())
            if self.model.nerf_fine and hasattr(self.model.nerf_fine, 'parameters'):
                params.extend(self.model.nerf_fine.parameters())
            
            self.optimizer = optim.Adam(params, lr=self.config.training.learning_rate)
            
        elif self.backend.backend_name == "mlx":
            import mlx.optimizers as optim
            # MLX optimizer setup
            self.optimizer = optim.Adam(learning_rate=self.config.training.learning_rate)
        
        else:
            raise NotImplementedError(f"Optimizer not implemented for {self.backend.backend_name}")
    
    def train_epoch(self, data_loader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            metrics: Dictionary of training metrics
        """
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(data_loader):
            # Forward pass
            outputs = self.model.forward(batch)
            
            # Compute losses
            rendering_losses = self.rendering_loss.compute_total_loss(outputs, batch)
            total_loss = rendering_losses['total_loss']
            
            # Add consistency loss if multiple views available
            if 'multi_view_outputs' in batch:
                consistency_loss = self.consistency_loss.compute_multi_view_consistency(
                    batch['multi_view_outputs'], batch['camera_poses']
                )
                total_loss += consistency_loss
            
            # Backward pass (backend-specific)
            self._backward_pass(total_loss)
            
            # Update metrics
            epoch_losses.append(self.backend.to_numpy(total_loss))
            self.global_step += 1
            
            # Logging
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {epoch_losses[-1]:.6f}"
                )
        
        # Compute epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'learning_rate': self.config.training.learning_rate
        }
    
    def _backward_pass(self, loss):
        """Perform backward pass based on backend."""
        if self.backend.backend_name == "pytorch":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        elif self.backend.backend_name == "mlx":
            # MLX-specific gradient computation and update
            # This is a simplified version - MLX has its own gradient system
            pass
        
        else:
            raise NotImplementedError(f"Backward pass not implemented for {self.backend.backend_name}")
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            metrics: Validation metrics
        """
        val_losses = []
        val_psnrs = []
        
        # Set model to evaluation mode (if applicable)
        self._set_eval_mode()
        
        try:
            for batch in val_loader:
                # Forward pass without gradients
                outputs = self._forward_no_grad(batch)
                
                # Compute losses
                losses = self.rendering_loss.compute_total_loss(outputs, batch)
                val_losses.append(self.backend.to_numpy(losses['total_loss']))
                
                # Compute metrics
                if 'rgb_fine' in outputs or 'rgb_coarse' in outputs:
                    rgb_pred = outputs.get('rgb_fine', outputs['rgb_coarse'])
                    rgb_gt = batch['rgb_gt']
                    
                    psnr = compute_metrics(
                        self.backend.to_numpy(rgb_pred),
                        self.backend.to_numpy(rgb_gt)
                    )['psnr']
                    val_psnrs.append(psnr)
        
        finally:
            # Set model back to training mode
            self._set_train_mode()
        
        return {
            'val_loss': np.mean(val_losses),
            'val_psnr': np.mean(val_psnrs) if val_psnrs else 0.0
        }
    
    def _forward_no_grad(self, batch):
        """Forward pass without gradients."""
        if self.backend.backend_name == "pytorch":
            import torch
            with torch.no_grad():
                return self.model.forward(batch)
        else:
            # For MLX, gradients are computed explicitly
            return self.model.forward(batch)
    
    def _set_eval_mode(self):
        """Set model to evaluation mode."""
        if self.backend.backend_name == "pytorch":
            if hasattr(self.model.nerf_coarse, 'eval'):
                self.model.nerf_coarse.eval()
            if self.model.nerf_fine and hasattr(self.model.nerf_fine, 'eval'):
                self.model.nerf_fine.eval()
    
    def _set_train_mode(self):
        """Set model to training mode."""
        if self.backend.backend_name == "pytorch":
            if hasattr(self.model.nerf_coarse, 'train'):
                self.model.nerf_coarse.train()
            if self.model.nerf_fine and hasattr(self.model.nerf_fine, 'train'):
                self.model.nerf_fine.train()
    
    def train(self, train_loader, val_loader=None):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self.logger.info(f"Starting training with {self.backend.backend_name} backend")
        self.logger.info(f"Training for {self.config.training.num_epochs} epochs")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Val PSNR: {val_metrics.get('val_psnr', 0.0):.3f} | "
                f"Time: {train_metrics['epoch_time']:.1f}s"
            )
            
            # Save checkpoint
            if val_metrics.get('val_psnr', 0) > self.best_psnr:
                self.best_psnr = val_metrics['val_psnr']
                self.save_checkpoint('best.pt')
            
            # Regular checkpoint saving
            if epoch % 50 == 0:
                self.save_checkpoint(f'epoch_{epoch:03d}.pt')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backend-specific checkpoint saving
        if self.backend.backend_name == "pytorch":
            import torch
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_psnr': self.best_psnr,
                'model_state_dict': {
                    'nerf_coarse': self.model.nerf_coarse.state_dict(),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }
            
            if self.model.nerf_fine:
                checkpoint['model_state_dict']['nerf_fine'] = self.model.nerf_fine.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            
        elif self.backend.backend_name == "mlx":
            # MLX checkpoint saving
            # This would need to be implemented based on MLX's serialization
            self.logger.warning("MLX checkpoint saving not fully implemented")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.backend.backend_name == "pytorch":
            import torch
            checkpoint = torch.load(checkpoint_path, map_location=self.backend.device)
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_psnr = checkpoint['best_psnr']
            
            self.model.nerf_coarse.load_state_dict(checkpoint['model_state_dict']['nerf_coarse'])
            if 'nerf_fine' in checkpoint['model_state_dict'] and self.model.nerf_fine:
                self.model.nerf_fine.load_state_dict(checkpoint['model_state_dict']['nerf_fine'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
