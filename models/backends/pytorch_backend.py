"""PyTorch backend implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Optional
import numpy as np
from .base_backend import BaseBackend
from utils.backend_detector import BACKEND

class PyTorchMLP(nn.Module):
    """PyTorch MLP implementation with skip connections."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 activation: str = "relu", skip_connections: list = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.skip_connections = skip_connections or []
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Add skip connection input if needed
            in_dim = dims[i]
            if i in self.skip_connections:
                in_dim += input_dim
            
            layers.append(nn.Linear(in_dim, dims[i + 1]))
            
            # Add activation (except for output layer)
            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        input_x = x
        layer_idx = 0
        
        for i in range(0, len(self.layers), 2):  # Step by 2 (linear + activation)
            # Add skip connection if needed
            if layer_idx in self.skip_connections:
                x = torch.cat([x, input_x], dim=-1)
            
            # Apply linear layer
            x = self.layers[i](x)
            
            # Apply activation if not the last layer
            if i + 1 < len(self.layers):
                x = self.layers[i + 1](x)
            
            layer_idx += 1
        
        return x

class PyTorchBackend(BaseBackend):
    """PyTorch backend implementation."""
    
    def __init__(self):
        self._device = BACKEND.device
    
    def create_tensor(self, data: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        """Create a tensor from numpy array."""
        return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad, device=self._device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().numpy()
    
    def zeros(self, shape: Tuple[int, ...], requires_grad: bool = False) -> torch.Tensor:
        """Create zero tensor with given shape."""
        return torch.zeros(shape, dtype=torch.float32, requires_grad=requires_grad, device=self._device)
    
    def ones(self, shape: Tuple[int, ...], requires_grad: bool = False) -> torch.Tensor:
        """Create ones tensor with given shape."""
        return torch.ones(shape, dtype=torch.float32, requires_grad=requires_grad, device=self._device)
    
    def random(self, shape: Tuple[int, ...], requires_grad: bool = False) -> torch.Tensor:
        """Create random tensor with given shape."""
        return torch.randn(shape, dtype=torch.float32, requires_grad=requires_grad, device=self._device)
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication."""
        return torch.matmul(a, b)
    
    def sin(self, x: torch.Tensor) -> torch.Tensor:
        """Sine function."""
        return torch.sin(x)
    
    def cos(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine function."""
        return torch.cos(x)
    
    def concatenate(self, tensors: list, axis: int = 0) -> torch.Tensor:
        """Concatenate tensors along given axis."""
        return torch.cat(tensors, dim=axis)
    
    def reshape(self, tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape tensor."""
        return tensor.reshape(shape)
    
    def create_mlp(self, input_dim: int, hidden_dims: list, output_dim: int,
                   activation: str = "relu", skip_connections: list = None) -> PyTorchMLP:
        """Create MLP network."""
        return PyTorchMLP(input_dim, hidden_dims, output_dim, activation, skip_connections).to(self._device)
    
    def mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared error loss."""
        return F.mse_loss(pred, target)
    
    def l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss."""
        return F.l1_loss(pred, target)
    
    @property
    def device(self) -> torch.device:
        """Get the device for this backend."""
        return self._device
    
    @property
    def backend_name(self) -> str:
        """Get the name of this backend."""
        return "pytorch"
