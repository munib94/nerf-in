"""MLX backend implementation for Apple Silicon."""

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx import nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from typing import Any, Tuple, Optional
import numpy as np
from .base_backend import BaseBackend

if MLX_AVAILABLE:
    class MLXMLP(nn.Module):
        """MLX MLP implementation with skip connections."""
        
        def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                     activation: str = "relu", skip_connections: list = None):
            super().__init__()
            
            self.input_dim = input_dim
            self.skip_connections = skip_connections or []
            
            # Build layers
            self.layers = []
            dims = [input_dim] + hidden_dims + [output_dim]
            
            for i in range(len(dims) - 1):
                # Add skip connection input if needed
                in_dim = dims[i]
                if i in self.skip_connections:
                    in_dim += input_dim
                
                layer = nn.Linear(in_dim, dims[i + 1])
                self.layers.append(layer)
        
        def __call__(self, x):
            input_x = x
            
            for i, layer in enumerate(self.layers[:-1]):  # All but last layer
                # Add skip connection if needed
                if i in self.skip_connections:
                    x = mx.concatenate([x, input_x], axis=-1)
                
                x = layer(x)
                x = mx.maximum(x, 0)  # ReLU activation
            
            # Final layer (no activation)
            if len(self.layers) > 0:
                if len(self.layers) - 1 in self.skip_connections:
                    x = mx.concatenate([x, input_x], axis=-1)
                x = self.layers[-1](x)
            
            return x

class MLXBackend(BaseBackend):
    """MLX backend implementation for Apple Silicon."""
    
    def __init__(self):
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available. Please install MLX for Apple Silicon support.")
        self._device = mx.default_device()
    
    def create_tensor(self, data: np.ndarray, requires_grad: bool = False) -> mx.array:
        """Create a tensor from numpy array."""
        return mx.array(data.astype(np.float32))
    
    def to_numpy(self, tensor: mx.array) -> np.ndarray:
        """Convert tensor to numpy array."""
        return np.array(tensor)
    
    def zeros(self, shape: Tuple[int, ...], requires_grad: bool = False) -> mx.array:
        """Create zero tensor with given shape."""
        return mx.zeros(shape, dtype=mx.float32)
    
    def ones(self, shape: Tuple[int, ...], requires_grad: bool = False) -> mx.array:
        """Create ones tensor with given shape."""
        return mx.ones(shape, dtype=mx.float32)
    
    def random(self, shape: Tuple[int, ...], requires_grad: bool = False) -> mx.array:
        """Create random tensor with given shape."""
        return mx.random.normal(shape, dtype=mx.float32)
    
    def matmul(self, a: mx.array, b: mx.array) -> mx.array:
        """Matrix multiplication."""
        return mx.matmul(a, b)
    
    def sin(self, x: mx.array) -> mx.array:
        """Sine function."""
        return mx.sin(x)
    
    def cos(self, x: mx.array) -> mx.array:
        """Cosine function."""
        return mx.cos(x)
    
    def concatenate(self, tensors: list, axis: int = 0) -> mx.array:
        """Concatenate tensors along given axis."""
        return mx.concatenate(tensors, axis=axis)
    
    def reshape(self, tensor: mx.array, shape: Tuple[int, ...]) -> mx.array:
        """Reshape tensor."""
        return mx.reshape(tensor, shape)
    
    def create_mlp(self, input_dim: int, hidden_dims: list, output_dim: int,
                   activation: str = "relu", skip_connections: list = None) -> 'MLXMLP':
        """Create MLP network."""
        return MLXMLP(input_dim, hidden_dims, output_dim, activation, skip_connections)
    
    def mse_loss(self, pred: mx.array, target: mx.array) -> mx.array:
        """Mean squared error loss."""
        return mx.mean(mx.square(pred - target))
    
    def l1_loss(self, pred: mx.array, target: mx.array) -> mx.array:
        """L1 loss."""
        return mx.mean(mx.abs(pred - target))
    
    @property
    def device(self) -> Any:
        """Get the device for this backend."""
        return self._device
    
    @property
    def backend_name(self) -> str:
        """Get the name of this backend."""
        return "mlx"

else:
    # Fallback implementation when MLX is not available
    class MLXBackend(BaseBackend):
        def __init__(self):
            raise ImportError("MLX not available. Please install MLX for Apple Silicon support.")
        
        def create_tensor(self, *args, **kwargs):
            raise NotImplementedError
        
        def to_numpy(self, *args, **kwargs):
            raise NotImplementedError
        
        def zeros(self, *args, **kwargs):
            raise NotImplementedError
        
        def ones(self, *args, **kwargs):
            raise NotImplementedError
        
        def random(self, *args, **kwargs):
            raise NotImplementedError
        
        def matmul(self, *args, **kwargs):
            raise NotImplementedError
        
        def sin(self, *args, **kwargs):
            raise NotImplementedError
        
        def cos(self, *args, **kwargs):
            raise NotImplementedError
        
        def concatenate(self, *args, **kwargs):
            raise NotImplementedError
        
        def reshape(self, *args, **kwargs):
            raise NotImplementedError
        
        def create_mlp(self, *args, **kwargs):
            raise NotImplementedError
        
        def mse_loss(self, *args, **kwargs):
            raise NotImplementedError
        
        def l1_loss(self, *args, **kwargs):
            raise NotImplementedError
        
        @property
        def device(self):
            raise NotImplementedError
        
        @property
        def backend_name(self):
            return "mlx_unavailable"
