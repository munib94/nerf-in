"""Automatic backend detection for cross-platform compatibility."""

import platform
import importlib
from typing import Optional, Any

class BackendDetector:
    """Detects and manages ML backends across platforms."""
    
    def __init__(self):
        self._backend = None
        self._device = None
        self._detect_backend()
    
    def _detect_backend(self):
        """Automatically detect the best available backend."""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            try:
                import mlx.core as mx
                self._backend = "mlx"
                self._device = mx.default_device()
                print(f"Detected macOS: Using MLX backend on {self._device}")
            except ImportError:
                self._fallback_to_cpu()
        else:  # Linux, Windows, etc.
            try:
                import torch
                self._backend = "pytorch"
                if torch.cuda.is_available():
                    self._device = torch.device('cuda')
                    print(f"Detected Linux/Windows: Using PyTorch with CUDA GPU {torch.cuda.get_device_name()}")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = torch.device('mps')
                    print("Detected macOS: Using PyTorch with MPS")
                else:
                    self._device = torch.device('cpu')
                    print("Using PyTorch with CPU")
            except ImportError:
                self._fallback_to_cpu()
    
    def _fallback_to_cpu(self):
        """Fallback to CPU-only mode."""
        print("Warning: No ML framework detected, falling back to CPU-only mode")
        self._backend = "cpu"
        self._device = "cpu"
    
    @property
    def backend(self) -> str:
        return self._backend
    
    @property
    def device(self) -> Any:
        return self._device
    
    def get_array_module(self):
        """Get the appropriate array module (torch/mlx/numpy)."""
        if self._backend == "pytorch":
            import torch
            return torch
        elif self._backend == "mlx":
            import mlx.core as mx
            return mx
        else:
            import numpy as np
            return np
    
    def get_nn_module(self):
        """Get the appropriate neural network module."""
        if self._backend == "pytorch":
            import torch.nn as nn
            return nn
        elif self._backend == "mlx":
            import mlx.nn as nn
            return nn
        else:
            raise NotImplementedError("No neural network module available for CPU fallback")

# Global backend detector instance
BACKEND = BackendDetector()
