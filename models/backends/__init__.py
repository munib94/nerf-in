"""Backend abstraction layer for cross-platform compatibility."""

from .base_backend import BaseBackend
from .pytorch_backend import PyTorchBackend
from .mlx_backend import MLXBackend
from utils.backend_detector import BACKEND

def create_backend() -> BaseBackend:
    """Create the appropriate backend for the current platform."""
    if BACKEND.backend == "pytorch":
        return PyTorchBackend()
    elif BACKEND.backend == "mlx":
        return MLXBackend()
    else:
        raise NotImplementedError(f"Backend {BACKEND.backend} not supported")

__all__ = ['BaseBackend', 'PyTorchBackend', 'MLXBackend', 'create_backend']
