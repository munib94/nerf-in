"""Abstract base class for ML backends."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np

class BaseBackend(ABC):
    """Abstract base class for ML framework backends."""
    
    @abstractmethod
    def create_tensor(self, data: np.ndarray, requires_grad: bool = False) -> Any:
        """Create a tensor from numpy array."""
        pass
    
    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        pass
    
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], requires_grad: bool = False) -> Any:
        """Create zero tensor with given shape."""
        pass
    
    @abstractmethod
    def ones(self, shape: Tuple[int, ...], requires_grad: bool = False) -> Any:
        """Create ones tensor with given shape."""
        pass
    
    @abstractmethod
    def random(self, shape: Tuple[int, ...], requires_grad: bool = False) -> Any:
        """Create random tensor with given shape."""
        pass
    
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        pass
    
    @abstractmethod
    def sin(self, x: Any) -> Any:
        """Sine function."""
        pass
    
    @abstractmethod
    def cos(self, x: Any) -> Any:
        """Cosine function."""
        pass
    
    @abstractmethod
    def concatenate(self, tensors: list, axis: int = 0) -> Any:
        """Concatenate tensors along given axis."""
        pass
    
    @abstractmethod
    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape tensor."""
        pass
    
    @abstractmethod
    def create_mlp(self, input_dim: int, hidden_dims: list, output_dim: int, 
                   activation: str = "relu", skip_connections: list = None) -> Any:
        """Create MLP network."""
        pass
    
    @abstractmethod
    def mse_loss(self, pred: Any, target: Any) -> Any:
        """Mean squared error loss."""
        pass
    
    @abstractmethod
    def l1_loss(self, pred: Any, target: Any) -> Any:
        """L1 loss."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> Any:
        """Get the device for this backend."""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get the name of this backend."""
        pass
