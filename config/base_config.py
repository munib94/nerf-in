"""Base configuration classes for NeRF-In."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from omegaconf import DictConfig

@dataclass
class ModelConfig:
    """Model configuration."""
    # NeRF architecture
    pos_embed_dim: int = 10
    dir_embed_dim: int = 4
    net_depth: int = 8
    net_width: int = 256
    skip_layers: List[int] = None
    
    # Inpainting specific
    use_rgbd_prior: bool = True
    depth_weight: float = 0.1
    consistency_weight: float = 0.01
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = [4]

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 1024
    learning_rate: float = 5e-4
    num_epochs: int = 200
    warmup_steps: int = 1000
    
    # Loss weights
    rgb_weight: float = 1.0
    depth_weight: float = 0.1
    perceptual_weight: float = 0.1
    consistency_weight: float = 0.01
    
    # Sampling
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    use_importance_sampling: bool = True

@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "data/sample_data"
    image_height: int = 480
    image_width: int = 640
    near_plane: float = 0.1
    far_plane: float = 10.0
    
    # Inpainting specific
    mask_ratio: float = 0.3  # Percentage of image to inpaint
    use_depth_prior: bool = True

@dataclass
class BaseConfig:
    """Base configuration combining all components."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    # System
    device: Optional[str] = None
    num_workers: int = 4
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
