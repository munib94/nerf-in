"""Data loading utilities."""

import numpy as np
from typing import Iterator, Dict, Any
from data.datasets.rgbd_dataset import RGBDDataset
from config.base_config import DataConfig, TrainingConfig

class DataLoader:
    """Simple data loader for NeRF-In training."""
    
    def __init__(self, dataset: RGBDDataset, batch_size: int = 1024, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over ray batches."""
        # For NeRF training, we typically sample rays randomly from all images
        total_rays_per_epoch = len(self.dataset) * 1000  # Approximate rays per epoch
        
        for _ in range(0, total_rays_per_epoch, self.batch_size):
            # Sample a random image
            if self.shuffle:
                img_idx = np.random.randint(0, len(self.dataset))
            else:
                img_idx = np.random.randint(0, len(self.dataset))  # Still random for now
            
            # Get ray batch from that image
            ray_batch = self.dataset.get_rays_batch(self.batch_size, img_idx)
            
            yield ray_batch
    
    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        return (len(self.dataset) * 1000) // self.batch_size

def create_data_loader(data_config: DataConfig, training_config: TrainingConfig, 
                      split: str = 'train') -> DataLoader:
    """Create a data loader for training or validation.
    
    Args:
        data_config: Data configuration
        training_config: Training configuration  
        split: Data split ('train', 'val', 'test')
        
    Returns:
        data_loader: Configured data loader
    """
    dataset = RGBDDataset(
        data_path=data_config.data_path,
        split=split,
        image_height=data_config.image_height,
        image_width=data_config.image_width,
        near_plane=data_config.near_plane,
        far_plane=data_config.far_plane
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.batch_size,
        shuffle=(split == 'train')
    )
    
    return data_loader
