"""Image processing utilities."""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional

def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        target_size: Optional target size (height, width)
        
    Returns:
        image: [H, W, 3] RGB image array in range [0, 1]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize if needed
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    return image_array

def load_depth(depth_path: str, target_size: Optional[Tuple[int, int]] = None,
               depth_scale: float = 1000.0) -> np.ndarray:
    """Load and preprocess a depth image.
    
    Args:
        depth_path: Path to depth image file
        target_size: Optional target size (height, width)
        depth_scale: Scale factor to convert depth to meters
        
    Returns:
        depth: [H, W] depth array in meters
    """
    # Load depth image (typically 16-bit)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    # Resize if needed
    if target_size is not None:
        depth = cv2.resize(depth, (target_size[1], target_size[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    # Convert to meters
    depth = depth.astype(np.float32) / depth_scale
    
    return depth

def save_image(image: np.ndarray, save_path: str):
    """Save an image array to file.
    
    Args:
        image: [H, W, 3] image array in range [0, 1]
        save_path: Path to save the image
    """
    # Convert to 8-bit
    image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(image_8bit).save(save_path)

def create_inpaint_mask(image_shape: Tuple[int, int], mask_ratio: float = 0.3) -> np.ndarray:
    """Create a random inpainting mask.
    
    Args:
        image_shape: (height, width) of the image
        mask_ratio: Fraction of pixels to mask
        
    Returns:
        mask: [H, W] binary mask (1 = inpaint region, 0 = known region)
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Create random rectangular regions to inpaint
    num_regions = np.random.randint(1, 4)  # 1-3 rectangular regions
    
    for _ in range(num_regions):
        # Random rectangle size and position
        rect_h = np.random.randint(height // 8, height // 3)
        rect_w = np.random.randint(width // 8, width // 3)
        
        start_h = np.random.randint(0, height - rect_h)
        start_w = np.random.randint(0, width - rect_w)
        
        mask[start_h:start_h + rect_h, start_w:start_w + rect_w] = 1.0
    
    # Adjust mask to match target ratio
    current_ratio = mask.mean()
    if current_ratio < mask_ratio:
        # Add more random pixels
        remaining_pixels = int((mask_ratio - current_ratio) * height * width)
        zero_positions = np.where(mask == 0)
        if len(zero_positions[0]) > 0:
            indices = np.random.choice(len(zero_positions[0]), 
                                     min(remaining_pixels, len(zero_positions[0])), 
                                     replace=False)
            mask[zero_positions[0][indices], zero_positions[1][indices]] = 1.0
    
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Apply inpainting mask to image.
    
    Args:
        image: [H, W, C] input image
        mask: [H, W] binary mask (1 = mask out, 0 = keep)
        fill_value: Value to fill masked regions
        
    Returns:
        masked_image: Image with masked regions filled
    """
    masked_image = image.copy()
    masked_image[mask > 0.5] = fill_value
    return masked_image
