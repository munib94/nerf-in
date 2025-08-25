#!/usr/bin/env python3
"""Demo script for NeRF-In."""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import BaseConfig
from models.inpainting.inpainting_model import NeRFInModel
from utils.image_utils import create_inpaint_mask, save_image
from utils.backend_detector import BACKEND

def create_demo_data():
    """Create synthetic demo data."""
    config = BaseConfig()
    
    # Create a simple synthetic scene
    H, W = config.data.image_height, config.data.image_width
    
    # Synthetic RGB image (simple gradient)
    y, x = np.mgrid[0:H, 0:W]
    image = np.stack([
        x / W,  # Red channel
        y / H,  # Green channel  
        0.5 * np.ones_like(x)  # Blue channel
    ], axis=-1).astype(np.float32)
    
    # Synthetic depth (linear depth)
    depth = 1.0 + 2.0 * y / H
    
    # Create inpainting mask
    mask = create_inpaint_mask((H, W), mask_ratio=0.2)
    
    return image, depth, mask

def main():
    parser = argparse.ArgumentParser(description='NeRF-In Demo')
    parser.add_argument('--output_path', type=str, default='demo_results',
                       help='Path to save demo results')
    args = parser.parse_args()
    
    print(f"Running NeRF-In demo with {BACKEND.backend} backend")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create demo data
    print("Creating synthetic demo data...")
    image, depth, mask = create_demo_data()
    
    # Save input data
    save_image(image, output_path / "original.png")
    
    # Apply mask to create input
    from utils.image_utils import apply_mask
    masked_image = apply_mask(image, mask, fill_value=0.0)
    save_image(masked_image, output_path / "masked_input.png")
    
    # Save mask
    mask_img = np.stack([mask] * 3, axis=-1)
    save_image(mask_img, output_path / "mask.png")
    
    # Initialize model
    print("Initializing NeRF-In model...")
    config = BaseConfig()
    model = NeRFInModel(config.model)
    
    print("Demo setup complete!")
    print(f"Results saved to: {output_path}")
    print("\nTo train a real model, use:")
    print("python scripts/train.py --data_path /path/to/your/data")
    print("\nTo run inference on trained model, use:")
    print("python scripts/infer.py --checkpoint /path/to/checkpoint.pt --input_path /path/to/data")

if __name__ == '__main__':
    main()
