#!/usr/bin/env python3
"""Inference script for NeRF-In."""

import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import BaseConfig
from models.inpainting.inpainting_model import NeRFInModel
from data.datasets.rgbd_dataset import RGBDDataset
from utils.image_utils import save_image
from utils.logging_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Run NeRF-In inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input data directory')
    parser.add_argument('--output_path', type=str, default='results',
                       help='Path to save results')
    parser.add_argument('--image_idx', type=int, default=0,
                       help='Index of image to inpaint')
    parser.add_argument('--chunk_size', type=int, default=1024,
                       help='Ray batch size for inference')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_path / "logs")
    logger.info("Starting NeRF-In inference")
    
    # Load configuration and model
    config = BaseConfig()
    config.data.data_path = args.input_path
    
    model = NeRFInModel(config.model)
    
    # Load checkpoint (simplified - would need proper implementation)
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    # checkpoint loading logic would go here
    
    # Load test data
    dataset = RGBDDataset(
        data_path=args.input_path,
        split='test',
        image_height=config.data.image_height,
        image_width=config.data.image_width
    )
    
    if args.image_idx >= len(dataset):
        logger.error(f"Image index {args.image_idx} out of range (0-{len(dataset)-1})")
        return
    
    # Get test sample
    sample = dataset[args.image_idx]
    
    # Prepare rays for full image
    rays = {
        'ray_origins': sample['rays_o'].reshape(-1, 3),
        'ray_directions': sample['rays_d'].reshape(-1, 3),
    }
    
    if 'inpaint_mask' in sample:
        rays['inpaint_mask'] = sample['inpaint_mask'].reshape(-1)
    
    if 'depth' in sample:
        rays['depth_prior'] = sample['depth'].reshape(-1)
    
    # Run inference
    logger.info("Running inference...")
    outputs = model.render_image(rays, chunk_size=args.chunk_size)
    
    # Reshape outputs back to image dimensions
    H, W = config.data.image_height, config.data.image_width
    
    if 'rgb_fine' in outputs:
        rgb_pred = outputs['rgb_fine'].reshape(H, W, 3)
    else:
        rgb_pred = outputs['rgb_coarse'].reshape(H, W, 3)
    
    # Save results
    save_image(model.backend.to_numpy(rgb_pred), output_path / f"inpainted_{args.image_idx:03d}.png")
    
    # Save ground truth and input for comparison
    save_image(sample['image'], output_path / f"ground_truth_{args.image_idx:03d}.png")
    
    if 'inpaint_mask' in sample:
        # Save masked input
        from utils.image_utils import apply_mask
        masked_input = apply_mask(sample['image'], sample['inpaint_mask'])
        save_image(masked_input, output_path / f"masked_input_{args.image_idx:03d}.png")
        
        # Save mask
        mask_img = np.stack([sample['inpaint_mask']] * 3, axis=-1)
        save_image(mask_img, output_path / f"mask_{args.image_idx:03d}.png")
    
    logger.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
