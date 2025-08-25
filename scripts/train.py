#!/usr/bin/env python3
"""Training script for NeRF-In."""

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import BaseConfig
from training.trainer import NeRFInTrainer
from data.loaders.data_utils import create_data_loader
from utils.logging_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train NeRF-In model')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config_dict = OmegaConf.load(args.config)
        config = OmegaConf.structured(BaseConfig(**config_dict))
    else:
        config = BaseConfig()
    
    # Override config with command line arguments
    config.data.data_path = args.data_path
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    
    # Setup logging
    logger = setup_logging(config.log_dir)
    logger.info("Starting NeRF-In training")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    # Create data loaders
    train_loader = create_data_loader(config.data, config.training, split='train')
    val_loader = create_data_loader(config.data, config.training, split='val')
    
    # Create trainer
    trainer = NeRFInTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    logger.info("Training completed")

if __name__ == '__main__':
    main()
