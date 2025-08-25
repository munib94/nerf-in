#!/usr/bin/env python3
"""NeRF-In demo showcasing the complete pipeline."""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import BaseConfig
from models.inpainting.stcn_mask_transfer import STCNMaskTransfer
from models.inpainting.mst_inpainting import MSTInpainter
from models.inpainting.bilateral_solver import FastBilateralSolver
from utils.image_utils import create_inpaint_mask, save_image, apply_mask
from utils.backend_detector import BACKEND

def create_synthetic_scene():
    """Create a synthetic scene for demonstration."""
    H, W = 480, 640
    
    # Create synthetic RGB images (multiple views)
    images = []
    depths = []
    
    for i in range(8):  # 8 views
        # Gradient background with some objects
        y, x = np.mgrid[0:H, 0:W]
        
        # Background gradient
        bg_r = x / W
        bg_g = y / H  
        bg_b = 0.3 * np.ones_like(x)
        
        # Add some objects (circles)
        center_x, center_y = W//2 + 100*np.cos(i*np.pi/4), H//2 + 100*np.sin(i*np.pi/4)
        mask = ((x - center_x)**2 + (y - center_y)**2) < 50**2
        
        # Object color
        obj_r = 0.8 * np.ones_like(x)
        obj_g = 0.2 * np.ones_like(x) 
        obj_b = 0.1 * np.ones_like(x)
        
        # Combine
        r = np.where(mask, obj_r, bg_r)
        g = np.where(mask, obj_g, bg_g)
        b = np.where(mask, obj_b, bg_b)
        
        image = np.stack([r, g, b], axis=-1).astype(np.float32)
        
        # Synthetic depth
        depth = 2.0 + 0.5 * y / H
        depth = np.where(mask, 1.5, depth).astype(np.float32)
        
        images.append(image)
        depths.append(depth)
    
    return images, depths

def demonstrate_full_pipeline(output_path: Path):
    """Demonstrate the complete NeRF-In pipeline."""
    print("🚀 Demonstrating Complete NeRF-In Pipeline")
    print("=" * 50)
    
    # Create synthetic scene
    images, depths = create_synthetic_scene()
    
    # Create user mask
    H, W = images[0].shape[:2]
    user_mask = create_inpaint_mask((H, W), mask_ratio=0.15)
    
    print("📍 Step 1: STCN Mask Transfer")
    mask_transfer = STCNMaskTransfer()
    transferred_masks = mask_transfer.transfer_mask(images, user_mask, user_image_idx=0)
    print(f"   ✅ Transferred mask to {len(transferred_masks)} views")
    
    print("🎨 Step 2: MST/LaMa Image Inpainting")
    inpainter = MSTInpainter(model_type='lama')
    inpainted_images = inpainter.batch_inpaint(images, transferred_masks)
    print(f"   ✅ Inpainted {len(inpainted_images)} images")
    
    print("🔍 Step 3: Bilateral Solver Depth Completion")
    solver = FastBilateralSolver()
    completed_depths = []
    for depth, mask, rgb in zip(depths, transferred_masks, inpainted_images):
        completed_depth = solver.complete_depth_image(depth, rgb, mask)
        completed_depths.append(completed_depth)
    print(f"   ✅ Completed {len(completed_depths)} depth images")
    
    print("💾 Step 4: Saving Results")
    save_demo_results(output_path, images, user_mask, transferred_masks, 
                     inpainted_images, completed_depths)
    
    print("✅ Complete pipeline demonstration finished")
    return images, transferred_masks, inpainted_images, completed_depths

def save_demo_results(output_path: Path, images, user_mask, transferred_masks,
                     inpainted_images, completed_depths):
    """Save all demo results."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save original images
    orig_dir = output_path / "01_original"
    orig_dir.mkdir(exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, orig_dir / f"view_{i:02d}.png")
    
    # Save user mask
    save_image(np.stack([user_mask] * 3, axis=-1), 
              output_path / "02_user_mask.png")
    
    # Save transferred masks
    mask_dir = output_path / "03_transferred_masks"
    mask_dir.mkdir(exist_ok=True)
    for i, mask in enumerate(transferred_masks):
        mask_img = np.stack([mask] * 3, axis=-1)
        save_image(mask_img, mask_dir / f"mask_{i:02d}.png")
    
    # Save masked inputs
    masked_dir = output_path / "04_masked_inputs"
    masked_dir.mkdir(exist_ok=True)
    for i, (img, mask) in enumerate(zip(images, transferred_masks)):
        masked_input = apply_mask(img, mask, fill_value=0.0)
        save_image(masked_input, masked_dir / f"masked_{i:02d}.png")
    
    # Save inpainted images
    inpaint_dir = output_path / "05_inpainted"
    inpaint_dir.mkdir(exist_ok=True)
    for i, img in enumerate(inpainted_images):
        save_image(img, inpaint_dir / f"inpainted_{i:02d}.png")
    
    # Save completed depths
    depth_dir = output_path / "06_depths"
    depth_dir.mkdir(exist_ok=True)
    for i, depth in enumerate(completed_depths):
        depth_norm = depth / depth.max()
        depth_img = np.stack([depth_norm] * 3, axis=-1)
        save_image(depth_img, depth_dir / f"depth_{i:02d}.png")
    
    # Create comparison images
    comp_dir = output_path / "07_comparisons"
    comp_dir.mkdir(exist_ok=True)
    for i in range(min(4, len(images))):  # First 4 views
        # Create side-by-side comparison
        orig = images[i]
        masked = apply_mask(orig, transferred_masks[i], fill_value=0.0)
        inpainted = inpainted_images[i]
        
        # Concatenate horizontally
        comparison = np.concatenate([orig, masked, inpainted], axis=1)
        save_image(comparison, comp_dir / f"comparison_{i:02d}.png")

def main():
    parser = argparse.ArgumentParser(description='NeRF-In Complete Demo')
    parser.add_argument('--output_path', type=str, default='demo_results',
                       help='Path to save demo results')
    parser.add_argument('--component', type=str, 
                       choices=['mask', 'inpaint', 'depth', 'full'], default='full',
                       help='Which component to demonstrate')
    args = parser.parse_args()
    
    print("🎯 NeRF-In Paper-Compliant Demo")
    print("=" * 70)
    print(f"🖥️  Backend: {BACKEND.backend}")
    print(f"📁 Output: {args.output_path}")
    print(f"🔧 Component: {args.component}")
    print()
    
    output_path = Path(args.output_path)
    
    try:
        if args.component == 'full':
            demonstrate_full_pipeline(output_path)
        else:
            print(f"⚠️  Component-specific demos not implemented")
            print("   Use --component full for complete pipeline")
            return
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📂 Results saved to: {output_path}")
        print("\n📋 Next steps:")
        print("1. Install full dependencies:")
        print("   pip install -r requirements.txt")
        print("2. Train a real NeRF-In model:")
        print("   python scripts/train_nerf_in.py --data_path /path/to/data")
        print("3. Run inference with trained model:")
        print("   python scripts/infer_nerf_in.py --checkpoint model.pt --mask_path mask.png")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
