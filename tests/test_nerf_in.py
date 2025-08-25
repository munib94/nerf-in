"""Tests for NeRF-In implementation."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all NeRF-In components can be imported."""
    try:
        from models.inpainting.nerf_in_model import NeRFInModel
        from models.inpainting.stcn_mask_transfer import STCNMaskTransfer
        from models.inpainting.mst_inpainting import MSTInpainter
        from models.inpainting.bilateral_solver import FastBilateralSolver
        from models.losses.nerf_in_losses import NeRFInLoss
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_mask_transfer():
    """Test STCN mask transfer."""
    from models.inpainting.stcn_mask_transfer import STCNMaskTransfer
    
    mask_transfer = STCNMaskTransfer()
    
    # Create dummy data
    images = [np.random.rand(240, 320, 3) for _ in range(4)]
    user_mask = np.random.rand(240, 320) > 0.7
    
    # Test transfer
    transferred_masks = mask_transfer.transfer_mask(images, user_mask.astype(np.float32), 0)
    
    assert len(transferred_masks) == len(images)
    assert all(mask.shape == user_mask.shape for mask in transferred_masks)
    print("✅ Mask transfer test passed")

def test_inpainting():
    """Test MST/LaMa inpainting.""" 
    from models.inpainting.mst_inpainting import MSTInpainter
    
    inpainter = MSTInpainter()
    
    # Create dummy data
    image = np.random.rand(240, 320, 3)
    mask = (np.random.rand(240, 320) > 0.8).astype(np.float32)
    
    # Test inpainting
    result = inpainter.inpaint_image(image, mask)
    
    assert result.shape == image.shape
    assert 0 <= result.min() and result.max() <= 1
    print("✅ Inpainting test passed")

def test_bilateral_solver():
    """Test bilateral solver depth completion."""
    from models.inpainting.bilateral_solver import FastBilateralSolver
    
    solver = FastBilateralSolver()
    
    # Create dummy data
    target = np.random.rand(240, 320) * 5  # Depth values
    reference = np.random.rand(240, 320, 3)  # RGB image
    mask = (np.random.rand(240, 320) > 0.8).astype(np.float32)  # Holes to fill
    
    # Test completion
    result = solver.solve(target, reference, mask)
    
    assert result.shape == target.shape
    print("✅ Bilateral solver test passed")

def test_loss_computation():
    """Test NeRF-In loss functions."""
    from models.losses.nerf_in_losses import NeRFInLoss
    from config.base_config import BaseConfig
    
    config = BaseConfig()
    loss_fn = NeRFInLoss(config)
    
    # Create dummy outputs and targets
    outputs = {
        'rgb_fine': [np.random.rand(100, 3) for _ in range(4)],
        'depth_fine': [np.random.rand(100) for _ in range(4)]
    }
    
    targets = {
        'rgb_guides': [np.random.rand(100, 3) for _ in range(4)],
        'depth_guides': [np.random.rand(100) for _ in range(4)],
        'masks': [(np.random.rand(100) > 0.8).astype(np.float32) for _ in range(4)]
    }
    
    view_assignments = {'O_all': [0], 'O_out': [0, 1, 2, 3]}
    
    # Test loss computation
    losses = loss_fn.compute_nerf_in_total_loss(outputs, targets, view_assignments)
    
    assert 'total_loss' in losses
    assert 'L_color_total' in losses  
    assert 'L_depth' in losses
    print("✅ Loss computation test passed")

def test_demo_pipeline():
    """Test the demo pipeline end-to-end."""
    print("🔄 Testing demo pipeline...")
    
    # Create synthetic data
    images = [np.random.rand(240, 320, 3) for _ in range(4)]
    depths = [np.random.rand(240, 320) * 5 for _ in range(4)]
    user_mask = (np.random.rand(240, 320) > 0.8).astype(np.float32)
    
    # Test mask transfer
    from models.inpainting.stcn_mask_transfer import STCNMaskTransfer
    mask_transfer = STCNMaskTransfer()
    transferred_masks = mask_transfer.transfer_mask(images, user_mask, 0)
    
    # Test inpainting
    from models.inpainting.mst_inpainting import MSTInpainter
    inpainter = MSTInpainter()
    inpainted_images = inpainter.batch_inpaint(images, transferred_masks)
    
    # Test depth completion
    from models.inpainting.bilateral_solver import FastBilateralSolver
    solver = FastBilateralSolver()
    completed_depths = []
    for depth, mask, rgb in zip(depths, transferred_masks, inpainted_images):
        completed_depth = solver.complete_depth_image(depth, rgb, mask)
        completed_depths.append(completed_depth)
    
    assert len(completed_depths) == len(depths)
    print("✅ Demo pipeline test passed")

if __name__ == '__main__':
    print("🧪 Running NeRF-In Tests")
    print("=" * 50)
    
    test_imports()
    test_mask_transfer()
    test_inpainting() 
    test_bilateral_solver()
    test_loss_computation()
    test_demo_pipeline()
    
    print("🎉 All tests passed!")
