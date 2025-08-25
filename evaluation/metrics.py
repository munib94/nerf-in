"""Evaluation metrics for NeRF-In."""

import numpy as np
from typing import Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    Args:
        img_pred: Predicted image [H, W, C]
        img_gt: Ground truth image [H, W, C]  
        mask: Optional mask for selective computation [H, W]
        
    Returns:
        psnr_value: PSNR value in dB
    """
    if mask is not None:
        img_pred = img_pred[mask > 0.5]
        img_gt = img_gt[mask > 0.5]
    
    return psnr(img_gt, img_pred, data_range=1.0)

def compute_ssim(img_pred: np.ndarray, img_gt: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute Structural Similarity Index.
    
    Args:
        img_pred: Predicted image [H, W, C]
        img_gt: Ground truth image [H, W, C]
        mask: Optional mask for selective computation [H, W]
        
    Returns:
        ssim_value: SSIM value between 0 and 1
    """
    if len(img_pred.shape) == 3 and img_pred.shape[-1] == 3:
        # Multi-channel SSIM
        ssim_value = ssim(img_gt, img_pred, multichannel=True, data_range=1.0,
                         channel_axis=-1)
    else:
        ssim_value = ssim(img_gt, img_pred, data_range=1.0)
    
    return ssim_value

def compute_lpips(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """Compute LPIPS perceptual similarity.
    
    Args:
        img_pred: Predicted image [H, W, C]
        img_gt: Ground truth image [H, W, C]
        
    Returns:
        lpips_value: LPIPS distance (lower is better)
    """
    try:
        import lpips
        
        # Convert to torch tensors and normalize to [-1, 1]
        import torch
        img_pred_tensor = torch.from_numpy(img_pred * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)
        img_gt_tensor = torch.from_numpy(img_gt * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)
        
        # Compute LPIPS
        lpips_fn = lpips.LPIPS(net='alex')
        with torch.no_grad():
            lpips_value = lpips_fn(img_pred_tensor, img_gt_tensor)
        
        return float(lpips_value.item())
        
    except ImportError:
        print("Warning: LPIPS not available. Install with: pip install lpips")
        return 0.0

def compute_depth_metrics(depth_pred: np.ndarray, depth_gt: np.ndarray, 
                         mask: np.ndarray = None) -> Dict[str, float]:
    """Compute depth evaluation metrics.
    
    Args:
        depth_pred: Predicted depth [H, W]
        depth_gt: Ground truth depth [H, W]
        mask: Optional valid depth mask [H, W]
        
    Returns:
        metrics: Dictionary of depth metrics
    """
    if mask is not None:
        depth_pred = depth_pred[mask > 0.5]
        depth_gt = depth_gt[mask > 0.5]
    
    # Remove invalid depth values
    valid_mask = (depth_gt > 0) & (depth_pred > 0)
    depth_pred = depth_pred[valid_mask]
    depth_gt = depth_gt[valid_mask]
    
    if len(depth_pred) == 0:
        return {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0}
    
    # Absolute relative error
    abs_rel = np.mean(np.abs(depth_pred - depth_gt) / depth_gt)
    
    # Square relative error  
    sq_rel = np.mean(((depth_pred - depth_gt) ** 2) / depth_gt)
    
    # RMSE
    rmse = np.sqrt(np.mean((depth_pred - depth_gt) ** 2))
    
    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(depth_pred) - np.log(depth_gt)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel, 
        'rmse': rmse,
        'rmse_log': rmse_log
    }

def compute_metrics(img_pred: np.ndarray, img_gt: np.ndarray, 
                   depth_pred: np.ndarray = None, depth_gt: np.ndarray = None,
                   mask: np.ndarray = None) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        img_pred: Predicted image [H, W, C]
        img_gt: Ground truth image [H, W, C]
        depth_pred: Optional predicted depth [H, W]
        depth_gt: Optional ground truth depth [H, W]
        mask: Optional evaluation mask [H, W]
        
    Returns:
        metrics: Dictionary of all computed metrics
    """
    metrics = {}
    
    # Image metrics
    metrics['psnr'] = compute_psnr(img_pred, img_gt, mask)
    metrics['ssim'] = compute_ssim(img_pred, img_gt, mask)
    metrics['lpips'] = compute_lpips(img_pred, img_gt)
    
    # Depth metrics (if available)
    if depth_pred is not None and depth_gt is not None:
        depth_metrics = compute_depth_metrics(depth_pred, depth_gt, mask)
        metrics.update(depth_metrics)
    
    return metrics
