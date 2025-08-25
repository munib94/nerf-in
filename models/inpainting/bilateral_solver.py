"""Fast Bilateral Solver for depth completion."""

import numpy as np
import cv2
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import Optional

class FastBilateralSolver:
    """Fast Bilateral Solver for depth image completion."""
    
    def __init__(self, sigma_spatial: float = 8.0, sigma_luma: float = 8.0, 
                 sigma_chroma: float = 8.0):
        """Initialize bilateral solver.
        
        Args:
            sigma_spatial: Spatial standard deviation
            sigma_luma: Luma standard deviation  
            sigma_chroma: Chroma standard deviation
        """
        self.sigma_spatial = sigma_spatial
        self.sigma_luma = sigma_luma
        self.sigma_chroma = sigma_chroma
    
    def solve(self, target: np.ndarray, reference: np.ndarray, 
              mask: np.ndarray, confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve bilateral optimization for depth completion.
        
        Args:
            target: Target depth image to complete [H, W]
            reference: Reference RGB image [H, W, 3] 
            mask: Binary mask [H, W] where 1 = regions to complete
            confidence: Optional confidence map [H, W]
            
        Returns:
            completed_depth: Completed depth image [H, W]
        """
        H, W = target.shape
        
        # Convert reference to YUV for bilateral weights
        reference_yuv = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        reference_yuv = reference_yuv.astype(np.float32) / 255.0
        
        # Create bilateral affinity matrix
        A = self._build_bilateral_matrix(reference_yuv, H, W)
        
        # Set up linear system: (A + λI)x = λt
        # where λ controls data fitting vs smoothness
        if confidence is None:
            confidence = np.ones((H, W), dtype=np.float32)
        
        # Higher confidence for known (non-masked) regions
        lambda_vals = confidence.copy()
        lambda_vals[mask > 0.5] = 0.01  # Low confidence for masked regions
        lambda_vals[mask <= 0.5] = 1.0  # High confidence for known regions
        
        # Flatten arrays
        target_flat = target.flatten()
        lambda_flat = lambda_vals.flatten()
        
        # Create diagonal confidence matrix
        C = diags(lambda_flat, format='csc')
        
        # Solve (A + C)x = C*target
        system_matrix = A + C
        rhs = C.dot(target_flat)
        
        try:
            solution = spsolve(system_matrix, rhs)
            completed_depth = solution.reshape(H, W)
        except:
            # Fallback to simple interpolation if solver fails
            print("⚠️  Bilateral solver failed, using interpolation fallback")
            completed_depth = self._interpolation_fallback(target, mask)
        
        return completed_depth
    
    def _build_bilateral_matrix(self, reference: np.ndarray, H: int, W: int) -> csc_matrix:
        """Build bilateral affinity matrix."""
        # Simplified bilateral matrix construction
        # In practice, this should implement the full bilateral solver matrix
        
        # Create spatial coordinates
        y, x = np.mgrid[0:H, 0:W]
        coords = np.stack([x.flatten(), y.flatten()], axis=1)
        
        # Simple spatial Laplacian as approximation
        # This is a simplified version - full implementation would include
        # bilateral weights based on color similarity
        
        num_pixels = H * W
        
        # Build simple spatial connectivity
        rows, cols, data = [], [], []
        
        for i in range(H):
            for j in range(W):
                pixel_idx = i * W + j
                
                # Add self-connection (regularization)
                rows.append(pixel_idx)
                cols.append(pixel_idx)
                data.append(4.0)  # Degree of connectivity
                
                # Add connections to neighbors
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_idx = ni * W + nj
                        
                        # Compute bilateral weight (simplified)
                        spatial_dist = np.sqrt((i - ni)**2 + (j - nj)**2)
                        color_dist = np.linalg.norm(reference[i, j] - reference[ni, nj])
                        
                        weight = np.exp(-spatial_dist**2 / (2 * self.sigma_spatial**2)) *                                 np.exp(-color_dist**2 / (2 * self.sigma_luma**2))
                        
                        rows.append(pixel_idx)
                        cols.append(neighbor_idx)
                        data.append(-weight)
        
        A = csc_matrix((data, (rows, cols)), shape=(num_pixels, num_pixels))
        return A
    
    def _interpolation_fallback(self, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback interpolation method."""
        # Convert to uint8 for OpenCV
        target_uint8 = (np.clip(target, 0, 10) * 25.5).astype(np.uint8)  # Scale depth
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Use OpenCV inpainting for depth completion
        completed = cv2.inpaint(target_uint8, mask_uint8, 3, cv2.INPAINT_NS)
        
        # Convert back to depth values
        return completed.astype(np.float32) / 25.5
    
    def complete_depth_image(self, depth: np.ndarray, rgb: np.ndarray, 
                           mask: np.ndarray) -> np.ndarray:
        """Complete depth image using bilateral solver.
        
        Args:
            depth: Input depth image [H, W] with holes
            rgb: Reference RGB image [H, W, 3]
            mask: Mask indicating holes [H, W]
            
        Returns:
            completed_depth: Completed depth image [H, W]
        """
        return self.solve(depth, rgb, mask)
