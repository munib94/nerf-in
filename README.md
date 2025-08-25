# NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors

**Paper-Compliant Implementation** of "NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors"

## 🎯 Overview

This is a **corrected and complete implementation** of NeRF-In following the paper methodology exactly:

- ✅ **STCN Mask Transfer**: Video object segmentation for mask propagation
- ✅ **MST/LaMa Inpainting**: RGB guidance generation
- ✅ **Bilateral Solver**: Depth completion with RGB-D priors  
- ✅ **Proper Loss Formulation**: L_color + L_depth as specified in paper
- ✅ **Two-Stage Training**: Pre-training (200K) + Inpainting (50K) steps

## 🚀 Key Features

### Paper-Compliant Components
- **Cross-Platform**: Automatic PyTorch (Linux) or MLX (macOS) backend
- **STCN Integration**: Mask transfer using video object segmentation
- **MST/LaMa Inpainting**: High-quality RGB guidance generation
- **Bilateral Solver**: Depth completion with color guidance
- **Multi-View Consistency**: Ensures consistency across viewpoints

### Training Methodology
- **Stage 1**: Standard NeRF pre-training (200,000 steps)
- **Stage 2**: NeRF-In optimization with RGB-D guidance (50,000 steps)
- **Loss Functions**: Exact implementation of paper equations

## 📦 Installation

```bash
# Clone and setup
git clone <repository>
cd nerf-in

# Create virtual environment  
python3.11 -m venv nerf_in_env
source nerf_in_env/bin/activate

# Install dependencies (auto-detects platform)
pip install -r requirements.txt
pip install -e .
```

## 🎬 Quick Demo

```bash
# Run complete pipeline demo
python scripts/demo_nerf_in.py --output_path demo_results

# Test individual components
python tests/test_nerf_in.py
```

## 🏃‍♂️ Training

### Full Pipeline (Recommended)
```bash
# Train complete NeRF-In model (both stages)
python scripts/train_nerf_in.py \
    --data_path /path/to/rgbd_data \
    --mask_path /path/to/user_mask.png \
    --user_view_idx 0
```

## 📊 Paper Compliance

### Algorithm Implementation

| Paper Component | Implementation | Status |
|----------------|----------------|---------|
| STCN Mask Transfer (Eq. 5 region) | ✅ `STCNMaskTransfer` | Complete |
| MST Inpainting I^G_s = ρ(I_s, M_s) | ✅ `MSTInpainter` | Complete |
| Bilateral Solver D^G_s = τ(D_s, M_s, I^G_s) | ✅ `FastBilateralSolver` | Complete |
| Color Loss L_color = L_all + L_out | ✅ `NeRFInLoss.compute_color_guiding_loss` | Complete |
| Depth Loss L_depth (Eq. 11) | ✅ `NeRFInLoss.compute_depth_guiding_loss` | Complete |
| Two-stage Training | ✅ `NeRFInTrainer` | Complete |
| View Sampling (K=24) | ✅ `sample_camera_trajectory` | Complete |

## 🆘 Troubleshooting

### Common Issues

**"STCN model not found"**
- Download STCN weights from [official repository](https://github.com/hkchengrex/STCN)
- Place in `models/weights/stcn_model.pth`

**"LaMa inpainting failed"**
- Install: `pip install lama-cleaner`
- Or use OpenCV fallback (automatically enabled)

**"CUDA out of memory"**
- Reduce `chunk_size` in inference scripts
- Lower `batch_size` in training config

---

**This implementation provides a complete, paper-compliant NeRF-In system ready for research and application use.** 🎯
