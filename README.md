# NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors

**Paper-Compliant Implementation** of "NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors"

## 🎯 Overview

This is an **complete and unofficial implementation** of NeRF-In following the paper methodology exactly:

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

## 📁 Codebase Structure

```
nerf-in/
├── 📁 config/                    # Configuration management
│   ├── __init__.py
│   ├── base_config.py           # Core configuration classes
│   └── nerf_in_config.yaml      # Default hyperparameters
│
├── 📁 data/                     # Data handling & preprocessing
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── rgbd_dataset.py      # RGB-D dataset loader
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── data_utils.py        # Data loading utilities
│   └── sample_data/             # Sample training data
│       ├── depth/               # Depth maps
│       ├── images/              # RGB images
│       ├── masks/               # Inpainting masks
│       └── poses/               # Camera poses
│
├── 📁 models/                   # Core neural network models
│   ├── backends/                # Cross-platform backends
│   │   ├── __init__.py
│   │   ├── base_backend.py      # Backend interface
│   │   ├── mlx_backend.py       # Apple MLX backend
│   │   └── pytorch_backend.py   # PyTorch backend
│   │
│   ├── inpainting/              # NeRF-In specific components
│   │   ├── __init__.py
│   │   ├── bilateral_solver.py  # Fast bilateral solver for depth
│   │   ├── inpainting_model.py  # Core inpainting model
│   │   ├── mst_inpainting.py    # MST/LaMa inpainting
│   │   ├── nerf_in_model.py     # Main NeRF-In model
│   │   └── stcn_mask_transfer.py # STCN mask propagation
│   │
│   ├── losses/                  # Loss functions
│   │   ├── __init__.py
│   │   ├── consistency_loss.py  # Multi-view consistency
│   │   ├── nerf_in_losses.py    # Paper-specific losses
│   │   └── rendering_loss.py    # Standard rendering losses
│   │
│   ├── nerf/                    # Standard NeRF components
│   │   ├── __init__.py
│   │   ├── nerf_model.py        # Core NeRF network
│   │   ├── positional_encoding.py # Positional encoding
│   │   ├── ray_sampling.py      # Ray sampling strategies
│   │   └── volume_rendering.py  # Volume rendering
│   │
│   ├── networks/                # Additional neural networks
│   │   └── __init__.py
│   └── weights/                 # Pre-trained model weights
│
├── 📁 scripts/                  # Training & inference scripts
│   ├── __init__.py
│   ├── demo.py                  # Basic demo
│   ├── demo_nerf_in.py         # Full pipeline demo
│   ├── infer.py                # Inference script
│   └── train.py                # Main training script
│
├── 📁 training/                 # Training infrastructure
│   ├── __init__.py
│   └── trainer.py              # NeRF-In trainer class
│
├── 📁 utils/                    # Utility functions
│   ├── __init__.py
│   ├── backend_detector.py     # Automatic backend detection
│   ├── camera_utils.py         # Camera & ray utilities
│   ├── image_utils.py          # Image processing
│   └── logging_utils.py        # Logging setup
│
├── 📁 evaluation/               # Evaluation & metrics
│   ├── __init__.py
│   └── metrics.py              # PSNR, SSIM, LPIPS metrics
│
├── 📁 tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_nerf_in.py         # Main integration tests
│   ├── test_data/              # Test data
│   ├── test_models/            # Model tests
│   └── test_utils/             # Utility tests
│
├── 📁 checkpoints/              # Saved model checkpoints
├── 📁 logs/                     # Training logs
├── 📁 results/                  # Inference results
└── 📁 notebooks/                # Jupyter notebooks for analysis
```

### Key Entry Points
- **Training**: `scripts/train.py` - Main training script
- **Demo**: `scripts/demo_nerf_in.py` - Complete pipeline demonstration  
- **Inference**: `scripts/infer.py` - Model inference
- **Core Model**: `models/inpainting/nerf_in_model.py` - Main NeRF-In implementation
- **Trainer**: `training/trainer.py` - Training loop and optimization

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

## Citation

```bibtex
@article{li2022nerf,
  title={NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors},
  author={Li, Hao and Zhong, Yiwen and Wang, Ruyu and others},
  journal={arXiv preprint arXiv:2206.04901},
  year={2022}
}
```