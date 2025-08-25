# NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors

Implementation of the NeRF-In paper with cross-platform backend support.

## Features

- **Cross-Platform**: Automatic PyTorch (Linux) or MLX (macOS) backend selection
- **RGB-D Integration**: Uses depth priors for improved inpainting quality
- **Free-Form Inpainting**: Support for arbitrary mask shapes and regions
- **Multi-View Consistency**: Ensures consistency across different viewpoints

## Installation

```bash
# Create virtual environment
python3.11 -m venv nerf_in_env
source nerf_in_env/bin/activate

# Install dependencies (automatically detects platform)
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Train NeRF-In model
python scripts/train.py --config config/base_config.yaml --data_path /path/to/data

# Run inference
python scripts/infer.py --checkpoint checkpoints/latest.pt --input_path /path/to/input
```

## Project Structure

- `models/backends/`: Platform-specific backend implementations
- `models/nerf/`: Core NeRF components
- `models/inpainting/`: NeRF-In specific modules
- `data/`: Dataset handling and preprocessing
- `training/`: Training loops and optimization
- `scripts/`: Command-line interfaces

## Citation

```bibtex
@article{li2022nerf,
  title={NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors},
  author={Li, Hao and Zhong, Yiwen and Wang, Ruyu and others},
  journal={arXiv preprint arXiv:2206.04901},
  year={2022}
}
```