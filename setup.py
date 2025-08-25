from setuptools import setup, find_packages
import platform

# Platform-specific requirements
requirements = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "imageio>=2.31.0",
    "imageio-ffmpeg>=0.4.8",
    "tqdm>=4.65.0",
    "tensorboard>=2.13.0",
    "trimesh>=3.21.0",
    "open3d>=0.17.0",
    "scipy>=1.11.0",
    "scikit-image>=0.21.0",
    "omegaconf>=2.3.0",
    "wandb>=0.15.0",
    "hydra-core>=1.3.0",
    "lpips>=0.1.4",
    "kornia>=0.7.0",
]

# Add platform-specific ML frameworks
if platform.system() == "Darwin":  # macOS
    requirements.extend(["mlx>=0.28.0", "mlx-nn>=0.1.0"])
else:  # Linux and others
    requirements.extend(["torch>=2.1.0", "torchvision>=0.16.0"])

setup(
    name="nerf-in",
    version="0.1.0",
    description="NeRF-In: Free-Form NeRF Inpainting with RGB-D Priors",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "nerf-in-train=scripts.train:main",
            "nerf-in-infer=scripts.infer:main",
        ],
    },
)
