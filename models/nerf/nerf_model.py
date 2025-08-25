"""Core NeRF model implementation."""

import numpy as np
from typing import Tuple, Optional
from models.backends import create_backend
from models.nerf.positional_encoding import get_positional_encoder
from config.base_config import ModelConfig

class NeRFModel:
    """Neural Radiance Fields model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.backend = create_backend()
        
        # Create positional encoders
        self.pos_encoder, self.pos_embed_dim = get_positional_encoder(3, config.pos_embed_dim)
        self.dir_encoder, self.dir_embed_dim = get_positional_encoder(3, config.dir_embed_dim)
        
        # Create networks
        self._build_networks()
    
    def _build_networks(self):
        """Build the NeRF networks."""
        # Density network (position -> density + features)
        density_hidden = [self.config.net_width] * self.config.net_depth
        self.density_net = self.backend.create_mlp(
            input_dim=self.pos_embed_dim,
            hidden_dims=density_hidden,
            output_dim=self.config.net_width + 1,  # features + density
            activation="relu",
            skip_connections=self.config.skip_layers
        )
        
        # Color network (features + direction -> RGB)
        self.color_net = self.backend.create_mlp(
            input_dim=self.config.net_width + self.dir_embed_dim,
            hidden_dims=[self.config.net_width // 2],
            output_dim=3,  # RGB
            activation="relu"
        )
    
    def forward(self, positions, directions):
        """Forward pass through NeRF.
        
        Args:
            positions: [..., 3] coordinates
            directions: [..., 3] view directions
            
        Returns:
            rgb: [..., 3] RGB colors
            density: [..., 1] volume densities
        """
        # Encode positions and directions
        pos_encoded = self.pos_encoder.encode(positions)
        dir_encoded = self.dir_encoder.encode(directions)
        
        # Get density and features from position
        density_features = self.density_net(pos_encoded)
        density = density_features[..., :1]  # First channel is density
        features = density_features[..., 1:]  # Rest are features
        
        # Get RGB from features and direction
        color_input = self.backend.concatenate([features, dir_encoded], axis=-1)
        rgb = self.color_net(color_input)
        
        # Apply activations
        rgb = self.backend.sigmoid(rgb) if hasattr(self.backend, 'sigmoid') else self._sigmoid(rgb)
        density = self.backend.relu(density) if hasattr(self.backend, 'relu') else self._relu(density)
        
        return rgb, density
    
    def _sigmoid(self, x):
        """Sigmoid activation fallback."""
        # Using numerical stable sigmoid: 1 / (1 + exp(-x))
        return 1.0 / (1.0 + self.backend.exp(-x))
    
    def _relu(self, x):
        """ReLU activation fallback."""
        zero = self.backend.zeros(x.shape)
        return self.backend.maximum(x, zero) if hasattr(self.backend, 'maximum') else x * (x > 0)
