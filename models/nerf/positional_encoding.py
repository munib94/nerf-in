"""Positional encoding for NeRF."""

import numpy as np
from models.backends import create_backend

class PositionalEncoder:
    """Positional encoding as used in NeRF."""
    
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.backend = create_backend()
        
        # Create frequency bands
        self.freq_bands = []
        for i in range(num_freqs):
            freq = 2.0 ** i
            self.freq_bands.append(freq)
    
    def encode(self, x):
        """Apply positional encoding to input coordinates."""
        encoded = []
        
        if self.include_input:
            encoded.append(x)
        
        # Apply sin/cos encoding at different frequencies
        for freq in self.freq_bands:
            encoded.append(self.backend.sin(x * freq))
            encoded.append(self.backend.cos(x * freq))
        
        return self.backend.concatenate(encoded, axis=-1)
    
    @property
    def output_dim(self):
        """Get the output dimension after encoding."""
        base_dim = 1 if self.include_input else 0
        return base_dim + 2 * self.num_freqs

def get_positional_encoder(input_dim: int, num_freqs: int = 10):
    """Create positional encoder for given input dimension."""
    encoder = PositionalEncoder(num_freqs=num_freqs)
    output_dim = input_dim * encoder.output_dim
    return encoder, output_dim
