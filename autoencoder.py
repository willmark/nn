"""
Simple Autoencoder for Dimensionality Reduction and Feature Learning.

This module demonstrates unsupervised learning through reconstruction, showing how
neural networks can learn compressed representations of data. The autoencoder
consists of an encoder that compresses input data into a lower-dimensional
latent space, and a decoder that reconstructs the original data from this
compressed representation.

Key Concepts:
    - Unsupervised Learning: Learning without labels through reconstruction
    - Dimensionality Reduction: Compressing high-dimensional data
    - Feature Learning: Discovering meaningful representations
    - Bottleneck Architecture: Forcing information compression
    - Reconstruction Loss: Measuring how well data is preserved

Architecture:
    - Encoder: Input → Hidden layers → Latent space (bottleneck)
    - Decoder: Latent space → Hidden layers → Reconstruction
    - Loss: Mean Squared Error between input and reconstruction

Example:
    >>> # Create autoencoder for MNIST digits
    >>> autoencoder = SimpleAutoencoder(input_dim=784, latent_dim=32)
    >>> # Train on MNIST data
    >>> # Reconstruct images from compressed representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the autoencoder model
class SimpleAutoencoder(nn.Module):
    """
    A simple autoencoder for dimensionality reduction and feature learning.
    
    This autoencoder implements a symmetric encoder-decoder architecture where
    the encoder compresses input data into a lower-dimensional latent space,
    and the decoder reconstructs the original data from this compressed
    representation. The bottleneck forces the network to learn meaningful
    compressed representations.
    
    Architecture:
        Encoder: input_dim → 128 → latent_dim
        Decoder: latent_dim → 128 → input_dim
    
    Expected input shape: [batch_size, input_dim]
    Expected output shape: [batch_size, input_dim]
    Latent representation shape: [batch_size, latent_dim]
    
    Attributes:
        encoder (nn.Sequential): Neural network that compresses input to latent space
        decoder (nn.Sequential): Neural network that reconstructs from latent space
    """
    
    def __init__(self, input_dim=784, latent_dim=32):
        """
        Initialize the autoencoder with specified dimensions.
        
        Args:
            input_dim (int): Dimension of input data (e.g., 784 for flattened MNIST).
                Default: 784
            latent_dim (int): Dimension of the latent space (bottleneck).
                Default: 32
                
        Note:
            The latent_dim should be smaller than input_dim to force compression.
            A typical ratio is 1:10 to 1:20 for meaningful compression.
        """
        super().__init__()
        # Encoder: input → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder: latent → reconstructed input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder (encode then decode).
        
        Args:
            x (torch.Tensor): Input data.
                Shape: [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Reconstructed data.
                Shape: [batch_size, input_dim]
                
        Note:
            This method performs the complete autoencoding process:
            input → encode → latent → decode → reconstruction
        """
        latent = self.encoder(x)             # compress
        reconstructed = self.decoder(latent) # decompress
        return reconstructed

# Instantiate model
model = SimpleAutoencoder(input_dim=784, latent_dim=32)

# Dummy input: batch of 4 flattened 28x28 images
x = torch.randn(4, 784)

# Forward pass
output = model(x)

# Print shapes
print("Input shape:        ", x.shape)        # [4, 784]
print("Reconstructed shape:", output.shape)   # [4, 784]

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Compute loss
loss = criterion(output, x)
print("Reconstruction loss:", loss.item())

# Backpropagation + parameter update
loss.backward()
optimizer.step()