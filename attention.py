"""
Scaled Dot-Product Attention Mechanism Demonstration.

This module demonstrates the implementation of the scaled dot-product attention mechanism
as introduced in "Attention Is All You Need" (Vaswani et al., 2017). It shows how to
compute attention weights and apply them to values using query, key, and value matrices.

The attention mechanism allows the model to focus on different parts of the input
sequence when processing each position, enabling better handling of long-range
dependencies and parallel processing.

Key Components:
    - Query (Q): Represents what we're looking for
    - Key (K): Represents what information is available
    - Value (V): Represents the actual information content
    - Attention Weights: Softmax of scaled dot-product of Q and K
    - Output: Weighted sum of values based on attention weights

Example:
    >>> # Input embeddings: [batch_size=1, seq_len=3, embed_dim=4]
    >>> x = torch.rand(1, 3, 4)
    >>> # Apply attention mechanism
    >>> attention_weights, output = scaled_dot_product_attention(x)
"""

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(x):
    """
    Compute scaled dot-product attention for input embeddings.
    
    This function implements the core attention mechanism by:
    1. Computing Query, Key, and Value matrices from input embeddings
    2. Computing attention scores using scaled dot-product
    3. Applying softmax to get attention weights
    4. Computing weighted sum of values
    
    Args:
        x (torch.Tensor): Input embeddings.
            Shape: [batch_size, seq_len, embed_dim]
    
    Returns:
        tuple: A tuple containing:
            - attention_weights (torch.Tensor): Attention weights matrix
                Shape: [batch_size, seq_len, seq_len]
            - output (torch.Tensor): Attention output
                Shape: [batch_size, seq_len, embed_dim]
    
    Note:
        This is a simplified implementation where Q, K, V are computed from the same
        input embeddings. In practice, these are often computed from different
        transformations of the input.
    """
    # Linear layers to compute Q, K, V from input embeddings
    W_q = torch.nn.Linear(x.size(-1), x.size(-1), bias=False)
    W_k = torch.nn.Linear(x.size(-1), x.size(-1), bias=False)
    W_v = torch.nn.Linear(x.size(-1), x.size(-1), bias=False)
    
    # Compute Query, Key, and Value matrices
    Q = W_q(x)  # [batch_size, seq_len, embed_dim]
    K = W_k(x)  # [batch_size, seq_len, embed_dim]
    V = W_v(x)  # [batch_size, seq_len, embed_dim]
    
    # Scaled dot-product attention
    dk = Q.size(-1)  # Dimension of key vectors
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk**0.5)  # [batch_size, seq_len, seq_len]
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, V)  # [batch_size, seq_len, embed_dim]
    
    return attention_weights, output


# Sample dimensions for demonstration
batch_size = 1
seq_len = 3
embed_dim = 4

# Simulated input embeddings
x = torch.rand(batch_size, seq_len, embed_dim)  # (1, 3, 4)

print("Input Embeddings Shape:", x.shape)
print("Input Embeddings:\n", x)

# Compute attention
attention_weights, output = scaled_dot_product_attention(x)

print("\nAttention Weights Shape:", attention_weights.shape)
print("Attention Weights:\n", attention_weights)
print("\nAttention Output Shape:", output.shape)
print("Attention Output:\n", output)