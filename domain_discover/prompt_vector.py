"""Learnable prompt vector for domain discovery."""

import torch
import torch.nn as nn
from typing import Optional


class PromptVector(nn.Module):
    """Learnable prompt vector for steering text generation.
    
    This module wraps a single learnable embedding vector that will be
    optimized to guide the diffusion model toward generating text with
    desired properties.
    
    Args:
        hidden_dim: Dimension of the embedding vector (matches model)
        init_scale: Scale for random initialization
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        init_scale: float = 0.02
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initialize learnable vector with small random values
        self.embedding = nn.Parameter(
            torch.randn(1, hidden_dim) * init_scale
        )
    
    def forward(self) -> torch.Tensor:
        """Return the current prompt vector.
        
        Returns:
            Tensor of shape (1, hidden_dim)
        """
        return self.embedding
    
    def clone_detached(self) -> torch.Tensor:
        """Return a detached copy of the prompt vector.
        
        Returns:
            Tensor of shape (1, hidden_dim), detached from computation graph
        """
        return self.embedding.detach().clone()


if __name__ == "__main__":
    # Simple unit test
    prompt = PromptVector(hidden_dim=768)
    vec = prompt()
    print(f"Vector shape: {vec.shape}")
    print(f"Requires grad: {vec.requires_grad}")
    
    # Test cloning
    vec_clone = prompt.clone_detached()
    print(f"Clone requires grad: {vec_clone.requires_grad}")
