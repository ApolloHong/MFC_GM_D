"""
Neural Network Models
=====================

Simple MLP architecture for the adjoint/control approximation Y_θ(t, x).
Direct concatenation of [t, x] without sinusoidal embeddings for simplicity.

Author: Lizhan HONG
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

from .config import ModelConfig, PhysicsConfig


class SimpleMLP(nn.Module):
    """
    Simple MLP for approximating Y_θ(t, x).
    
    Architecture:
    1. Concatenate [t, x] directly → input_dim = state_dim + 1
    2. 3 hidden layers with Tanh activation (smooth for control laws)
    3. Output: Y ∈ R^dim (adjoint/control variable)
    
    The network learns: Y_θ_k(x) ≈ E[Y_{k+1} | X_k = x]
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        physics_config: PhysicsConfig,
    ):
        """
        Initialize simple MLP.
        
        Args:
            model_config: Network architecture parameters
            physics_config: Physics parameters (for state dimension)
        """
        super().__init__()
        
        self.state_dim = physics_config.dim
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers
        
        # Input: [t, x] concatenated → dim = state_dim + 1
        input_dim = physics_config.dim + 1
        
        # Build MLP layers
        layers = []
        
        for i in range(model_config.n_layers):
            layers.append(nn.Linear(input_dim, model_config.hidden_dim))
            layers.append(nn.Tanh())  # Tanh for smooth control laws
            input_dim = model_config.hidden_dim
        
        # Output layer (no activation - unbounded output for control)
        layers.append(nn.Linear(model_config.hidden_dim, physics_config.dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable gradients."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_θ(t, x).
        
        Args:
            t: Time tensor of shape (batch_size,) or scalar
            x: State tensor of shape (batch_size, dim) or (batch_size,)
            
        Returns:
            y: Output tensor of shape (batch_size, dim)
        """
        # Ensure proper shapes
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.size(0))  # scalar → (batch_size,)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size,) → (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (batch_size,) → (batch_size, 1)
        
        # Concatenate [t, x]
        tx = torch.cat([t, x], dim=-1)  # (batch_size, 1 + dim)
        
        return self.mlp(tx)


# Alias for backward compatibility
TimeConditionedMLP = SimpleMLP


def create_networks(
    model_config: ModelConfig,
    physics_config: PhysicsConfig,
    device: torch.device,
) -> List[SimpleMLP]:
    """
    Create N independent networks (one per timestep).
    
    Args:
        model_config: Network architecture parameters
        physics_config: Physics parameters
        device: Torch device (cpu/cuda)
        
    Returns:
        List of N SimpleMLP networks
    """
    networks = []
    for k in range(physics_config.N):
        net = SimpleMLP(model_config, physics_config).to(device)
        networks.append(net)
    return networks
