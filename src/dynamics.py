"""
Dynamics Module
===============

Euler-Maruyama simulation engine for SDE dynamics.

The forward dynamics are:
    X_{k+1} = X_k + α_k Δt + σ √Δt Z_k

where:
    - X_k is the state at timestep k
    - α_k = -Y_θ_k(X_k) is the control (from adjoint networks)
    - σ is the diffusion coefficient
    - Z_k ~ N(0, I) is Gaussian noise

Author: Lizhan HONG
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .config import PhysicsConfig, InitialConfig


class EulerMaruyama:
    """
    Euler-Maruyama integrator for controlled SDE.
    
    Simulates the forward dynamics:
        dX_t = α(t, X_t) dt + σ dW_t
    
    using the discrete scheme:
        X_{k+1} = X_k + α_k Δt + σ √Δt Z_k
    
    Attributes:
        N: Number of time steps
        dt: Time step size
        sqrt_dt: Square root of time step
        sigma: Diffusion coefficient
        dim: State dimension
        device: Torch device
    """
    
    def __init__(
        self,
        physics_config: PhysicsConfig,
        initial_config: InitialConfig,
        device: torch.device,
    ):
        """
        Initialize Euler-Maruyama integrator.
        
        Args:
            physics_config: Physics/SDE parameters
            initial_config: Initial distribution parameters
            device: Torch device
        """
        self.N = physics_config.N
        self.T = physics_config.T
        self.dt = physics_config.dt
        self.sqrt_dt = physics_config.sqrt_dt
        self.sigma = physics_config.sigma
        self.dim = physics_config.dim
        self.device = device
        
        # Initial distribution
        self.initial_type = initial_config.type
        self.initial_mean = initial_config.mean
        self.initial_std = initial_config.std
        
        # Time grid as tensor
        self.time_grid = torch.tensor(
            physics_config.time_grid, 
            dtype=torch.float32, 
            device=device
        )
    
    def sample_initial(self, batch_size: int) -> torch.Tensor:
        """
        Sample initial conditions X_0 ~ μ_0.
        
        Args:
            batch_size: Number of particles to sample
            
        Returns:
            x0: Initial states of shape (batch_size, dim)
        """
        if self.initial_type == "dirac":
            # Dirac mass at initial_mean
            x0 = torch.full(
                (batch_size, self.dim),
                self.initial_mean,
                dtype=torch.float32,
                device=self.device
            )
        else:
            # Gaussian initial distribution
            x0 = torch.randn(batch_size, self.dim, device=self.device)
            x0 = x0 * self.initial_std + self.initial_mean
        
        return x0
    
    def step(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one Euler-Maruyama step.
        
        X_{k+1} = X_k + α_k Δt + σ √Δt Z_k
        
        Args:
            x: Current state (batch_size, dim)
            alpha: Control/drift (batch_size, dim)
            noise: Optional pre-sampled noise (batch_size, dim)
            
        Returns:
            x_next: Next state (batch_size, dim)
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        x_next = x + alpha * self.dt + self.sigma * self.sqrt_dt * noise
        return x_next
    
    def simulate(
        self,
        networks: List[nn.Module],
        batch_size: int,
        return_controls: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Simulate full trajectories X_0 -> X_N using current networks.
        
        The control at step k is: α_k = -Y_θ_k(X_k)
        (negative because Y represents the adjoint/costate)
        
        Args:
            networks: List of N networks [Y_θ_0, ..., Y_θ_{N-1}]
            batch_size: Number of particles
            return_controls: Whether to return control values
            
        Returns:
            trajectories: Tensor of shape (N+1, batch_size, dim)
            controls: Tensor of shape (N, batch_size, dim) if return_controls
        """
        # Initialize storage
        trajectories = torch.zeros(
            self.N + 1, batch_size, self.dim, 
            device=self.device
        )
        
        if return_controls:
            controls = torch.zeros(
                self.N, batch_size, self.dim,
                device=self.device
            )
        else:
            controls = None
        
        # Sample initial condition
        x = self.sample_initial(batch_size)
        trajectories[0] = x
        
        # Forward simulation
        for k in range(self.N):
            t_k = self.time_grid[k]
            
            # Get control: α_k = -Y_θ_k(X_k)
            # Y represents gradient of terminal cost, so α = -Y to minimize cost
            with torch.no_grad():
                y_k = networks[k](t_k, x)
            alpha_k = -y_k  # Negative of Y to push towards lower cost
            
            if return_controls:
                controls[k] = alpha_k
            
            # Euler-Maruyama step
            x = self.step(x, alpha_k)
            trajectories[k + 1] = x
        
        return trajectories, controls
    
    def simulate_with_gradients(
        self,
        networks: List[nn.Module],
        batch_size: int,
        fixed_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate trajectories WITH gradient tracking for training.
        
        This version stores all intermediate values needed for
        the backward regression training.
        
        Args:
            networks: List of N networks
            batch_size: Number of particles
            fixed_noise: Optional fixed noise tensor (N, batch_size, dim)
            
        Returns:
            trajectories: (N+1, batch_size, dim)
            controls: (N, batch_size, dim) - with gradients
            noise: (N, batch_size, dim) - the noise used
        """
        # Pre-sample noise if not provided (for reproducibility)
        if fixed_noise is None:
            noise = torch.randn(
                self.N, batch_size, self.dim,
                device=self.device
            )
        else:
            noise = fixed_noise
        
        # Storage
        trajectories = torch.zeros(
            self.N + 1, batch_size, self.dim,
            device=self.device
        )
        controls = torch.zeros(
            self.N, batch_size, self.dim,
            device=self.device
        )
        
        # Initial condition
        x = self.sample_initial(batch_size)
        trajectories[0] = x.clone()
        
        # Forward pass with gradient tracking
        for k in range(self.N):
            t_k = self.time_grid[k]
            
            # Get Y value (WITH gradients for the k-th network)
            y_k = networks[k](t_k, x)
            alpha_k = -y_k  # Negative of Y to push towards lower cost
            controls[k] = alpha_k
            
            # Euler-Maruyama step
            x = x + alpha_k * self.dt + self.sigma * self.sqrt_dt * noise[k]
            trajectories[k + 1] = x.clone()
        
        return trajectories, controls, noise
