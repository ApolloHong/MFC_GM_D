"""
Configuration Module
====================

Dataclasses for parsing YAML configuration with strict typing.
Provides type-safe access to all hyperparameters.

Author: Lizhan HONG
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import numpy as np
from pathlib import Path


@dataclass
class PhysicsConfig:
    """
    Physics/SDE parameters for the Mean-Field Control problem.
    
    Attributes:
        T: Terminal time horizon
        N: Number of discrete time steps
        sigma: Diffusion coefficient
        dim: State space dimension
    """
    T: float = 1.0
    N: int = 50
    sigma: float = 1.0
    dim: int = 1
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.dt = self.T / self.N
        self.sqrt_dt = np.sqrt(self.dt)
        self.time_grid = np.linspace(0, self.T, self.N + 1)


@dataclass
class TargetConfig:
    """
    Target distribution parameters (Gaussian).
    
    Attributes:
        mean: Target mean μ
        std: Target standard deviation σ
    """
    mean: float = 1.0
    std: float = 1.0
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.var = self.std ** 2


@dataclass
class InitialConfig:
    """
    Initial distribution parameters.
    
    Attributes:
        type: Distribution type ("dirac" or "gaussian")
        mean: Initial mean
        std: Initial standard deviation (0 for dirac)
    """
    type: str = "dirac"
    mean: float = 0.0
    std: float = 0.0


@dataclass
class ModelConfig:
    """
    Neural network architecture parameters.
    
    Attributes:
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        time_embedding_dim: Dimension of sinusoidal time embedding
        activation: Activation function name
    """
    hidden_dim: int = 64
    n_layers: int = 3
    time_embedding_dim: int = 16
    activation: str = "leaky_relu"


@dataclass
class TrainConfig:
    """
    Training hyperparameters for Iterative Backward Regression.
    
    MFC Problem:
        min_α E[ Σ_{k=0}^{N-1} (1/2)|α_k|² Δt + λ * KL(P_{X_N} || μ_T) ]
    
    Attributes:
        iterations: Number of coupling iterations (outer loop)
        backward_epochs: Epochs for each backward step (inner loop)
        batch_size: Number of particles for Monte Carlo estimation
        learning_rate: Adam optimizer learning rate
        grad_clip: Gradient norm clipping threshold
        print_every: Print frequency (in iterations)
        terminal_weight: λ coefficient for KL divergence term
    """
    iterations: int = 20
    backward_epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 0.001
    grad_clip: float = 1.0
    print_every: int = 5
    terminal_weight: float = 10.0  # λ: weight for KL(P_{X_N} || μ_T)
    y_clamp: float = 100.0  # Max value for terminal gradient


@dataclass
class ExperimentConfig:
    """
    Experiment and logging settings.
    
    Attributes:
        name: Experiment name prefix
        seed: Random seed for reproducibility
        save_plots: Whether to save visualization plots
        n_plot_trajectories: Number of trajectories to plot
        n_eval_samples: Number of samples for evaluation
    """
    name: str = "mfc_backward_regression"
    seed: int = 42
    save_plots: bool = True
    n_plot_trajectories: int = 100
    n_eval_samples: int = 5000


@dataclass
class Config:
    """
    Top-level configuration container.
    
    Aggregates all sub-configurations and provides
    factory method to load from YAML file.
    """
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    initial: InitialConfig = field(default_factory=InitialConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config instance with all parameters loaded
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse each section
        physics = PhysicsConfig(**data.get('physics', {}))
        target = TargetConfig(**data.get('target', {}))
        initial = InitialConfig(**data.get('initial', {}))
        model = ModelConfig(**data.get('model', {}))
        training = TrainConfig(**data.get('training', {}))
        experiment = ExperimentConfig(**data.get('experiment', {}))
        
        return cls(
            physics=physics,
            target=target,
            initial=initial,
            model=model,
            training=training,
            experiment=experiment
        )
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = [
            "=" * 70,
            "MFC Configuration (Iterative Backward Regression)",
            "=" * 70,
            f"  Physics:    T={self.physics.T}, N={self.physics.N}, "
            f"σ={self.physics.sigma}, dim={self.physics.dim}",
            f"  Target:     N({self.target.mean}, {self.target.var})",
            f"  Initial:    {self.initial.type} at {self.initial.mean}",
            "-" * 70,
            f"  Model:      hidden={self.model.hidden_dim}, "
            f"layers={self.model.n_layers}, time_emb={self.model.time_embedding_dim}",
            f"  Training:   {self.training.iterations} iterations × "
            f"{self.training.backward_epochs} epochs, "
            f"batch={self.training.batch_size}, lr={self.training.learning_rate}",
            f"  λ (terminal_weight): {self.training.terminal_weight}",
            "=" * 70,
        ]
        return "\n".join(lines)
