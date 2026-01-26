"""
MFC Generative Modeling Framework
=================================

Publication-quality implementation of Generative Modeling via Mean-Field Control
using Iterative Backward Regression (Deep BSDE-style algorithm).

Modules:
    config: Configuration dataclasses for parsing YAML
    models: Neural network architectures with time embeddings
    dynamics: Euler-Maruyama SDE simulation engine
    solver: Core Iterative Backward Regression solver
    utils: Experiment management and utilities
"""

from .config import Config, PhysicsConfig, ModelConfig, TrainConfig
from .models import TimeConditionedMLP
from .dynamics import EulerMaruyama
from .solver import IterativeSolver
from .utils import ExperimentManager, GaussianProxy, KernelScoreEstimator


__all__ = [
    "Config",
    "PhysicsConfig", 
    "ModelConfig",
    "TrainConfig",
    "TimeConditionedMLP",
    "EulerMaruyama",
    "IterativeSolver",
    "ExperimentManager",
    "GaussianProxy",
    "KernelScoreEstimator",
    "SinkhornEstimator",
    "MomentMatchingEstimator",
    "MMDEstimator",
]

__version__ = "1.0.0"
__author__ = "Lizhan HONG"
