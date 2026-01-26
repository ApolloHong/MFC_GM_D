"""
===============================================================================
Generative Modeling via Mean-Field Control (Discrete Time)
===============================================================================

Entry point for the MFC generative modeling framework using
Iterative Backward Regression (Deep BSDE-style algorithm).

Algorithm:
1. Initialize N neural networks {Y_θ_0, ..., Y_θ_{N-1}}
2. Coupling Loop (iterations):
   a. Forward Pass: Simulate X_0 → X_N using current controls α_k = -Y_θ_k
   b. Terminal Condition: Y_N = GaussianProxy(X_N)
   c. Backward Induction: Train Y_θ_k to regress E[Y_{k+1} | X_k]

Mathematical Framework:
-----------------------
We solve the Mean-Field Control problem:

    min_α E[ Σ_{k=0}^{N-1} (1/2)|α_k|² Δt + λ * KL(P_{X_N} || μ_T) ]

where λ = terminal_weight is the coefficient for the KL divergence term.

subject to:
    X_{k+1} = X_k + α_k Δt + σ √Δt Z_k

The optimal control is α* = -Y where Y is the adjoint variable.
We approximate Y using N networks trained via backward regression.

Author: Lizhan HONG (lizhan.hong@polytechnique.edu)
Date: 2024
===============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
from pathlib import Path

# Import from the src package
from src import (
    Config,
    IterativeSolver,
    ExperimentManager,
)

warnings.filterwarnings('ignore')


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str = "config.yaml", config_obj: Config = None, show_plots: bool = True):
    """
    Main function to run the MFC generative modeling experiment.
    
    Problem Setup:
    - Initial: μ_0 = δ_0 (Dirac at zero)
    - Target: μ_T = N(μ, σ²) (configurable Gaussian)
    - Goal: Learn optimal control via Iterative Backward Regression
    
    Args:
        config_path: Path to YAML configuration file
        config_obj: Optional Config object (overrides config_path)
        show_plots: Whether to call plt.show()
    """
    print("\n" + "=" * 70)
    print("  GENERATIVE MODELING VIA MEAN-FIELD CONTROL (DISCRETE TIME)")
    print("  Algorithm: Iterative Backward Regression")
    print("=" * 70)
    
    # =========================================================================
    # Load Configuration
    # =========================================================================
    print("\n[1] Loading configuration...")
    
    if config_obj is not None:
        config = config_obj
        print(f"Using provided config object: {config.experiment.name}")
    else:
        config = Config.from_yaml(config_path)
        print(config)
    
    # Set seeds
    set_seeds(config.experiment.seed)
    
    # =========================================================================
    # Initialize Experiment Manager
    # =========================================================================
    print("\n[2] Initializing experiment manager...")
    
    exp_manager = ExperimentManager(config, config_path)
    
    # =========================================================================
    # Initialize Solver
    # =========================================================================
    print("\n[3] Initializing iterative solver...")
    
    solver = IterativeSolver(config, exp_manager)
    
    print(f"    Time horizon: T = {config.physics.T}")
    print(f"    Time steps: N = {config.physics.N}")
    print(f"    Diffusion: σ = {config.physics.sigma}")
    print(f"    Initial: {config.initial.type} at {config.initial.mean}")
    print(f"    Target: N({config.target.mean}, {config.target.var})")
    
    # =========================================================================
    # Run Training
    # =========================================================================
    print("\n[4] Running iterative backward regression...")
    
    final_stats = solver.run()
    
    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n[5] Generating visualizations...")
    
    if config.experiment.save_plots:
        figures = solver.visualize(save=True)
        
        # Also show plots
        if show_plots:
            plt.show()
    
    # =========================================================================
    # Finalize
    # =========================================================================
    exp_manager.finalize(final_stats)
    
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print(f"  Results saved to: {exp_manager.exp_dir}")
    print("=" * 70 + "\n")
    
    return solver, final_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MFC Generative Modeling via Iterative Backward Regression"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    solver, stats = main(args.config)