"""
===============================================================================
Compare Terminal Distribution for Different λ Values
===============================================================================
This script uses the current src architecture to compare terminal distributions
for different terminal_weight (λ) values using the ScoreMatchingEstimator.

λ values: 1.0, 5.0, 10.0, 20.0

Author: Lizhan HONG
Date: 01/2026
===============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import Config, IterativeSolver, ExperimentManager

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

def main():
    """Compare terminal distributions for different λ values."""
    
    print("\n" + "=" * 70)
    print("  COMPARING TERMINAL DISTRIBUTIONS FOR DIFFERENT λ VALUES")
    print("  λ ∈ {1.0, 5.0, 10.0, 20.0}")
    print("=" * 70)
    
    # λ values to compare (focus on meaningful range)
    lambda_values = [1.0, 5.0]
    
    # Load default config
    base_config = Config.from_yaml("config.yaml")
    
    # Modify for faster comparison
    base_config.training.iterations = 15  # Sufficient for convergence
    base_config.training.backward_epochs = 100
    base_config.training.print_every = 5
    base_config.experiment.save_plots = False  # We'll plot manually
    
    n_samples = 5000
    
    # Store results
    results = {}
    
    # Train models for each λ
    for lam in lambda_values:
        print(f"\n{'='*70}")
        print(f"  Training with λ = {lam}")
        print(f"{'='*70}")
        
        # Create specific config (deep copy to avoid side effects)
        config = copy.deepcopy(base_config)
        config.training.terminal_weight = lam
        config.experiment.name = f"compare_lambda_{lam}"
        
        # Initialize solver
        # We don't need full experiment manager for this script
        solver = IterativeSolver(config, exp_manager=None)
        
        # Run training
        stats = solver.run()
        
        # Generate trajectories for evaluation
        with torch.no_grad():
            trajectories, _ = solver.dynamics.simulate(
                solver.target_networks,
                n_samples,
                return_controls=False
            )
        
        trajectories_np = trajectories.cpu().numpy()
        # Squeeze if needed: (N+1, batch, 1) -> (N+1, batch)
        if trajectories_np.ndim == 3:
            trajectories_np = trajectories_np.squeeze(-1)
            
        x_terminal = trajectories_np[-1, :]
        
        mean = np.mean(x_terminal)
        var = np.var(x_terminal)
        
        results[lam] = {
            'trajectories': trajectories_np,
            'samples': x_terminal,
            'mean': mean,
            'var': var,
            'time_grid': solver.dynamics.time_grid.cpu().numpy()
        }
        print(f">>> Result: Mean = {mean:.4f}, Var = {var:.4f}")

    # ==========================================================================
    # Figure 1: Generated Trajectories (2x2 subplots)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  GENERATING FIGURE 1: TRAJECTORIES COMPARISON")
    print("=" * 70)
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    axes1 = axes1.flatten()
    
    n_plot_trajectories = 100
    
    for i, lam in enumerate(lambda_values):
        ax = axes1[i]
        traj = results[lam]['trajectories']
        time_grid = results[lam]['time_grid']
        mean_val = results[lam]['mean']
        var_val = results[lam]['var']
        
        # Compute statistics
        means = np.mean(traj, axis=1)
        stds = np.std(traj, axis=1)
        
        # Plot sample trajectories
        for j in range(min(n_plot_trajectories, traj.shape[1])):
            ax.plot(time_grid, traj[:, j], alpha=0.15, color='blue', linewidth=0.5)
        
        # Plot mean trajectory
        ax.plot(time_grid, means, 'r-', linewidth=2.5, label='Mean trajectory')
        
        # Plot ± 2σ band
        ax.fill_between(time_grid, means - 2*stds, means + 2*stds,
                        alpha=0.25, color='red', label='±2σ band')
        
        # Target mean line
        ax.axhline(y=base_config.target.mean, color='green', linestyle='--', 
                   linewidth=2, label=f'Target μ={base_config.target.mean}')
        
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('$X_t$', fontsize=11)
        ax.set_title(f'λ = {lam:g}  (μ={mean_val:.2f}, σ²={var_val:.2f})', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(-3, 5) # Auto-scale is usually better
    
    fig1.suptitle(
        r"Generated Trajectories for Different $\lambda$ (DSM Estimator)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('lambda_trajectories_comparison.png', dpi=150, bbox_inches='tight')
    print("Figure 1 saved to 'lambda_trajectories_comparison.png'")
    
    # ==========================================================================
    # Figure 2: Terminal Distribution vs Target (2x2 subplots)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  GENERATING FIGURE 2: TERMINAL DISTRIBUTION COMPARISON")
    print("=" * 70)
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    
    # Target PDF
    target_mean = base_config.target.mean
    target_var = base_config.target.var
    target_std = base_config.target.std
    
    x_range = np.linspace(target_mean - 4*target_std, target_mean + 4*target_std, 300)
    pdf_target = (
        1 / np.sqrt(2 * np.pi * target_var) * 
        np.exp(-0.5 * (x_range - target_mean) ** 2 / target_var)
    )
    
    for i, lam in enumerate(lambda_values):
        ax = axes2[i]
        samples = results[lam]['samples']
        mean_val = results[lam]['mean']
        var_val = results[lam]['var']
        
        # Histogram
        ax.hist(samples, bins=50, density=True, alpha=0.6, 
                color='blue', edgecolor='black', linewidth=0.5,
                label=f'Generated $X_N$')
        
        # Target PDF
        ax.plot(x_range, pdf_target, 'r-', linewidth=2.5, 
                label=f'Target $\\mathcal{{N}}({target_mean}, {target_var})$')
        
        ax.set_xlabel('$X_N$', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'λ = {lam:g}  (μ={mean_val:.2f}, σ²={var_val:.2f})', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig2.suptitle(
        r"Terminal Distribution vs Target for Different $\lambda$ (DSM Estimator)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('lambda_distributions_comparison.png', dpi=150, bbox_inches='tight')
    print("Figure 2 saved to 'lambda_distributions_comparison.png'")
    
    # ==========================================================================
    # Summary table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (DSM-based Estimator)")
    print("=" * 70)
    print(f"{'λ':^10} | {'Mean':^10} | {'Variance':^10} | {'Target Mean':^12} | {'Target Var':^10}")
    print("-" * 65)
    for lam in lambda_values:
        mean = results[lam]['mean']
        var = results[lam]['var']
        print(f"{lam:^10g} | {mean:^10.4f} | {var:^10.4f} | {target_mean:^12.4f} | {target_var:^10.4f}")
    print("=" * 70)
    print("\nAll figures saved!")

if __name__ == "__main__":
    main()
