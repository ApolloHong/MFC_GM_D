"""
===============================================================================
Compare Terminal Distribution for Different λ Values
===============================================================================
This script imports functions from main.py and compares terminal distributions
for different terminal_weight (λ) values.

λ values: 0.1, 1.0, 10.0, 100.0

Author: Lizhan HONG(lizhan.hong@polytechnique.edu)
Date: 01/2026
===============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import from main.py
from main import DriftNetwork, MFCSimulator, train_mfc, device

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def main():
    """Compare terminal distributions for different λ values."""
    
    print("\n" + "=" * 70)
    print("  COMPARING TERMINAL DISTRIBUTIONS FOR DIFFERENT λ VALUES")
    print("  λ ∈ {0.1, 1.0, 10.0, 100.0}")
    print("=" * 70)
    
    # λ values to compare
    lambda_values = [0.1, 1.0, 10.0, 100.0]
    
    # Fixed parameters
    T = 1.0
    N = 50
    sigma = 1.0
    target_mean = 1.0
    target_std = 1.0
    state_dim = 1
    n_epochs = 1000
    batch_size = 1024
    learning_rate = 1e-3
    n_samples = 5000
    
    # Store results
    results = {}
    
    # Train models for each λ
    for lam in lambda_values:
        print(f"\n{'='*70}")
        print(f"  Training with λ = {lam}")
        print(f"{'='*70}")
        
        # Reset seeds for fair comparison
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create model and simulator
        drift_net = DriftNetwork(state_dim=state_dim, hidden_dims=[64, 64, 64]).to(device)
        simulator = MFCSimulator(
            T=T, N=N, sigma=sigma,
            target_mean=target_mean, target_std=target_std,
            state_dim=state_dim, terminal_weight=lam
        )
        
        # Train
        history = train_mfc(
            drift_net=drift_net,
            simulator=simulator,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            print_every=200,
        )
        
        # Generate trajectories for visualization
        drift_net.eval()
        with torch.no_grad():
            trajectories, _ = simulator.simulate_trajectories(drift_net, n_samples)
            trajectories_np = trajectories.cpu().numpy().squeeze(-1)  # (N+1, n_samples)
            x_terminal = trajectories_np[-1, :]
        
        mean = np.mean(x_terminal)
        var = np.var(x_terminal)
        
        # Store all results including full trajectories
        results[lam] = {
            'trajectories': trajectories_np,
            'samples': x_terminal,
            'mean': mean,
            'var': var,
            'time_grid': simulator.time_grid.cpu().numpy()
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
    
    n_plot_trajectories = 100  # Number of trajectories to plot
    
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
        ax.axhline(y=target_mean, color='green', linestyle='--', 
                   linewidth=2, label=f'Target μ={target_mean}')
        
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('$X_t$', fontsize=11)
        ax.set_title(f'λ = {lam:g}  (μ={mean_val:.2f}, σ²={var_val:.2f})', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T)
        ax.set_ylim(-3, 4)
    
    fig1.suptitle(
        r"Generated Trajectories for Different $\lambda$ Values "
        r"$J(\alpha) = \mathbb{E}\!\left[\frac{1}{2}\,\Sigma\,|\alpha|^2\,\Delta t\right]"
        r" + \lambda\,\mathrm{KL}(P_{X_N}\|\mu_T)$",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('lambda_trajectories_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
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
    x_range = np.linspace(-3, 5, 300)
    pdf_target = (
        1 / np.sqrt(2 * np.pi * target_std**2) * 
        np.exp(-0.5 * (x_range - target_mean) ** 2 / target_std**2)
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
                label=f'Target $\\mathcal{{N}}({target_mean}, {target_std**2})$')
        
        ax.set_xlabel('$X_N$', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'λ = {lam:g}  (μ={mean_val:.2f}, σ²={var_val:.2f})', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 5)
    
    fig2.suptitle(
    r"Terminal Distribution vs Target for Different $\lambda$ Values "
    r"$J(\alpha) = \mathbb{E}\!\left[\frac{1}{2}\,\Sigma\,|\alpha|^2\,\Delta t\right]"
    r" + \lambda\,\mathrm{KL}(P_{X_N}\|\mu_T)$",
    fontsize=14,
    fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('lambda_distributions_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 2 saved to 'lambda_distributions_comparison.png'")
    
    # ==========================================================================
    # Summary table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"{'λ':^10} | {'Mean':^10} | {'Variance':^10} | {'Target Mean':^12} | {'Target Var':^10}")
    print("-" * 65)
    for lam in lambda_values:
        mean = results[lam]['mean']
        var = results[lam]['var']
        print(f"{lam:^10g} | {mean:^10.4f} | {var:^10.4f} | {1.0:^12.4f} | {1.0:^10.4f}")
    print("=" * 70)
    print("\nAll figures saved!")


if __name__ == "__main__":
    main()
