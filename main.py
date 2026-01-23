"""
===============================================================================
Generative Modeling via Mean-Field Control (MFC) in Discrete Time
===============================================================================

This script implements a discrete-time generative model based on Mean-Field 
Control theory. The goal is to transport an initial distribution μ₀ to a 
target distribution μ_T with minimum control energy.

Mathematical Framework:
-----------------------
We consider the discrete-time stochastic dynamics (Euler-Maruyama scheme):

    X_{k+1} = X_k + α_θ(t_k, X_k) Δt + σ √Δt Z_k

where:
    - X_k is the state at time step k
    - α_θ(t, x) is the control (drift) parameterized by a neural network θ
    - Δt = T/N is the time step
    - σ is the diffusion coefficient
    - Z_k ~ N(0, I) is standard Gaussian noise

The optimal control problem is to minimize:

    J(α) = E[ Σ_{k=0}^{N-1} (1/2)|α(t_k, X_k)|² Δt + λ * KL(P_{X_N} || μ_T) ]

where:
    - (1/2)|α|² Δt is the running cost (control energy)
    - g(x) = -log p_target(x) is the terminal cost (KL divergence proxy)

Connection to Maximum Principle:
-------------------------------------------------
The Hamiltonian for this problem is:
    H(t, x, p, α) = p · α + (1/2)|α|²

Minimizing H over α gives the optimal control:
    α*(t, x) = - Y_{t_k}

where Y_{t_k} is the adjoint (costate) variable satisfying a backward equation.
The neural network α_θ learns to approximate this optimal control.

Author: Lizhan HONG(lizhan.hong@polytechnique.edu)
Date: 2024
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# Model Definition
# =============================================================================

class DriftNetwork(nn.Module):
    """
    Neural network to parameterize the optimal control α_θ(t, x).
    
    Architecture:
    - Input: (t, x) ∈ R^{1+d} where d is the state dimension
    - Output: α ∈ R^d (the drift/control)
    
    The network learns to approximate the optimal control α* = -p where p
    is the adjoint variable from the Pontryagin Maximum Principle.
    
    We use a simple MLP with:
    - Time embedding through concatenation
    - LeakyReLU activations for training stability
    - No output activation (unbounded control)
    """
    
    def __init__(self, state_dim: int = 1, hidden_dims: list = [64, 64, 64]):
        """
        Args:
            state_dim: Dimension of state space (d)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        
        # Build MLP layers
        layers = []
        input_dim = state_dim + 1  # (t, x)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (important for stable training)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable gradients."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the control α_θ(t, x).
        
        Args:
            t: Time tensor of shape (batch_size, 1) or (batch_size,)
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            alpha: Control tensor of shape (batch_size, state_dim)
        """
        # Ensure proper shapes
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        
        # Concatenate time and state: (t, x)
        tx = torch.cat([t, x], dim=-1)
        
        return self.network(tx)


# =============================================================================
# MFC Simulation and Loss
# =============================================================================

class MFCSimulator:
    """
    Mean-Field Control simulator for discrete-time dynamics.
    
    This class handles:
    1. Forward simulation of SDE using Euler-Maruyama
    2. Computation of running cost (control energy)
    3. Computation of terminal cost: λ * KL(P_{X_N} || μ_T)
    
    The total cost to minimize:
        J(α) = E[ Σ_{k=0}^{N-1} (1/2)|α_k|² Δt ] + λ * KL(P_{X_N} || μ_T)
    
    where:
        - Running cost: control energy (integrated over time)
        - Terminal cost: KL divergence between generated and target distributions
        - λ (terminal_weight): weighting factor for the KL penalty
    """
    
    def __init__(
        self,
        T: float = 1.0,
        N: int = 50,
        sigma: float = 1.0,
        target_mean: float = 1.0,
        target_std: float = 1.0,
        state_dim: int = 1,
        terminal_weight: float = 10.0,
    ):
        """
        Args:
            T: Terminal time
            N: Number of time steps
            sigma: Diffusion coefficient
            target_mean: Mean of target Gaussian
            target_std: Standard deviation of target Gaussian
            state_dim: Dimension of state space
            terminal_weight: Weight λ for terminal KL divergence penalty
        """
        self.T = T
        self.N = N
        self.dt = T / N
        self.sigma = sigma
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_var = target_std ** 2
        self.state_dim = state_dim
        self.terminal_weight = terminal_weight  # λ: KL penalty weight
        
        # Time grid: t_k = k * dt for k = 0, 1, ..., N
        self.time_grid = torch.linspace(0, T, N + 1).to(device)
    
    def simulate_trajectories(
        self,
        drift_net: DriftNetwork,
        batch_size: int,
        x0: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate trajectories using Euler-Maruyama scheme:
            X_{k+1} = X_k + α_θ(t_k, X_k) Δt + σ √Δt Z_k
        
        Args:
            drift_net: Neural network for drift α_θ
            batch_size: Number of particles/trajectories
            x0: Initial state (default: Dirac at 0)
            
        Returns:
            trajectories: Tensor of shape (N+1, batch_size, state_dim)
            controls: Tensor of shape (N, batch_size, state_dim)
        """
        # Initialize trajectories storage
        trajectories = torch.zeros(
            self.N + 1, batch_size, self.state_dim, device=device
        )
        controls = torch.zeros(
            self.N, batch_size, self.state_dim, device=device
        )
        
        # Initial condition: X_0 ~ μ_0 (Dirac at 0)
        if x0 is None:
            x0 = torch.zeros(batch_size, self.state_dim, device=device)
        trajectories[0] = x0.clone()
        
        # Current state
        x = x0.clone()
        
        # Forward simulation using Euler-Maruyama
        for k in range(self.N):
            t_k = self.time_grid[k]
            
            # Get control: α_θ(t_k, X_k)
            t_batch = t_k.expand(batch_size, 1)  # (batch_size, 1)
            alpha_k = drift_net(t_batch, x)  # (batch_size, state_dim)
            controls[k] = alpha_k
            
            # Sample noise: Z_k ~ N(0, I)
            z_k = torch.randn(batch_size, self.state_dim, device=device)
            
            # Euler-Maruyama update:
            # X_{k+1} = X_k + α_k Δt + σ √Δt Z_k
            x = x + alpha_k * self.dt + self.sigma * np.sqrt(self.dt) * z_k
            
            # Store trajectory
            trajectories[k + 1] = x
        
        return trajectories, controls
    
    def compute_running_cost(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute the running cost (control energy):
            L = Σ_{k=0}^{N-1} (1/2)|α_k|² Δt
        
        This represents the Lagrangian integrated over time.
        From PMP: L(x, α) = (1/2)|α|² is the running cost in the Hamiltonian.
        
        Args:
            controls: Tensor of shape (N, batch_size, state_dim)
            
        Returns:
            running_cost: Scalar tensor (mean over batch)
        """
        # |α_k|² for each time step
        control_squared = torch.sum(controls ** 2, dim=-1)  # (N, batch_size)
        
        # Sum over time: Σ_k (1/2)|α_k|² Δt
        running_cost = 0.5 * torch.sum(control_squared, dim=0) * self.dt  # (batch_size,)
        
        # Average over batch
        return running_cost.mean()
    
    def compute_terminal_cost(self, x_terminal: torch.Tensor) -> torch.Tensor:
        """
        Compute the terminal cost as KL divergence: KL(P_{X_N} || μ_T)
        
        For generated samples X_N ~ P_{X_N} and target μ_T = N(μ, σ²):
            KL(P || Q) = E_P[log P(x)] - E_P[log Q(x)]
                       = -H(P) - E_P[log Q(x)]
        
        Since P_{X_N} is empirical, we estimate:
            KL ≈ E_{X_N}[-log q(X_N)] - H(P_{X_N})
        
        where:
            - E_{X_N}[-log q(X_N)] = cross-entropy with target
            - H(P_{X_N}) = entropy of generated distribution (estimated)
        
        For Gaussian target q(x) = N(μ, σ²):
            -log q(x) = (x - μ)² / (2σ²) + (1/2)log(2πσ²)
        
        Args:
            x_terminal: Terminal states of shape (batch_size, state_dim)
            
        Returns:
            kl_divergence: Scalar tensor (KL divergence estimate)
        """
        batch_size = x_terminal.shape[0]
        
        # =====================================================================
        # Part 1: Cross-entropy term E_{X_N}[-log q(X_N)]
        # For Gaussian target: -log q(x) = (x-μ)²/(2σ²) + 0.5*log(2πσ²)
        # =====================================================================
        diff = x_terminal - self.target_mean
        nll_target = 0.5 * torch.sum(diff ** 2, dim=-1) / self.target_var
        log_norm = 0.5 * self.state_dim * np.log(2 * np.pi * self.target_var)
        cross_entropy = (nll_target + log_norm).mean()  # E[-log q(X_N)]
        
        # =====================================================================
        # Part 2: Entropy term H(P_{X_N}) - estimated using sample statistics
        # For a Gaussian approximation: H(N(μ,σ²)) = 0.5*log(2πeσ²)
        # We estimate the variance of X_N from samples
        # =====================================================================
        # Estimate variance of generated distribution
        sample_var = torch.var(x_terminal, dim=0, unbiased=True)  # (state_dim,)
        sample_var = torch.clamp(sample_var, min=1e-6)  # Avoid log(0)
        
        # Gaussian entropy: H = 0.5 * log(2πe * σ²) = 0.5 * (1 + log(2π) + log(σ²))
        entropy = 0.5 * self.state_dim * (1 + np.log(2 * np.pi)) + 0.5 * torch.sum(torch.log(sample_var))
        
        # =====================================================================
        # KL divergence: KL(P || Q) = Cross-Entropy(P, Q) - Entropy(P)
        # =====================================================================
        kl_divergence = cross_entropy - entropy
        
        return kl_divergence
    
    def compute_total_cost(
        self,
        drift_net: DriftNetwork,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the total MFC cost:
            J(α) = E[ Σ_{k=0}^{N-1} (1/2)|α_k|² Δt ] + λ * KL(P_{X_N} || μ_T)
        
        where λ = self.terminal_weight is the KL penalty weight.
        
        Args:
            drift_net: Neural network for drift
            batch_size: Number of particles
            
        Returns:
            total_cost: Total loss (running + λ * KL)
            running_cost: Control energy cost
            terminal_cost: Weighted KL divergence (λ * KL)
            trajectories: Simulated trajectories for visualization
        """
        # Simulate trajectories
        trajectories, controls = self.simulate_trajectories(drift_net, batch_size)
        
        # Compute costs
        running_cost = self.compute_running_cost(controls)
        kl_divergence = self.compute_terminal_cost(trajectories[-1])
        
        # Apply weight λ to KL divergence
        # Total cost: J = Running Cost + λ * KL(P_{X_N} || μ_T)
        weighted_kl = self.terminal_weight * kl_divergence
        total_cost = running_cost + weighted_kl
        
        return total_cost, running_cost, weighted_kl, trajectories


# =============================================================================
# Training
# =============================================================================

def train_mfc(
    drift_net: DriftNetwork,
    simulator: MFCSimulator,
    n_epochs: int = 2000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    print_every: int = 100,
) -> dict:
    """
    Train the drift network to solve the MFC optimal control problem.
    
    We minimize:
        J(α_θ) = E[ Σ_{k=0}^{N-1} (1/2)|α_θ|² Δt ] + λ * KL(P_{X_N} || μ_T)
    
    where λ = simulator.terminal_weight is the KL penalty weight.
    
    Args:
        drift_net: Neural network to train
        simulator: MFC simulator
        n_epochs: Number of training epochs
        batch_size: Batch size for Monte Carlo estimation
        lr: Learning rate
        print_every: Print frequency
        
    Returns:
        history: Dictionary with training history
    """
    optimizer = optim.Adam(drift_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    history = {
        'total_loss': [],
        'running_cost': [],
        'terminal_cost': [],
    }
    
    print("=" * 70)
    print("Training MFC Generative Model")
    print("=" * 70)
    print(f"Time horizon T = {simulator.T}, Steps N = {simulator.N}")
    print(f"Target: N({simulator.target_mean}, {simulator.target_var})")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print(f">>> Terminal Weight λ = {simulator.terminal_weight} <<<")
    print("=" * 70)
    print(f"Loss: J(α) = Running Cost + {simulator.terminal_weight} × KL(P_{{X_N}} || μ_T)")
    print("=" * 70)
    
    drift_net.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute total cost (forward pass)
        total_cost, running_cost, terminal_cost, _ = simulator.compute_total_cost(
            drift_net, batch_size
        )
        
        # Backward pass
        total_cost.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(drift_net.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Record history
        history['total_loss'].append(total_cost.item())
        history['running_cost'].append(running_cost.item())
        history['terminal_cost'].append(terminal_cost.item())
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            kl_raw = terminal_cost.item() / simulator.terminal_weight  # Raw KL value
            print(
                f"Epoch [{epoch+1:4d}/{n_epochs}] | "
                f"Total: {total_cost.item():.4f} | "
                f"Running: {running_cost.item():.4f} | "
                f"λ*KL: {terminal_cost.item():.4f} (KL={kl_raw:.4f})"
            )
    
    print("=" * 70)
    print("Training complete!")
    print(f"Final Loss: {history['total_loss'][-1]:.4f} (λ = {simulator.terminal_weight})")
    print("=" * 70)
    
    return history


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(
    drift_net: DriftNetwork,
    simulator: MFCSimulator,
    history: dict,
    n_trajectories: int = 100,
    n_samples: int = 5000,
):
    """
    Comprehensive visualization of MFC generative model results.
    
    Creates four plots:
    1. Training loss curves
    2. Sample trajectories X_t
    3. Final distribution vs target
    4. Mean and variance evolution
    
    Args:
        drift_net: Trained drift network
        simulator: MFC simulator
        history: Training history
        n_trajectories: Number of trajectories to plot
        n_samples: Number of samples for distribution estimation
    """
    drift_net.eval()
    
    # Generate trajectories for visualization
    with torch.no_grad():
        trajectories, _ = simulator.simulate_trajectories(drift_net, n_samples)
    
    trajectories_np = trajectories.cpu().numpy().squeeze(-1)  # (N+1, n_samples)
    time_grid = simulator.time_grid.cpu().numpy()
    
    # Compute statistics over time
    means = np.mean(trajectories_np, axis=1)
    variances = np.var(trajectories_np, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # =========================================================================
    # Plot 1: Training Loss Curves
    # =========================================================================
    ax1 = axes[0, 0]
    epochs = np.arange(1, len(history['total_loss']) + 1)
    ax1.plot(epochs, history['total_loss'], 'b-', label='Total Cost', linewidth=2)
    ax1.plot(epochs, history['running_cost'], 'g--', label='Running Cost', linewidth=1.5)
    ax1.plot(epochs, history['terminal_cost'], 'r--', label='Terminal Cost', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # =========================================================================
    # Plot 2: Sample Trajectories
    # =========================================================================
    ax2 = axes[0, 1]
    
    # Plot subset of trajectories
    for i in range(min(n_trajectories, n_samples)):
        ax2.plot(
            time_grid, trajectories_np[:, i], 
            alpha=0.2, color='blue', linewidth=0.5
        )
    
    # Plot mean trajectory
    ax2.plot(time_grid, means, 'r-', linewidth=3, label='Mean trajectory')
    
    # Plot ± 2 std band
    stds = np.sqrt(variances)
    ax2.fill_between(
        time_grid, means - 2*stds, means + 2*stds,
        alpha=0.3, color='red', label='±2σ band'
    )
    
    # Target mean
    ax2.axhline(y=simulator.target_mean, color='green', linestyle='--', 
                linewidth=2, label=f'Target μ={simulator.target_mean}')
    
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('X_t', fontsize=12)
    ax2.set_title('Generated Trajectories', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 3: Final Distribution Histogram
    # =========================================================================
    ax3 = axes[1, 0]
    
    # Terminal samples
    x_terminal = trajectories_np[-1, :]
    
    # Histogram
    ax3.hist(
        x_terminal, bins=50, density=True, alpha=0.7, 
        color='blue', edgecolor='black', label='Generated $X_N$'
    )
    
    # Target Gaussian PDF
    x_range = np.linspace(
        simulator.target_mean - 4*simulator.target_std,
        simulator.target_mean + 4*simulator.target_std, 
        200
    )
    pdf_target = (
        1 / (simulator.target_std * np.sqrt(2 * np.pi)) * 
        np.exp(-0.5 * ((x_range - simulator.target_mean) / simulator.target_std) ** 2)
    )
    ax3.plot(x_range, pdf_target, 'r-', linewidth=3, 
             label=f'Target $\\mathcal{{N}}({simulator.target_mean}, {simulator.target_var})$')
    
    ax3.set_xlabel('$X_N$', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Terminal Distribution vs Target', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics text
    gen_mean = np.mean(x_terminal)
    gen_var = np.var(x_terminal)
    stats_text = f'Generated: μ={gen_mean:.3f}, σ²={gen_var:.3f}'
    ax3.text(
        0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # =========================================================================
    # Plot 4: Mean and Variance Evolution
    # =========================================================================
    ax4 = axes[1, 1]
    
    # Theoretical targets
    target_mean = simulator.target_mean
    target_var = simulator.target_var
    
    # Plot mean
    color_mean = 'tab:blue'
    ax4.plot(time_grid, means, color=color_mean, linewidth=2, label='Mean $\\mathbb{E}[X_t]$')
    ax4.axhline(y=target_mean, color=color_mean, linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time t', fontsize=12)
    ax4.set_ylabel('Mean', color=color_mean, fontsize=12)
    ax4.tick_params(axis='y', labelcolor=color_mean)
    
    # Create secondary y-axis for variance
    ax4b = ax4.twinx()
    color_var = 'tab:orange'
    ax4b.plot(time_grid, variances, color=color_var, linewidth=2, label='Var $\\mathbb{V}[X_t]$')
    ax4b.axhline(y=target_var, color=color_var, linestyle='--', alpha=0.5)
    ax4b.set_ylabel('Variance', color=color_var, fontsize=12)
    ax4b.tick_params(axis='y', labelcolor=color_var)
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    ax4.set_title('Mean & Variance Evolution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations
    ax4.annotate(
        f'Start: μ=0, σ²=0', xy=(0, 0), xytext=(0.1, 0.3),
        fontsize=9, arrowprops=dict(arrowstyle='->', color='gray')
    )
    ax4.annotate(
        f'End: μ→{target_mean}, σ²→{target_var}', 
        xy=(1, target_mean), xytext=(0.6, 0.5),
        fontsize=9, arrowprops=dict(arrowstyle='->', color='gray')
    )
    
    plt.tight_layout()
    plt.savefig('mfc_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved to 'mfc_results.png'")
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("Final Distribution Statistics")
    print("=" * 60)
    print(f"Generated X_N:  Mean = {gen_mean:.4f}, Variance = {gen_var:.4f}")
    print(f"Target μ_T:     Mean = {target_mean:.4f}, Variance = {target_var:.4f}")
    print("=" * 60)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main function to run the MFC generative modeling experiment.
    
    Problem Setup:
    - Initial: μ_0 = δ_0 (Dirac at zero)
    - Target: μ_T = N(1, 1) (Gaussian with mean 1, variance 1)
    - Time: T = 1.0, N = 50 steps
    - Goal: Learn optimal control α_θ(t, x) to transport μ_0 → μ_T
    """
    print("\n" + "=" * 70)
    print("  GENERATIVE MODELING VIA MEAN-FIELD CONTROL (DISCRETE TIME)")
    print("=" * 70)
    
    # =========================================================================
    # Problem Parameters
    # =========================================================================
    T = 1.0              # Terminal time
    N = 50               # Number of time steps
    sigma = 1.0          # Diffusion coefficient
    target_mean = 1.0    # Target Gaussian mean
    target_std = 1.0     # Target Gaussian std
    state_dim = 1        # State dimension
    terminal_weight = 10.0  # λ: KL divergence penalty weight
    
    # Training parameters
    n_epochs = 2000
    batch_size = 1024
    learning_rate = 1e-3
    
    # =========================================================================
    # Initialize Model and Simulator
    # =========================================================================
    print("\n[1] Initializing model and simulator...")
    
    # Create drift network (control parameterization)
    drift_net = DriftNetwork(
        state_dim=state_dim,
        hidden_dims=[64, 64, 64]
    ).to(device)
    
    print(f"    Drift Network: {sum(p.numel() for p in drift_net.parameters())} parameters")
    
    # Create MFC simulator
    simulator = MFCSimulator(
        T=T,
        N=N,
        sigma=sigma,
        target_mean=target_mean,
        target_std=target_std,
        state_dim=state_dim,
        terminal_weight=terminal_weight,
    )
    
    print(f"    Time: T={T}, Steps: N={N}, Δt={simulator.dt:.4f}")
    print(f"    Diffusion: σ={sigma}")
    print(f"    Initial: δ_0 (Dirac at 0)")
    print(f"    Target: N({target_mean}, {target_std**2})")
    print(f"    >>> Terminal Weight λ = {terminal_weight} <<<")
    
    # =========================================================================
    # Train the Model
    # =========================================================================
    print("\n[2] Training the MFC model...")
    
    history = train_mfc(
        drift_net=drift_net,
        simulator=simulator,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        print_every=200,
    )
    
    # =========================================================================
    # Visualize Results
    # =========================================================================
    print("\n[3] Generating visualizations...")
    
    visualize_results(
        drift_net=drift_net,
        simulator=simulator,
        history=history,
        n_trajectories=100,
        n_samples=5000,
    )
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n[4] Final evaluation with large sample size...")
    
    drift_net.eval()
    with torch.no_grad():
        n_eval = 10000
        trajectories, _ = simulator.simulate_trajectories(drift_net, n_eval)
        x_final = trajectories[-1].cpu().numpy().flatten()
        
        print(f"\n    Sample size: {n_eval}")
        print(f"    Generated Mean:     {np.mean(x_final):.4f} (target: {target_mean})")
        print(f"    Generated Variance: {np.var(x_final):.4f} (target: {target_std**2})")
        print(f"    Generated Std Dev:  {np.std(x_final):.4f} (target: {target_std})")
    
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70 + "\n")
    
    return drift_net, simulator, history


if __name__ == "__main__":
    drift_net, simulator, history = main()