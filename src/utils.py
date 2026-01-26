"""
Utilities Module
================

Experiment management, Gaussian Proxy score estimation, and visualization utilities.

Author: Lizhan HONG
"""

import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config


class ExperimentManager:
    """
    Manages experiment directories, logging, and artifact saving.
    
    On initialization:
    1. Creates a unique timestamped directory: experiments/YYYY-MM-DD_HH-MM-SS/
    2. Copies the config.yaml for reproducibility
    3. Initializes a log file
    
    Provides methods to save plots and logs to the experiment folder.
    """
    
    def __init__(self, config: Config, config_path: str = "config.yaml"):
        """
        Initialize experiment manager.
        
        Args:
            config: Configuration object
            config_path: Path to the YAML config file (to copy)
        """
        self.config = config
        self.config_path = config_path
        
        # Create unique experiment directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{timestamp}_{config.experiment.name}"
        self.exp_dir = Path("experiments") / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config for reproducibility
        if os.path.exists(config_path):
            shutil.copy(config_path, self.exp_dir / "config.yaml")
        
        # Initialize log file
        self.log_path = self.exp_dir / "training.log"
        with open(self.log_path, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")
            f.write(str(config) + "\n\n")
        
        print(f"📁 Experiment directory: {self.exp_dir}")
    
    def log(self, message: str, also_print: bool = True):
        """
        Log a message to the log file.
        
        Args:
            message: Message to log
            also_print: Whether to also print to console
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        with open(self.log_path, 'a') as f:
            f.write(log_line + "\n")
        
        if also_print:
            print(message)
    
    def save_plot(self, fig: plt.Figure, name: str, dpi: int = 150):
        """
        Save a matplotlib figure to the experiment directory.
        
        Args:
            fig: Matplotlib figure
            name: Filename (without extension)
            dpi: Resolution
        """
        path = self.exp_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        self.log(f"Saved plot: {name}.png", also_print=False)
    
    def save_checkpoint(self, networks: List[torch.nn.Module], iteration: int):
        """
        Save network checkpoints.
        
        Args:
            networks: List of networks to save
            iteration: Current iteration number
        """
        checkpoint_dir = self.exp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        for k, net in enumerate(networks):
            path = checkpoint_dir / f"network_{k:03d}_iter_{iteration:04d}.pt"
            torch.save(net.state_dict(), path)
    
    def finalize(self, final_stats: Dict):
        """
        Finalize the experiment log with final statistics.
        
        Args:
            final_stats: Dictionary of final statistics
        """
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("=" * 70 + "\n")
            for key, value in final_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nCompleted: {datetime.now().isoformat()}\n")


class GaussianProxy:
    """
    Gaussian Proxy for terminal gradient estimation.
    
    For the MFC problem:
        min_α E[ Σ_{k=0}^{N-1} (1/2)|α_k|² Δt + λ * KL(P_{X_N} || μ_T) ]
    
    We compute the terminal gradient Y_N using score matching:
        Y_N ≈ λ * [Σ_target^{-1}(X_N - μ_target) - Σ_batch^{-1}(X_N - μ_batch)]
    
    The λ coefficient (terminal_weight) scales the terminal cost relative
    to the running cost in the MFC objective.
    """
    
    def __init__(
        self, 
        target_mean: float, 
        target_std: float, 
        device: torch.device,
        terminal_weight: float = 10.0,  # Kept for documentation, not used in computation
    ):
        """
        Initialize Gaussian Proxy.
        
        Args:
            target_mean: Target distribution mean μ
            target_std: Target distribution std σ
            device: Torch device
            terminal_weight: λ in the MFC cost (for reference only)
        """
        self.target_mean = target_mean
        self.target_var = target_std ** 2
        self.target_precision = 1.0 / self.target_var  # Σ^{-1} for 1D
        self.device = device
        self.terminal_weight = terminal_weight  # λ (applied to Y_N)
    
    def compute_terminal_gradient(
        self, 
        x_terminal: torch.Tensor,
        y_clamp: float = 100.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute the terminal gradient Y_N using Gaussian Proxy with numerical stability.
        
        Formula:
            Y_N = (X_N - μ_target) / σ²_target - (X_N - μ_batch) / σ²_batch
        
        Numerical Stability:
            1. Variance floor: min variance = 0.001 (prevents score explosion)
            2. Y_N clamping to [-y_clamp, y_clamp]
            3. Debug logging of Y_N statistics
        
        Args:
            x_terminal: Terminal states (batch_size, dim)
            y_clamp: Maximum absolute value for Y_N (default: 10.0)
            verbose: Whether to print debug statistics
            
        Returns:
            y_terminal: Terminal adjoint values (batch_size, dim), clamped
        """
        # Batch statistics with STRONG variance floor
        batch_mean = x_terminal.mean(dim=0, keepdim=True)  # (1, dim)
        batch_var = x_terminal.var(dim=0, keepdim=True, unbiased=True)  # (1, dim)
        # Regularized precision to prevent score explosion: 1/(σ² + ε)
        batch_precision = 1.0 / (batch_var + 0.1)
        
        # Y_N = λ * [(X_N - μ_target) / σ²_target - (X_N - μ_batch) / σ²_batch]
        target_term = self.target_precision * (x_terminal - self.target_mean)
        generated_term = batch_precision * (x_terminal - batch_mean)
        
        y_terminal = self.terminal_weight * (target_term - generated_term)
        
        # Debug logging BEFORE clamping
        if verbose:
            y_mean = y_terminal.mean().item()
            y_std = y_terminal.std().item()
            y_min = y_terminal.min().item()
            y_max = y_terminal.max().item()
            x_mean = x_terminal.mean().item()
            x_std = x_terminal.std().item()
            print(f"    [DEBUG] X_N: μ={x_mean:.4f}, σ={x_std:.4f}")
            print(f"    [DEBUG] Y_N (pre-clamp): μ={y_mean:.4f}, σ={y_std:.4f}, "
                  f"range=[{y_min:.2f}, {y_max:.2f}], batch_var={batch_var.item():.4f}")
        
        # Clamp Y_N to prevent gradient explosion
        y_terminal = torch.clamp(y_terminal, min=-y_clamp, max=y_clamp)
        
        if verbose:
            clipped_pct = ((y_terminal.abs() >= y_clamp - 0.01).float().mean() * 100).item()
            if clipped_pct > 5:
                print(f"    [WARNING] {clipped_pct:.1f}% of Y_N values clipped!")
        
        return y_terminal


class KernelScoreEstimator:
    """
    Kernel Density Score (KDS) Estimator with Adaptive Bandwidth Selection.
    
    Replaces GaussianProxy with a more robust estimator that can handle
    multimodal distributions and avoids variance collapse.
    
    The density is estimated as a GMM where each particle is a centroid:
        p̂(x) = (1/M) Σᵢ N(x | xᵢ, h²I)
    
    The score ∇log p̂(x) is computed analytically using the softmax kernel trick.
    
    Bandwidth h is optimized via K-Fold Cross-Validation to maximize
    the held-out log-likelihood (U-shaped curve, minimum = optimal).
    """
    
    def __init__(
        self,
        target_mean: float,
        target_std: float,
        device: torch.device,
        terminal_weight: float = 10.0,
        n_folds: int = 5,
        n_grid: int = 15,  # Coarse grid for speed
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize Kernel Score Estimator.
        
        Args:
            target_mean: Target distribution mean μ
            target_std: Target distribution std σ
            device: Torch device
            terminal_weight: λ coefficient for terminal cost
            n_folds: Number of folds for cross-validation
            n_grid: Number of grid points for bandwidth search (log scale)
            save_dir: Directory to save optimization plots
        """
        self.target_mean = target_mean
        self.target_var = target_std ** 2
        self.target_precision = 1.0 / self.target_var
        self.terminal_weight = terminal_weight
        self.device = device
        
        self.n_folds = n_folds
        self.n_grid = n_grid
        self.save_dir = save_dir
        
        # Optimal bandwidth (set after optimization)
        self.optimal_h: Optional[float] = None
    
    def _silverman_bandwidth(self, x: torch.Tensor) -> float:
        """
        Compute Silverman's rule of thumb bandwidth.
        
        h = 1.06 * σ * n^(-1/5)  (for 1D Gaussian-like data)
        
        Args:
            x: Data points (N, dim)
            
        Returns:
            Reference bandwidth
        """
        n = x.size(0)
        dim = x.size(-1)
        
        # Use median absolute deviation for robustness
        std = x.std(dim=0).mean().item()
        
        # Silverman's rule
        h_silverman = 1.06 * std * (n ** (-1.0 / 5))
        
        return max(h_silverman, 0.01)  # Ensure minimum bandwidth
    
    def _log_kernel(
        self, 
        x: torch.Tensor, 
        centroids: torch.Tensor, 
        h: float
    ) -> torch.Tensor:
        """
        Compute log Gaussian kernel values (non-normalized, for score computation).
        
        log K_h(x, c) ∝ -||x - c||² / (2h²)
        
        Normalization constant is omitted since it cancels in score gradient.
        
        Args:
            x: Query points (N, dim)
            centroids: Kernel centroids (M, dim)
            h: Bandwidth
            
        Returns:
            log_kernel: Log kernel values (N, M)
        """
        # Pairwise squared distances: (N, M)
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
        c_sq = (centroids ** 2).sum(dim=-1, keepdim=True).T  # (1, M)
        cross = x @ centroids.T  # (N, M)
        
        dist_sq = x_sq + c_sq - 2 * cross  # (N, M)
        
        # Log kernel (non-normalized)
        log_kernel = -dist_sq / (2 * h ** 2)
        
        return log_kernel
        
    def _log_kernel_normalized(
        self, 
        x: torch.Tensor, 
        centroids: torch.Tensor, 
        h: float
    ) -> torch.Tensor:
        """
        Compute NORMALIZED log Gaussian kernel values.
        
        log K_h(x, c) = -||x - c||² / (2h²) - (d/2) * log(2πh²)
        
        The normalization constant is CRUCIAL for U-shaped CV curve.
        
        Args:
            x: Query points (N, dim)
            centroids: Kernel centroids (M, dim)
            h: Bandwidth
            
        Returns:
            log_kernel: Log kernel values (N, M)
        """
        dim = x.size(-1)
        
        # Pairwise squared distances: (N, M)
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
        c_sq = (centroids ** 2).sum(dim=-1, keepdim=True).T  # (1, M)
        cross = x @ centroids.T  # (N, M)
        
        dist_sq = x_sq + c_sq - 2 * cross  # (N, M)
        
        # Log kernel WITH normalization constant
        # log N(x|c, h²I) = -||x-c||²/(2h²) - (d/2)*log(2π) - d*log(h)
        log_norm = -0.5 * dim * np.log(2 * np.pi) - dim * np.log(h)
        log_kernel = -dist_sq / (2 * h ** 2) + log_norm
        
        return log_kernel
     
    def _compute_log_likelihood(
        self, 
        x_val: torch.Tensor, 
        x_train: torch.Tensor, 
        h: float
    ) -> float:
        """
        Compute log-likelihood of validation set under KDE with training set.
        
        log p(x_val) = Σᵢ log[ (1/M) Σⱼ K_h(x_val_i, x_train_j) ]
        
        Using log-sum-exp trick for numerical stability.
        
        Args:
            x_val: Validation points (N_val, dim)
            x_train: Training centroids (N_train, dim)
            h: Bandwidth
            
        Returns:
            Total log-likelihood (scalar)
        """
        M = x_train.size(0)
        
        # Log kernel values WITH normalization: (N_val, N_train)
        log_k = self._log_kernel_normalized(x_val, x_train, h)
        
        # Log-sum-exp over training points: log Σⱼ K_h
        # Then subtract log(M) for the 1/M normalization
        log_density = torch.logsumexp(log_k, dim=1) - np.log(M)  # (N_val,)
        
        # Average over validation points (per-sample LL)
        avg_log_lik = log_density.mean().item()
        
        return avg_log_lik
    
    def optimize_bandwidth(
        self, 
        x_batch: torch.Tensor,
        plot: bool = True,
    ) -> float:
        """
        Find optimal bandwidth using K-Fold Cross-Validation.
        
        Uses Silverman's rule to set adaptive search range, then finds
        the bandwidth that maximizes held-out log-likelihood.
        
        The CV curve should be U-shaped:
        - Small h: Undersmoothing → high variance → low LL
        - Large h: Oversmoothing → high bias → low LL  
        - Optimal h: Balance → maximum LL
        
        Args:
            x_batch: Batch of particles (M, dim) - typically X_N
            plot: Whether to generate and save optimization plot
            
        Returns:
            Optimal bandwidth h*
        """
        M = x_batch.size(0)
        dim = x_batch.size(-1)
        
        # Use Silverman's rule as reference
        h_ref = self._silverman_bandwidth(x_batch)
        
        # Search range: 0.1 * h_ref to 10 * h_ref (covers undersmooth to oversmooth)
        h_min = max(0.01, 0.1 * h_ref)
        h_max = min(10.0, 10.0 * h_ref)
        
        # Coarse log-spaced grid (15 points for speed)
        h_grid = np.logspace(np.log10(h_min), np.log10(h_max), self.n_grid)
        
        # K-Fold split indices
        indices = torch.randperm(M, device=self.device)
        fold_size = M // self.n_folds
        
        # Track CV log-likelihoods for each bandwidth
        cv_scores = []
        
        for h in h_grid:
            fold_lls = []
            
            for fold_idx in range(self.n_folds):
                # Split into train/val
                val_start = fold_idx * fold_size
                val_end = val_start + fold_size
                val_indices = indices[val_start:val_end]
                train_indices = torch.cat([indices[:val_start], indices[val_end:]])
                
                x_val = x_batch[val_indices]
                x_train = x_batch[train_indices]
                
                # Compute log-likelihood on validation fold
                ll = self._compute_log_likelihood(x_val, x_train, h)
                fold_lls.append(ll)
            
            # Average across folds
            cv_scores.append(np.mean(fold_lls))
        
        cv_scores = np.array(cv_scores)
        
        # Find optimal bandwidth (MAXIMUM log-likelihood)
        best_idx = np.argmax(cv_scores)
        self.optimal_h = h_grid[best_idx]
        
        # Generate optimization plot
        if plot and self.save_dir is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.semilogx(h_grid, cv_scores, 'b-o', linewidth=2, markersize=6)
            ax.axvline(
                self.optimal_h, color='r', linestyle='--', linewidth=2,
                label=f'Optimal h* = {self.optimal_h:.4f}'
            )
            ax.axvline(
                h_ref, color='g', linestyle=':', linewidth=2, alpha=0.7,
                label=f'Silverman h = {h_ref:.4f}'
            )
            ax.scatter([self.optimal_h], [cv_scores[best_idx]], 
                      color='r', s=150, zorder=5, marker='*')
            
            ax.set_xlabel('Bandwidth h', fontsize=12)
            ax.set_ylabel('CV Log-Likelihood (per sample)', fontsize=12)
            ax.set_title('Bandwidth Optimization via K-Fold Cross-Validation', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.save_dir / f"bandwidth_optimization_{timestamp}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return self.optimal_h
    
    def compute_score(
        self, 
        x_batch: torch.Tensor, 
        h: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute the score ∇log p̂(x) for each point in the batch.
        
        For KDE: p̂(x) = (1/M) Σⱼ K_h(x, xⱼ)
        
        Score formula:
            ∇log p̂(xᵢ) = Σⱼ wᵢⱼ * (xⱼ - xᵢ) / h²
            
        where wᵢⱼ = K_h(xᵢ, xⱼ) / Σₖ K_h(xᵢ, xₖ)  (softmax weights)
        
        Uses log-sum-exp trick for numerical stability.
        
        Args:
            x_batch: Particles (M, dim)
            h: Bandwidth (uses optimal_h if not provided)
            
        Returns:
            score: Score vectors (M, dim)
        """
        if h is None:
            if self.optimal_h is None:
                raise ValueError("Bandwidth not set. Call optimize_bandwidth first.")
            h = self.optimal_h
        
        M = x_batch.size(0)
        dim = x_batch.size(-1)
        
        # Log kernel weights: (M, M)
        log_k = self._log_kernel(x_batch, x_batch, h)
        
        # Softmax weights: wᵢⱼ = exp(log_k_ij) / Σₖ exp(log_k_ik)
        # Using log-sum-exp for stability
        log_sum = torch.logsumexp(log_k, dim=1, keepdim=True)  # (M, 1)
        log_weights = log_k - log_sum  # (M, M)
        weights = torch.exp(log_weights)  # (M, M)
        
        # Compute (xⱼ - xᵢ) for all pairs: (M, M, dim)
        # diff[i, j, :] = x[j, :] - x[i, :]
        diff = x_batch.unsqueeze(0) - x_batch.unsqueeze(1)  # (M, M, dim)
        
        # Weighted sum: Σⱼ wᵢⱼ * (xⱼ - xᵢ) / (h² + ε)
        # weights: (M, M) -> (M, M, 1) for broadcasting
        weighted_diff = weights.unsqueeze(-1) * diff  # (M, M, dim)
        score = weighted_diff.sum(dim=1) / (h ** 2 + 0.1)  # (M, dim) regularized
        
        return score
    
    def compute_terminal_gradient(
        self, 
        x_terminal: torch.Tensor,
        y_clamp: float = 100.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute terminal gradient Y_N using KDS estimator with numerical stability.
        
        For MFC with terminal cost KL(ρ_N || μ_target):
            Y_N = λ * [(x - μ_target)/σ²_target - score_KDE]
        
        Numerical Stability:
            1. Bandwidth has minimum floor (from optimize_bandwidth)
            2. Y_N is clamped to [-y_clamp, y_clamp]
            3. Debug logging of Y_N statistics
        
        Args:
            x_terminal: Terminal states (batch_size, dim)
            y_clamp: Maximum absolute value for Y_N (default: 10.0)
            verbose: Whether to print debug statistics
            
        Returns:
            y_terminal: Terminal adjoint values (batch_size, dim), clamped
        """
        # Optimize bandwidth if not already done
        if self.optimal_h is None:
            self.optimize_bandwidth(x_terminal, plot=True)
        
        # Enforce minimum bandwidth to prevent score explosion
        h_effective = max(self.optimal_h, 0.1)  # Minimum bandwidth floor
        
        # Generated score from KDE with effective bandwidth
        generated_score = self.compute_score(x_terminal, h=h_effective)
        
        # Y_N = λ * [(x - μ_target)/σ²_target - score_KDE]
        target_term = self.target_precision * (x_terminal - self.target_mean)
        y_terminal = self.terminal_weight * (target_term - generated_score)
        
        # Debug logging BEFORE clamping
        if verbose:
            y_mean = y_terminal.mean().item()
            y_std = y_terminal.std().item()
            y_min = y_terminal.min().item()
            y_max = y_terminal.max().item()
            x_mean = x_terminal.mean().item()
            x_std = x_terminal.std().item()
            print(f"    [DEBUG] X_N: μ={x_mean:.4f}, σ={x_std:.4f}")
            print(f"    [DEBUG] Y_N (pre-clamp): μ={y_mean:.4f}, σ={y_std:.4f}, "
                  f"range=[{y_min:.2f}, {y_max:.2f}], h={h_effective:.4f}")
        
        # Clamp Y_N to prevent gradient explosion
        y_terminal = torch.clamp(y_terminal, min=-y_clamp, max=y_clamp)
        
        if verbose:
            clipped_pct = ((y_terminal.abs() >= y_clamp - 0.01).float().mean() * 100).item()
            if clipped_pct > 5:
                print(f"    [WARNING] {clipped_pct:.1f}% of Y_N values clipped!")
        
        return y_terminal


def plot_trajectories(
    trajectories: np.ndarray,
    time_grid: np.ndarray,
    target_mean: float,
    n_plot: int = 100,
    title: str = "Generated Trajectories",
) -> plt.Figure:
    """
    Plot generated trajectories with mean and confidence bands.
    
    Args:
        trajectories: Array of shape (N+1, n_samples)
        time_grid: Time points of shape (N+1,)
        target_mean: Target distribution mean
        n_plot: Number of trajectories to plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_samples = trajectories.shape[1]
    means = np.mean(trajectories, axis=1)
    stds = np.std(trajectories, axis=1)
    
    # Plot sample trajectories
    for i in range(min(n_plot, n_samples)):
        ax.plot(time_grid, trajectories[:, i], alpha=0.15, color='blue', linewidth=0.5)
    
    # Mean trajectory
    ax.plot(time_grid, means, 'r-', linewidth=2.5, label='Mean trajectory')
    
    # ±2σ band
    ax.fill_between(
        time_grid, means - 2*stds, means + 2*stds,
        alpha=0.25, color='red', label='±2σ band'
    )
    
    # Target mean line
    ax.axhline(y=target_mean, color='green', linestyle='--', 
               linewidth=2, label=f'Target μ={target_mean}')
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('$X_t$', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_terminal_distribution(
    x_terminal: np.ndarray,
    target_mean: float,
    target_std: float,
    title: str = "Terminal Distribution vs Target",
) -> plt.Figure:
    """
    Plot histogram of terminal distribution against target PDF.
    
    Args:
        x_terminal: Terminal samples of shape (n_samples,)
        target_mean: Target mean
        target_std: Target std
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(x_terminal, bins=50, density=True, alpha=0.7,
            color='blue', edgecolor='black', label='Generated $X_N$')
    
    # Target PDF
    x_range = np.linspace(target_mean - 4*target_std, target_mean + 4*target_std, 200)
    pdf_target = (
        1 / (target_std * np.sqrt(2 * np.pi)) * 
        np.exp(-0.5 * ((x_range - target_mean) / target_std) ** 2)
    )
    ax.plot(x_range, pdf_target, 'r-', linewidth=3,
            label=f'Target $\\mathcal{{N}}({target_mean}, {target_std**2})$')
    
    # Statistics
    gen_mean = np.mean(x_terminal)
    gen_var = np.var(x_terminal)
    stats_text = f'Generated: μ={gen_mean:.3f}, σ²={gen_var:.3f}'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('$X_N$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_statistics_evolution(
    trajectories: np.ndarray,
    time_grid: np.ndarray,
    target_mean: float,
    target_var: float,
    title: str = "Mean & Variance Evolution",
) -> plt.Figure:
    """
    Plot mean and variance evolution over time.
    
    Args:
        trajectories: Array of shape (N+1, n_samples)
        time_grid: Time points
        target_mean: Target mean
        target_var: Target variance
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = np.mean(trajectories, axis=1)
    variances = np.var(trajectories, axis=1)
    
    # Mean
    color_mean = 'tab:blue'
    ax.plot(time_grid, means, color=color_mean, linewidth=2, label='Mean $\\mathbb{E}[X_t]$')
    ax.axhline(y=target_mean, color=color_mean, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Mean', color=color_mean, fontsize=12)
    ax.tick_params(axis='y', labelcolor=color_mean)
    
    # Variance (secondary axis)
    ax2 = ax.twinx()
    color_var = 'tab:orange'
    ax2.plot(time_grid, variances, color=color_var, linewidth=2, label='Var $\\mathbb{V}[X_t]$')
    ax2.axhline(y=target_var, color=color_var, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Variance', color=color_var, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_var)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig
