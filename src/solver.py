"""
Iterative Backward Regression Solver with Polyak Averaging
============================================================

Core solver implementing the deep BSDE-style algorithm for MFC with
target/trainable network separation for stable training.

Algorithm (with Polyak Averaging):
1. Initialize N trainable networks {Y_θ_0, ..., Y_θ_{N-1}}
2. Initialize N target networks (copy of trainable)
3. Coupling Loop (iterations):
   a. Forward Pass: Simulate X using **TARGET** networks (stable policy)
   b. Terminal Condition: Y_N = KernelScoreEstimator(X_N)
   c. Backward Induction: Train **TRAINABLE** networks
   d. Soft Update: target = (1-τ)*target + τ*trainable  (τ=0.5)

Author: Lizhan HONG
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .config import Config
from .models import TimeConditionedMLP, create_networks
from .dynamics import EulerMaruyama
from .utils import (
    ExperimentManager, 
    GaussianProxy,
    KernelScoreEstimator,
    plot_trajectories,
    plot_terminal_distribution,
    plot_statistics_evolution,
)


class IterativeSolver:
    """
    Iterative Backward Regression solver with Polyak Averaging.
    
    Key Features:
    - Two sets of networks: target (for simulation) and trainable (for learning)
    - Soft update after each iteration prevents oscillation
    - Target networks provide stable policy during simulation
    
    Attributes:
        config: Full configuration
        device: Torch device
        trainable_networks: Networks being trained (updated each backward pass)
        target_networks: Networks for simulation (soft-updated)
        tau: Soft update coefficient (default 0.5)
    """
    
    def __init__(
        self, 
        config: Config, 
        exp_manager: Optional[ExperimentManager] = None,
        tau: float = 0.5,
    ):
        """
        Initialize the iterative solver.
        
        Args:
            config: Full configuration object
            exp_manager: Optional experiment manager for logging
            tau: Polyak averaging coefficient (0.5 = equal weight)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = tau
        
        # Create TRAINABLE networks (will be trained)
        self.trainable_networks = create_networks(
            config.model, 
            config.physics, 
            self.device
        )
        
        # Create TARGET networks (deep copy, for stable simulation)
        self.target_networks = [
            copy.deepcopy(net) for net in self.trainable_networks
        ]
        for net in self.target_networks:
            net.eval()  # Target networks always in eval mode
            for param in net.parameters():
                param.requires_grad = False  # No gradients for target
        
        # Dynamics engine
        self.dynamics = EulerMaruyama(
            config.physics,
            config.initial,
            self.device
        )
        
        # Terminal gradient estimator
        save_dir = exp_manager.exp_dir if exp_manager else None
        self.score_estimator = KernelScoreEstimator(
            config.target.mean,
            config.target.std,
            self.device,
            terminal_weight=config.training.terminal_weight,
            n_folds=5,
            n_grid=15,
            save_dir=save_dir,
        )
        
        # Experiment manager
        self.exp_manager = exp_manager
        
        # Training history
        self.history = {
            'iteration_losses': [],
            'terminal_means': [],
            'terminal_vars': [],
        }
        
        self._log(f"Initialized solver on device: {self.device}")
        self._log(f"Number of networks: {len(self.trainable_networks)}")
        self._log(f"Polyak averaging τ: {self.tau}")
        total_params = sum(
            sum(p.numel() for p in net.parameters()) 
            for net in self.trainable_networks
        )
        self._log(f"Total parameters (trainable): {total_params:,}")
    
    def _log(self, message: str):
        """Log a message."""
        if self.exp_manager:
            self.exp_manager.log(message)
        else:
            print(message)
    
    def _soft_update(self):
        """
        Perform Polyak (soft) update of target networks.
        
        target = (1 - τ) * target + τ * trainable
        """
        for target_net, train_net in zip(self.target_networks, self.trainable_networks):
            for target_param, train_param in zip(target_net.parameters(), train_net.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * train_param.data
                )
    
    def forward_pass_with_target(self, batch_size: int) -> torch.Tensor:
        """
        Perform forward simulation using TARGET networks (stable policy).
        
        Args:
            batch_size: Number of particles to simulate
            
        Returns:
            trajectories: Tensor of shape (N+1, batch_size, dim)
        """
        # Use target networks for simulation (stable)
        trajectories, _ = self.dynamics.simulate(
            self.target_networks, 
            batch_size, 
            return_controls=False
        )
        return trajectories
    
    def compute_terminal_gradient(self, x_terminal: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_N using Kernel Score Estimator with numerical stability.
        """
        return self.score_estimator.compute_terminal_gradient(
            x_terminal, 
            y_clamp=self.config.training.y_clamp, 
            verbose=True
        )
    
    def backward_step(
        self,
        k: int,
        x_k: torch.Tensor,
        y_target: torch.Tensor,
        epochs: int,
        lr: float,
    ) -> float:
        """
        Train TRAINABLE network k to regress target Y values.
        """
        network = self.trainable_networks[k]
        network.train()
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        
        # Time value for this step
        t_k = self.dynamics.time_grid[k]
        
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            y_pred = network(t_k, x_k)
            loss = torch.mean((y_pred - y_target) ** 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), 
                self.config.training.grad_clip
            )
            optimizer.step()
            
            final_loss = loss.item()
        
        network.eval()
        return final_loss
    
    def run_iteration(self, iteration: int) -> Dict:
        """
        Run one full coupling iteration with Polyak averaging.
        
        1. Forward pass using TARGET networks (stable)
        2. Compute terminal condition
        3. Backward pass to train TRAINABLE networks
        4. Soft update TARGET networks
        """
        batch_size = self.config.training.batch_size
        epochs = self.config.training.backward_epochs
        lr = self.config.training.learning_rate
        N = self.config.physics.N
        
        # =====================================================================
        # Step A: Forward Pass using TARGET networks
        # =====================================================================
        trajectories = self.forward_pass_with_target(batch_size)
        all_x = trajectories.detach()
        
        # =====================================================================
        # Step B: Terminal Condition
        # =====================================================================
        x_terminal = all_x[-1]
        y_current = self.compute_terminal_gradient(x_terminal)
        
        # =====================================================================
        # Step C: Backward Induction (train TRAINABLE networks)
        # =====================================================================
        backward_losses = []
        
        for k in range(N - 1, -1, -1):
            x_k = all_x[k].detach()
            y_target = y_current.detach()
            
            # Train trainable network k
            loss = self.backward_step(k, x_k, y_target, epochs, lr)
            backward_losses.append(loss)
            
            # Update y_current using trainable network
            with torch.no_grad():
                t_k = self.dynamics.time_grid[k]
                y_current = self.trainable_networks[k](t_k, x_k)
        
        # =====================================================================
        # Step D: Soft Update Target Networks
        # =====================================================================
        self._soft_update()
        
        # =====================================================================
        # Compute statistics
        # =====================================================================
        terminal_np = x_terminal.cpu().numpy().flatten()
        terminal_mean = np.mean(terminal_np)
        terminal_var = np.var(terminal_np)
        terminal_std = np.sqrt(terminal_var)
        avg_loss = np.mean(backward_losses)
        
        # Compute distance to target
        target_mean = self.config.target.mean
        target_std = self.config.target.std
        dist_mean = abs(terminal_mean - target_mean)
        dist_std = abs(terminal_std - target_std)
        
        stats = {
            'iteration': iteration,
            'avg_backward_loss': avg_loss,
            'terminal_mean': terminal_mean,
            'terminal_var': terminal_var,
            'dist_mean': dist_mean,
            'dist_std': dist_std,
        }
        
        # Record history
        self.history['iteration_losses'].append(avg_loss)
        self.history['terminal_means'].append(terminal_mean)
        self.history['terminal_vars'].append(terminal_var)
        
        return stats
    
    def run(self, n_iterations: Optional[int] = None) -> Dict:
        """
        Run the full iterative training procedure.
        """
        if n_iterations is None:
            n_iterations = self.config.training.iterations
        
        self._log("=" * 70)
        self._log("Starting Iterative Backward Regression (Polyak Averaging)")
        self._log("=" * 70)
        self._log(f"Iterations: {n_iterations}")
        self._log(f"Backward epochs per step: {self.config.training.backward_epochs}")
        self._log(f"Batch size: {self.config.training.batch_size}")
        self._log(f"Polyak τ: {self.tau}")
        self._log(f"λ (terminal_weight): {self.config.training.terminal_weight}")
        self._log(f"Target: N({self.config.target.mean}, {self.config.target.var})")
        self._log("=" * 70)
        
        for i in range(1, n_iterations + 1):
            # Note: bandwidth is optimized once on first iteration, then reused
            
            stats = self.run_iteration(i)
            
            # Log progress
            if i == 1 or i % self.config.training.print_every == 0 or i == n_iterations:
                self._log(
                    f"Iter [{i:3d}/{n_iterations}] | "
                    f"Loss: {stats['avg_backward_loss']:.6f} | "
                    f"μ={stats['terminal_mean']:.3f} (Δ={stats['dist_mean']:.3f}), "
                    f"σ={np.sqrt(stats['terminal_var']):.3f} (Δ={stats['dist_std']:.3f})"
                )
        
        # Final evaluation using target networks
        self._log("=" * 70)
        self._log("Training Complete! Evaluating with target networks...")
        with torch.no_grad():
            final_trajectories, _ = self.dynamics.simulate(
                self.target_networks,
                self.config.experiment.n_eval_samples,
                return_controls=False
            )
        final_terminal = final_trajectories[-1].cpu().numpy().flatten()
        final_mean = np.mean(final_terminal)
        final_var = np.var(final_terminal)
        final_std = np.sqrt(final_var)
        
        self._log(f"Final Mean: {final_mean:.4f} (target: {self.config.target.mean})")
        self._log(f"Final Std:  {final_std:.4f} (target: {self.config.target.std})")
        self._log("=" * 70)
        
        return {
            'history': self.history,
            'final_mean': final_mean,
            'final_var': final_var,
            'final_trajectories': final_trajectories,
        }
    
    def get_networks(self) -> List[nn.Module]:
        """Return target networks for evaluation."""
        return self.target_networks
    
    def visualize(self, save: bool = True) -> Dict:
        """
        Generate all visualization plots.
        
        Args:
            save: Whether to save plots to experiment directory
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # Generate trajectories for visualization
        with torch.no_grad():
            trajectories, controls = self.dynamics.simulate(
                self.target_networks,
                self.config.experiment.n_eval_samples,
                return_controls=True
            )
        
        trajectories_np = trajectories.cpu().numpy()
        time_grid_np = self.dynamics.time_grid.cpu().numpy()
        
        # Squeeze for 1D case: (N+1, batch, 1) -> (N+1, batch)
        if trajectories_np.ndim == 3 and trajectories_np.shape[-1] == 1:
            trajectories_np = trajectories_np.squeeze(-1)
        
        # 1. Trajectory plot
        fig_traj = plot_trajectories(
            trajectories_np,
            time_grid_np,
            target_mean=self.config.target.mean,
            n_plot=min(100, trajectories_np.shape[1]),
        )
        figures['trajectories'] = fig_traj
        
        if save and self.exp_manager:
            fig_traj.savefig(
                self.exp_manager.exp_dir / "trajectories.png",
                dpi=150, bbox_inches='tight'
            )
            self._log("Saved plot: trajectories.png")
        
        # 2. Terminal distribution
        terminal = trajectories_np[-1].flatten()
        fig_terminal = plot_terminal_distribution(
            terminal,
            target_mean=self.config.target.mean,
            target_std=self.config.target.std,
        )
        figures['terminal_distribution'] = fig_terminal
        
        if save and self.exp_manager:
            fig_terminal.savefig(
                self.exp_manager.exp_dir / "terminal_distribution.png",
                dpi=150, bbox_inches='tight'
            )
            self._log("Saved plot: terminal_distribution.png")
        
        # 3. Statistics evolution over time (from trajectories)
        fig_stats = plot_statistics_evolution(
            trajectories_np,
            time_grid_np,
            target_mean=self.config.target.mean,
            target_var=self.config.target.var,
        )
        figures['statistics_evolution'] = fig_stats
        
        if save and self.exp_manager:
            fig_stats.savefig(
                self.exp_manager.exp_dir / "statistics_evolution.png",
                dpi=150, bbox_inches='tight'
            )
            self._log("Saved plot: statistics_evolution.png")
        
        self._log("Saved all visualization plots")
        
        return figures
