"""
Score Matching Estimator Module
================================

Neural network-based terminal score estimation using Denoising Score Matching (DSM).
Replaces kernel methods with a learned estimator per supervisor's instructions.

The DSM loss:
    L(θ) = E_{x ~ Batch, ε ~ N(0,I)} [ || σ * S_θ(x + σε) + ε ||² ]

Supports two modes for the target distribution score ∇log ν(x):
  - "analytical": Hardcoded Gaussian formula -(x - μ)/σ² (original behavior)
  - "learned":    Pre-trained TargetScoreTrainer via DSM on target samples

Author: Lizhan HONG
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class ScoreNet(nn.Module):
    """
    Simple MLP for score function estimation S_θ(x) ≈ ∇log ρ(x).
    
    Architecture:
    - Input: x ∈ ℝ^dim
    - Hidden: 3 layers with SiLU (smooth, works well for scores)
    - Output: score ∈ ℝ^dim
    """
    
    def __init__(self, dim: int, hidden_dim: int = 128):
        """
        Initialize ScoreNet.
        
        Args:
            dim: State space dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        
        # Initialize with small weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable gradients."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score S_θ(x).
        
        Args:
            x: Input states (batch_size, dim)
            
        Returns:
            score: Score vectors (batch_size, dim)
        """
        return self.net(x)


# =============================================================================
# Target Score Trainer (for arbitrary target distributions)
# =============================================================================

class TargetScoreTrainer:
    """
    Pre-trains a ScoreNet to learn ∇log ν(x) from target samples via DSM.
    
    This is trained ONCE before the MFC control loop starts.
    The learned score replaces the analytical Gaussian formula -(x-μ)/σ²,
    enabling support for arbitrary target distributions (GMM, images, etc.).
    
    DSM Loss:
        L(θ) = E_{x ~ ν, ε ~ N(0,I)} [ || σ S_θ(x + σε) + ε ||² ]
    """
    
    def __init__(
        self,
        dim: int,
        device: torch.device,
        hidden_dim: int = 128,
        n_pretrain_steps: int = 2000,
        lr: float = 0.001,
        sigma_dsm: float = 0.1,
    ):
        """
        Initialize TargetScoreTrainer.
        
        Args:
            dim: State space dimension
            device: Torch device
            hidden_dim: Hidden layer dimension for ScoreNet
            n_pretrain_steps: Number of DSM training steps
            lr: Learning rate for Adam optimizer
            sigma_dsm: Smoothing noise std for DSM
        """
        self.dim = dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_pretrain_steps = n_pretrain_steps
        self.lr = lr
        self.sigma_dsm = sigma_dsm
        
        # Create the network
        self.net = ScoreNet(dim, hidden_dim).to(device)
        self._trained = False
    
    def train(self, target_samples: torch.Tensor, verbose: bool = True) -> float:
        """
        Train the score network on target samples using DSM.
        
        Args:
            target_samples: Samples from the target distribution ν,
                            shape (n_samples, dim)
            verbose: Whether to print training progress
            
        Returns:
            Final DSM loss value
        """
        if target_samples.dim() == 1:
            target_samples = target_samples.unsqueeze(-1)
        
        target_samples = target_samples.to(self.device)
        n_samples = target_samples.shape[0]
        
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        final_loss = 0.0
        for step in range(self.n_pretrain_steps):
            optimizer.zero_grad()
            
            # Mini-batch: sample a random subset
            batch_size = min(1024, n_samples)
            idx = torch.randint(0, n_samples, (batch_size,), device=self.device)
            x_batch = target_samples[idx]
            
            # DSM: perturb samples
            epsilon = torch.randn_like(x_batch)
            x_noisy = x_batch + self.sigma_dsm * epsilon
            
            # Network prediction
            score_pred = self.net(x_noisy)
            
            # DSM loss: || σ S_θ(x + σε) + ε ||²
            residual = self.sigma_dsm * score_pred + epsilon
            loss = (residual ** 2).sum(dim=-1).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()
            
            final_loss = loss.item()
            
            if verbose and (step + 1) % 500 == 0:
                print(f"    [TargetScore] Step {step+1}/{self.n_pretrain_steps}, "
                      f"DSM loss: {final_loss:.6f}")
        
        self.net.eval()
        self._trained = True
        
        if verbose:
            print(f"    [TargetScore] Training complete. Final loss: {final_loss:.6f}")
        
        return final_loss
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the learned target score ∇log ν(x).
        
        Args:
            x: Query points (batch_size, dim)
            
        Returns:
            score: Score vectors (batch_size, dim)
        """
        if not self._trained:
            raise RuntimeError(
                "TargetScoreTrainer not trained. Call train() first."
            )
        
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        with torch.no_grad():
            return self.net(x)
    
    @property
    def is_trained(self) -> bool:
        return self._trained


# =============================================================================
# Score Estimator (terminal condition Y_N)
# =============================================================================

class ScoreEstimator:
    """
    Denoising Score Matching (DSM) Estimator for terminal condition.
    
    Replaces KernelScoreEstimator with a neural network trained via DSM.
    The network is re-initialized and trained from scratch on each batch
    of terminal particles X_N.
    
    Supports two modes for the target score -∇log ν(x):
      - "analytical": Gaussian formula (x - μ)/σ²  (default, backward compatible)
      - "learned":    Pre-trained TargetScoreTrainer
    
    DSM Loss:
        L(θ) = E_{x, ε} [ || σ S_θ(x + σε) + ε ||² ]
    
    This implicitly matches:
        S_θ(x) ≈ ∇log ρ_σ(x)  (smoothed density)
    """
    
    def __init__(
        self,
        target_mean: float,
        target_std: float,
        device: torch.device,
        terminal_weight: float = 10.0,
        hidden_dim: int = 128,
        n_steps: int = 150,
        lr: float = 0.01,
        sigma_smooth: float = 0.1,
        target_score_mode: str = "analytical",
        target_score_trainer: Optional[TargetScoreTrainer] = None,
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """
        Initialize Score Estimator.
        
        Args:
            target_mean: Target distribution mean μ (used in analytical mode)
            target_std: Target distribution std σ (used in analytical mode)
            device: Torch device
            terminal_weight: λ coefficient for terminal cost
            hidden_dim: Hidden dimension for ScoreNet
            n_steps: Number of DSM training steps per batch
            lr: Learning rate for Adam optimizer
            sigma_smooth: Smoothing noise std for DSM
            target_score_mode: "analytical" or "learned"
            target_score_trainer: Pre-trained TargetScoreTrainer (required if mode="learned")
        """
        self.target_mean = target_mean
        self.target_var = target_std ** 2
        self.target_precision = 1.0 / self.target_var
        self.terminal_weight = terminal_weight
        self.device = device
        
        # Target score mode
        self.target_score_mode = target_score_mode
        self.target_score_trainer = target_score_trainer
        
        if target_score_mode == "learned" and target_score_trainer is None:
            raise ValueError(
                "target_score_trainer must be provided when "
                "target_score_mode='learned'"
            )
        
        # DSM hyperparameters (for the generated density ρ_N score)
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.lr = lr
        self.sigma_smooth = sigma_smooth
        
        # Network will be created on first call
        self.score_net: Optional[ScoreNet] = None
        self.dim: Optional[int] = None
    
    def _create_network(self, dim: int) -> ScoreNet:
        """Create and initialize a new ScoreNet."""
        self.dim = dim
        net = ScoreNet(dim, self.hidden_dim).to(self.device)
        return net
    
    def train(self, x_batch: torch.Tensor) -> float:
        """
        Train ScoreNet on a batch of particles using DSM.
        
        The network is re-initialized from scratch to avoid bias
        from previous iterations.
        
        DSM Loss:
            L = (1/N) Σᵢ || σ S_θ(xᵢ + σεᵢ) + εᵢ ||²
        
        Args:
            x_batch: Particles (batch_size, dim)
            
        Returns:
            Final loss value
        """
        # Ensure 2D shape
        if x_batch.dim() == 1:
            x_batch = x_batch.unsqueeze(-1)
        
        batch_size, dim = x_batch.shape
        
        # Create fresh network
        self.score_net = self._create_network(dim)
        self.score_net.train()
        
        # Fast optimizer for inner loop
        optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr)
        
        # DSM training loop
        final_loss = 0.0
        for step in range(self.n_steps):
            optimizer.zero_grad()
            
            # Sample noise
            epsilon = torch.randn_like(x_batch)  # ε ~ N(0, I)
            
            # Perturbed samples
            x_noisy = x_batch + self.sigma_smooth * epsilon
            
            # Network output
            score_pred = self.score_net(x_noisy)
            
            # DSM loss: || σ * S_θ(x + σε) + ε ||²
            residual = self.sigma_smooth * score_pred + epsilon
            loss = (residual ** 2).sum(dim=-1).mean()
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            
            optimizer.step()
            
            final_loss = loss.item()
        
        self.score_net.eval()
        return final_loss
    
    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute learned score ∇log ρ(x) for given points.
        
        Args:
            x: Query points (batch_size, dim)
            
        Returns:
            score: Score vectors (batch_size, dim)
        """
        if self.score_net is None:
            raise RuntimeError("ScoreNet not trained. Call train() first.")
        
        # Ensure 2D shape
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        with torch.no_grad():
            return self.score_net(x)
    
    def _compute_target_term(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute -∇log ν(x) (negative target score).
        
        In analytical mode: (x - μ) / σ²  (Gaussian formula)
        In learned mode:    -TargetScoreTrainer.score(x)
        
        Args:
            x: Points at which to evaluate (batch_size, dim)
            
        Returns:
            target_term: -∇log ν(x), shape (batch_size, dim)
        """
        if self.target_score_mode == "learned":
            # Learned: -∇log ν(x) = -score_net(x)
            return -self.target_score_trainer.score(x)
        else:
            # Analytical Gaussian: -∇log ν(x) = (x - μ) / σ²
            return self.target_precision * (x - self.target_mean)
    
    def compute_terminal_gradient(
        self,
        x_terminal: torch.Tensor,
        y_clamp: float = 100.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute terminal gradient Y_N using DSM estimator.
        
        For MFC with terminal cost KL(ρ_N || ν):
            Y_N = λ * [∇log ρ_N(x) - ∇log ν(x)]
               = λ * [generated_score + target_term]
        
        where target_term = -∇log ν(x), computed analytically or via learned net.
        
        The ScoreNet for ρ_N is re-trained on x_terminal before computing.
        
        Args:
            x_terminal: Terminal states (batch_size, dim)
            y_clamp: Maximum absolute value for Y_N
            verbose: Whether to print debug statistics
            
        Returns:
            y_terminal: Terminal adjoint values (batch_size, dim), clamped
        """
        # Ensure 2D shape
        if x_terminal.dim() == 1:
            x_terminal = x_terminal.unsqueeze(-1)
        
        # Train ScoreNet on current batch (re-initialize each time)
        train_loss = self.train(x_terminal)
        
        if verbose:
            print(f"    [DSM] Trained ScoreNet: final_loss={train_loss:.6f}")
        
        # Compute learned score for generated density ∇log ρ_N(x)
        generated_score = self.compute_score(x_terminal)
        
        # Compute -∇log ν(x) (analytical or learned)
        target_term = self._compute_target_term(x_terminal)
        
        # Terminal Condition / Adjoint State Calculation: Y_N
        # Y_N = λ * (∇log ρ_N(x) - ∇log ν(x))
        #     = λ * (generated_score + target_term)
        y_terminal = self.terminal_weight * (target_term + generated_score)
        
        # Debug logging BEFORE clamping
        if verbose:
            y_mean = y_terminal.mean().item()
            y_std = y_terminal.std().item()
            y_min = y_terminal.min().item()
            y_max = y_terminal.max().item()
            x_mean = x_terminal.mean().item()
            x_std = x_terminal.std().item()
            score_mean = generated_score.mean().item()
            score_std = generated_score.std().item()
            target_mean_val = target_term.mean().item()
            target_std_val = target_term.std().item()
            print(f"    [DEBUG] X_N: μ={x_mean:.4f}, σ={x_std:.4f}")
            print(f"    [DEBUG] Score(ρ_N): μ={score_mean:.4f}, σ={score_std:.4f}")
            print(f"    [DEBUG] -Score(ν) [{self.target_score_mode}]: "
                  f"μ={target_mean_val:.4f}, σ={target_std_val:.4f}")
            print(f"    [DEBUG] Y_N (pre-clamp): μ={y_mean:.4f}, σ={y_std:.4f}, "
                  f"range=[{y_min:.2f}, {y_max:.2f}]")
        
        # Clamp Y_N to prevent gradient explosion
        y_terminal = torch.clamp(y_terminal, min=-y_clamp, max=y_clamp)
        
        if verbose:
            clipped_pct = ((y_terminal.abs() >= y_clamp - 0.01).float().mean() * 100).item()
            if clipped_pct > 5:
                print(f"    [WARNING] {clipped_pct:.1f}% of Y_N values clipped!")
        
        return y_terminal
