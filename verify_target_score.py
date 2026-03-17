"""
Verification Script: Target Score Network
==========================================

Tests that the TargetScoreTrainer correctly learns ∇log ν(x) via DSM
and that the refactored ScoreEstimator produces consistent results 
in both "analytical" and "learned" modes.

Author: Lizhan HONG
"""

import torch
import numpy as np
import sys

# Ensure we can import from parent
sys.path.insert(0, '.')

from src.score_matching import TargetScoreTrainer, ScoreEstimator


def test_gaussian_score_learning():
    """
    Test 1: Train TargetScoreTrainer on Gaussian samples N(μ, σ²)
    and verify the learned score matches -(x-μ)/σ².
    """
    print("=" * 60)
    print("Test 1: Gaussian Score Learning")
    print("=" * 60)
    
    device = torch.device('cpu')
    mu, sigma = 2.0, 0.5
    dim = 1
    n_samples = 10000
    
    # Generate Gaussian samples
    target_samples = torch.randn(n_samples, dim) * sigma + mu
    
    # Train TargetScoreTrainer
    trainer = TargetScoreTrainer(
        dim=dim,
        device=device,
        hidden_dim=128,
        n_pretrain_steps=2000,
        lr=0.001,
        sigma_dsm=0.05,
    )
    final_loss = trainer.train(target_samples, verbose=True)
    
    # Evaluate on test points
    x_test = torch.linspace(mu - 3*sigma, mu + 3*sigma, 200).unsqueeze(-1)
    
    # Analytical score: -(x - μ) / σ²
    analytical_score = -(x_test - mu) / (sigma ** 2)
    
    # Learned score
    learned_score = trainer.score(x_test)
    
    # Compute error
    mse = ((learned_score - analytical_score) ** 2).mean().item()
    max_err = (learned_score - analytical_score).abs().max().item()
    
    print(f"\n  Results:")
    print(f"    MSE(learned vs analytical): {mse:.6f}")
    print(f"    Max absolute error:         {max_err:.4f}")
    print(f"    Analytical score range:     [{analytical_score.min():.2f}, {analytical_score.max():.2f}]")
    print(f"    Learned score range:        [{learned_score.min():.2f}, {learned_score.max():.2f}]")
    
    passed = mse < 1.0  # Generous threshold for simple DSM
    print(f"    PASS: {passed}")
    print()
    return passed


def test_gmm_score_no_nan():
    """
    Test 2: Train on a Gaussian Mixture Model and verify no NaN/Inf.
    """
    print("=" * 60)
    print("Test 2: GMM Score (no NaN/Inf)")
    print("=" * 60)
    
    device = torch.device('cpu')
    dim = 1
    n_samples = 10000
    
    # Generate GMM samples: 50% N(-2, 0.5²) + 50% N(2, 0.5²)
    n_half = n_samples // 2
    samples_1 = torch.randn(n_half, dim) * 0.5 - 2.0
    samples_2 = torch.randn(n_samples - n_half, dim) * 0.5 + 2.0
    target_samples = torch.cat([samples_1, samples_2], dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    target_samples = target_samples[perm]
    
    # Train
    trainer = TargetScoreTrainer(
        dim=dim,
        device=device,
        hidden_dim=128,
        n_pretrain_steps=2000,
        lr=0.001,
        sigma_dsm=0.1,
    )
    final_loss = trainer.train(target_samples, verbose=True)
    
    # Evaluate
    x_test = torch.linspace(-5, 5, 300).unsqueeze(-1)
    learned_score = trainer.score(x_test)
    
    has_nan = torch.isnan(learned_score).any().item()
    has_inf = torch.isinf(learned_score).any().item()
    
    print(f"\n  Results:")
    print(f"    Final DSM loss: {final_loss:.6f}")
    print(f"    Score range:    [{learned_score.min():.2f}, {learned_score.max():.2f}]")
    print(f"    Has NaN: {has_nan}")
    print(f"    Has Inf: {has_inf}")
    
    passed = not has_nan and not has_inf
    print(f"    PASS: {passed}")
    print()
    return passed


def test_analytical_vs_learned_terminal():
    """
    Test 3: Compare terminal gradient from analytical and learned modes
    for a Gaussian target. They should produce similar Y_N values.
    """
    print("=" * 60)
    print("Test 3: Analytical vs Learned Terminal Gradient")
    print("=" * 60)
    
    device = torch.device('cpu')
    mu, sigma = 1.0, 1.0
    dim = 1
    terminal_weight = 10.0
    
    # Generate "terminal particles" (pretend these are X_N)
    n_particles = 2048
    x_terminal = torch.randn(n_particles, dim) * 1.2 + 0.8  # Roughly near target
    
    # --- Analytical mode ---
    estimator_analytical = ScoreEstimator(
        target_mean=mu,
        target_std=sigma,
        device=device,
        terminal_weight=terminal_weight,
        target_score_mode="analytical",
    )
    y_analytical = estimator_analytical.compute_terminal_gradient(
        x_terminal.clone(), y_clamp=100.0, verbose=False
    )
    
    # --- Learned mode ---
    # Pre-train target score on Gaussian samples
    target_samples = torch.randn(10000, dim) * sigma + mu
    trainer = TargetScoreTrainer(
        dim=dim,
        device=device,
        hidden_dim=128,
        n_pretrain_steps=2000,
        lr=0.001,
        sigma_dsm=0.1,
    )
    trainer.train(target_samples, verbose=True)
    
    estimator_learned = ScoreEstimator(
        target_mean=mu,
        target_std=sigma,
        device=device,
        terminal_weight=terminal_weight,
        target_score_mode="learned",
        target_score_trainer=trainer,
    )
    y_learned = estimator_learned.compute_terminal_gradient(
        x_terminal.clone(), y_clamp=100.0, verbose=False
    )
    
    # Compare
    diff_mean = (y_analytical.mean() - y_learned.mean()).abs().item()
    diff_std = (y_analytical.std() - y_learned.std()).abs().item()
    mse = ((y_analytical - y_learned) ** 2).mean().item()
    
    print(f"\n  Results:")
    print(f"    Y_N analytical: μ={y_analytical.mean():.4f}, σ={y_analytical.std():.4f}")
    print(f"    Y_N learned:    μ={y_learned.mean():.4f}, σ={y_learned.std():.4f}")
    print(f"    |Δμ|: {diff_mean:.4f}")
    print(f"    |Δσ|: {diff_std:.4f}")
    print(f"    MSE(Y_analytical, Y_learned): {mse:.4f}")
    
    # The learned score won't be perfect, but should be in the same ballpark
    passed = diff_mean < 5.0 and mse < 50.0
    print(f"    PASS: {passed}")
    print()
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VERIFICATION: Target Score Network (DSM)")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = []
    results.append(("Gaussian Score Learning", test_gaussian_score_learning()))
    results.append(("GMM Score (no NaN/Inf)", test_gmm_score_no_nan()))
    results.append(("Analytical vs Learned", test_analytical_vs_learned_terminal()))
    
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("  All tests passed!")
    else:
        print("  Some tests failed!")
        sys.exit(1)
