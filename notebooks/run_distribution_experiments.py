"""
MFC Generative Modeling — Distribution Experiments
===================================================

Runs three experiments using the current best configuration from config.yaml:
  I.  Dirac → Gaussian  N(1, 1)
  II. Dirac → Student-t  (df = 2)
  III.Dirac → Gaussian Mixture Model  (bimodal)

All training hyper-parameters (T, τ, λ, y_clamp, iterations, …) are loaded
from ../config.yaml so this script always uses the latest "best" settings.

Usage:
    cd notebooks/
    python run_distribution_experiments.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive: save to file, never block

import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup – make sure src/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import Config, IterativeSolver, ExperimentManager
from src.score_matching import ScoreEstimator

plt.style.use("seaborn-v0_8-paper")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True


# ===================================================================
# 1.  Custom Score Estimator for Non-Gaussian Targets
# ===================================================================
class GeneralScoreEstimator(ScoreEstimator):
    """
    Extended ScoreEstimator that supports arbitrary target distributions
    via a user-supplied score function  ∇log μ_T(x).
    """

    def __init__(self, target_score_fn, y_clamp: float = 5.0, **kwargs):
        # Base init with dummy target stats (not used for score calc)
        super().__init__(target_mean=0.0, target_std=1.0, **kwargs)
        self.target_score_fn = target_score_fn
        self._y_clamp = y_clamp          # store config's y_clamp

    def compute_terminal_gradient(
        self,
        x_terminal: torch.Tensor,
        y_clamp: float = 100.0,           # kept for API compat
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Y_N = λ * [-∇log μ_T(x)  +  ∇log ρ_N(x)]
        Uses self._y_clamp (from config) instead of the default argument.
        """
        clamp = self._y_clamp             # always use config value

        # 1. Train ScoreNet (DSM) on current batch
        _ = self.train(x_terminal)

        # 2. Compute learned score  ∇log ρ_N(x)
        generated_score = self.compute_score(x_terminal)

        # 3. Compute target term:  -∇log μ_T(x)
        score_target = self.target_score_fn(x_terminal)
        target_term = -score_target

        # 4. Combine
        y_terminal = self.terminal_weight * (target_term + generated_score)
        y_terminal = torch.clamp(y_terminal, -clamp, clamp)

        return y_terminal


# ===================================================================
# 2.  Target Distribution Definitions (Score & PDF)
# ===================================================================

# --- Gaussian ---
def gaussian_score(x, mean=1.0, std=1.0):
    return -(x - mean) / (std ** 2)


def gaussian_pdf(x, mean=1.0, std=1.0):
    return stats.norm.pdf(x, loc=mean, scale=std)


# --- Student-t ---
def student_t_score(x, df=2.0):
    return -(df + 1) * x / (df + x ** 2)


def student_t_pdf(x, df=2.0):
    return stats.t.pdf(x, df=df)


# --- Gaussian Mixture Model (GMM) ---
GMM_MEANS = [-2.0, 2.0]
GMM_STDS = [0.5, 0.5]
GMM_WEIGHTS = [0.5, 0.5]


def mix_gaussian_score(x, means=GMM_MEANS, stds=GMM_STDS, weights=GMM_WEIGHTS):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    densities, grads = [], []
    for mu, sigma, w in zip(means, stds, weights):
        sigma2 = sigma ** 2
        term = (x - mu) / sigma2
        density = w * torch.exp(-0.5 * (x - mu) ** 2 / sigma2) / (
            np.sqrt(2 * np.pi) * sigma
        )
        densities.append(density)
        grads.append(density * (-term))

    total_density = sum(densities) + 1e-10
    total_grad = sum(grads)
    return total_grad / total_density


def mix_gaussian_pdf(x, means=GMM_MEANS, stds=GMM_STDS, weights=GMM_WEIGHTS):
    total_pdf = np.zeros_like(x)
    for mu, sigma, w in zip(means, stds, weights):
        total_pdf += w * stats.norm.pdf(x, loc=mu, scale=sigma)
    return total_pdf


# ===================================================================
# 3.  Visualization Helper
# ===================================================================
def plot_experiment_results(solver, target_pdf_fn, title_keyword):
    """
    Plot (a) sample trajectories and (b) terminal distribution vs target PDF.
    """
    try:
        print(f"\n[Plotting] Generating visualisations for {title_keyword}…")
        with torch.no_grad():
            trajectories, _ = solver.dynamics.simulate(
                solver.target_networks, batch_size=5000, return_controls=False
            )

        if trajectories.dim() == 3 and trajectories.shape[-1] == 1:
            trajectories = trajectories.squeeze(-1)

        traj_np = trajectories.cpu().numpy()
        terminal = traj_np[-1, :]
        t_grid = solver.dynamics.time_grid.cpu().numpy()

        # — Figure 1 : Trajectories —
        plt.figure(figsize=(10, 6))
        plt.plot(t_grid, traj_np[:, :100], alpha=0.1, color="blue")
        plt.plot(
            t_grid,
            traj_np.mean(axis=1),
            color="red",
            linewidth=2,
            label="Mean Path",
        )
        plt.xlabel("Time")
        plt.ylabel("State $X_t$")
        plt.title(f"Trajectories: Dirac → {title_keyword}")
        plt.legend()
        plt.tight_layout()
        fname1 = f"trajectories_{title_keyword.replace(' ', '_')}.png"
        plt.savefig(fname1, dpi=150)
        print(f"  Saved → {fname1}")
        plt.close()

        # — Figure 2 : Terminal distribution vs target PDF —
        plt.figure(figsize=(10, 6))
        plt.hist(
            terminal,
            bins=50,
            density=True,
            alpha=0.6,
            color="skyblue",
            label="Generated (MFC)",
        )
        x_lo, x_hi = terminal.min() - 1, terminal.max() + 1
        x_grid = np.linspace(x_lo, x_hi, 500)
        plt.plot(
            x_grid,
            target_pdf_fn(x_grid),
            "r--",
            linewidth=2,
            label=f"Target PDF ({title_keyword})",
        )
        plt.title(f"Terminal Distribution Validation: {title_keyword}")
        plt.xlabel("State $X_T$")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        fname2 = f"distribution_{title_keyword.replace(' ', '_')}.png"
        plt.savefig(fname2, dpi=150)
        print(f"  Saved → {fname2}")
        plt.close()

        print("[Plotting] Done.")
    except Exception as exc:
        print(f"\n[Plotting Error] {exc}")
        import traceback
        traceback.print_exc()


# ===================================================================
# 4.  Experiment Runner  (reads everything from config.yaml)
# ===================================================================
def run_experiment(name, target_score_fn, config, target_mean=0.0, target_std=1.0):
    """
    Run a single experiment.  All training hyper-parameters come from *config*
    (which is loaded from config.yaml), so there is no hard-coding.
    """
    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  T={config.physics.T}, τ={config.training.tau}, "
          f"λ={config.training.terminal_weight}, y_clamp={config.training.y_clamp}")
    print(f"  iterations={config.training.iterations}, "
          f"batch_size={config.training.batch_size}")
    print(f"{'='*70}")

    # Override target params (only affects logging / Gaussian baseline)
    config.target.mean = target_mean
    config.target.std = target_std
    config.experiment.save_plots = False
    config.experiment.name = f"exp_{name}"

    # Init solver
    exp_manager = ExperimentManager(config)
    solver = IterativeSolver(config, exp_manager, tau=config.training.tau)

    # Inject custom score estimator (with y_clamp from config)
    solver.score_estimator = GeneralScoreEstimator(
        target_score_fn=target_score_fn,
        y_clamp=config.training.y_clamp,
        device=solver.device,
        terminal_weight=config.training.terminal_weight,
        hidden_dim=config.model.hidden_dim,
    )

    # Train
    solver.run()

    # ---- Print final stats ----
    with torch.no_grad():
        trajs, _ = solver.dynamics.simulate(
            solver.target_networks, batch_size=5000, return_controls=False
        )
    if trajs.dim() == 3 and trajs.shape[-1] == 1:
        trajs = trajs.squeeze(-1)
    terminal = trajs[-1].cpu().numpy()
    print(f"\n  ▸ Final Mean = {terminal.mean():.4f}")
    print(f"  ▸ Final Std  = {terminal.std():.4f}")

    return solver


# ===================================================================
# 5.  Main — Run All Three Experiments
# ===================================================================
if __name__ == "__main__":

    # Load the single source of truth
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    base_config = Config.from_yaml(config_path)

    print("=" * 70)
    print("  MFC Distribution Experiments")
    print(f"  Config: T={base_config.physics.T}, N={base_config.physics.N}, "
          f"σ={base_config.physics.sigma}")
    print(f"  Training: iterations={base_config.training.iterations}, "
          f"τ={base_config.training.tau}, λ={base_config.training.terminal_weight}, "
          f"y_clamp={base_config.training.y_clamp}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Experiment I  –  Gaussian N(1, 1)
    # ------------------------------------------------------------------
    import copy
    cfg1 = copy.deepcopy(base_config)
    solver_gauss = run_experiment(
        name="Gaussian",
        target_score_fn=lambda x: gaussian_score(x, mean=1.0, std=1.0),
        config=cfg1,
        target_mean=1.0,
        target_std=1.0,
    )
    plot_experiment_results(
        solver_gauss,
        lambda x: gaussian_pdf(x, mean=1.0, std=1.0),
        "Gaussian N(1,1)",
    )

    # ------------------------------------------------------------------
    # Experiment II  –  Student-t  (df = 2)
    # ------------------------------------------------------------------
    cfg2 = copy.deepcopy(base_config)
    solver_t = run_experiment(
        name="Student-t",
        target_score_fn=lambda x: student_t_score(x, df=2.0),
        config=cfg2,
        target_mean=0.0,
        target_std=1.0,
    )
    plot_experiment_results(
        solver_t,
        lambda x: student_t_pdf(x, df=2.0),
        "Student-t (df=2)",
    )

    # ------------------------------------------------------------------
    # Experiment III  –  Gaussian Mixture Model
    # ------------------------------------------------------------------
    cfg3 = copy.deepcopy(base_config)
    solver_gmm = run_experiment(
        name="GMM",
        target_score_fn=mix_gaussian_score,
        config=cfg3,
        target_mean=0.0,
        target_std=1.0,
    )
    plot_experiment_results(
        solver_gmm,
        mix_gaussian_pdf,
        "GMM",
    )

    print("\n✅  All experiments complete.")
