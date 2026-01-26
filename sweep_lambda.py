
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src import Config
from main import main

def run_parameter_sweep():
    print("========================================================")
    print("STARTING PARAMETER SWEEP: terminal_weight 1.0 -> 10.0")
    print("========================================================")
    
    lambdas = np.arange(1.0, 11.0, 1.0)
    
    # Load base config
    base_config = Config.from_yaml("config.yaml")
    
    # Create directory for aggregated plots
    summary_dir = Path("experiments") / "sweep_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    for val in lambdas:
        print(f"\n>>> Running with terminal_weight = {val}")
        
        # Modify config
        config = base_config
        config.training.terminal_weight = float(val)
        config.experiment.name = f"sweep_lambda_{val:.1f}"
        
        # Run experiment
        solver, stats = main(config_obj=config, show_plots=False)
        
        # Copy the terminal distribution plot to summary
        exp_dir = solver.exp_manager.exp_dir
        src_plot = exp_dir / "terminal_distribution.png"
        dst_plot = summary_dir / f"terminal_dist_lambda_{val:.1f}.png"
        
        if src_plot.exists():
            shutil.copy(src_plot, dst_plot)
            print(f"Saved plot copy to: {dst_plot}")
        else:
            print("Warning: Plot not found!")

    print("\n========================================================")
    print("SWEEP COMPLETE")
    print(f"Summary plots saved to: {summary_dir}")
    print("========================================================")

if __name__ == "__main__":
    run_parameter_sweep()
