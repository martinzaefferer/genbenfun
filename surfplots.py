from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bbob_disturbance_experiments import Problem, SineDisturbance


def evaluate_grid_2d(problem: Problem,
                     disturbance: Optional[SineDisturbance] = None,
                     n_grid: int = 200,
                     center_on_opt: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a problem (with optional disturbance) on a 2D grid.

    Returns X, Y, Z suitable for contourf, where problem.dim == 2.
    """
    assert problem.dim == 2, "evaluate_grid_2d only valid for dim=2"

    lo, hi = problem.bounds
    xopt = problem.xopt

    if center_on_opt:
        # Grid centered at optimum, clipped to bounds
        width = hi - lo
        radius = 0.5 * width  # show most of the domain
        x_min = np.maximum(lo, xopt - radius)
        x_max = np.minimum(hi, xopt + radius)
    else:
        x_min, x_max = lo, hi

    xs = np.linspace(x_min[0], x_max[0], n_grid)
    ys = np.linspace(x_min[1], x_max[1], n_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = np.empty_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([X[i, j], Y[i, j]], dtype=float)
            base_f = problem.f(x)
            if disturbance is not None:
                Z[i, j] = disturbance.apply(x, base_f, problem.fopt, problem.xopt)
            else:
                Z[i, j] = base_f

    return X, Y, Z


def evaluate_surrogate_grid_2d(gp_dict, problem: Problem, n_grid: int = 200, center_on_opt: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a fitted GP surrogate on a 2D grid.
    Returns X, Y, Z suitable for contourf, where problem.dim == 2.
    """
    assert problem.dim == 2, "evaluate_surrogate_grid_2d only valid for dim=2"
    
    lo, hi = problem.bounds
    xopt = problem.xopt

    if center_on_opt:
        # Grid centered at optimum, clipped to bounds
        width = hi - lo
        radius = 0.5 * width  # show most of the domain
        x_min = np.maximum(lo, xopt - radius)
        x_max = np.minimum(hi, xopt + radius)
    else:
        x_min, x_max = lo, hi

    xs = np.linspace(x_min[0], x_max[0], n_grid)
    ys = np.linspace(x_min[1], x_max[1], n_grid)
    X, Y = np.meshgrid(xs, ys)
    
    Z = np.empty_like(X)
    predict_mean = gp_dict["predict_mean"]
    
    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([X[i, j], Y[i, j]], dtype=float)
            Z[i, j] = predict_mean(x)
    
    return X, Y, Z


def evaluate_surrogate_slice(
    gp_dict,
    problem: Problem,
    n_grid: int = 200,
    slice_radius: float = 0.5,
    v1: np.ndarray | None = None,
    v2: np.ndarray | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a fitted GP surrogate on a 2D slice through a high-D problem,
    centered at xopt. Returns U, V, Z suitable for contourf.
    """
    d = problem.dim
    lo, hi = problem.bounds
    xopt = problem.xopt

    if v1 is None or v2 is None:
        # Generate two random orthonormal directions
        rng = np.random.default_rng(seed)
        v1 = rng.normal(size=d)
        v1 /= np.linalg.norm(v1)
        v2 = rng.normal(size=d)
        v2 -= v2.dot(v1) * v1
        v2 /= np.linalg.norm(v2)

    width = hi - lo
    base_radius = slice_radius * float(np.min(width))

    u = np.linspace(-base_radius, base_radius, n_grid)
    v = np.linspace(-base_radius, base_radius, n_grid)
    U, V = np.meshgrid(u, v)

    Z = np.empty_like(U)
    predict_mean = gp_dict["predict_mean"]

    for i in range(n_grid):
        for j in range(n_grid):
            x = xopt + U[i, j] * v1 + V[i, j] * v2
            x = problem.clip(x)  # keep in bounds
            Z[i, j] = predict_mean(x)

    return U, V, Z

def plot_true_disturbed_surrogate_comparison(
    problem: Problem,
    disturbance: SineDisturbance,
    gp_dict,
    path_df: Optional[pd.DataFrame] = None,
    n_grid: int = 200,
    output_path: Optional[str] = None,
    dpi: int = 100,
    slice_radius: float = 1.0,
    slice_seed: int = 0,
    amp_multiplier: float | None = None,
    gp_rmse_true: Optional[float] = None,
    gp_rmse_disturbed: Optional[float] = None,
):
    """
    Create a 1×3 subplot figure showing:
    1. True (undisturbed) objective
    2. Disturbed objective
    3. GP surrogate
    
    For 2D problems: uses 2D grid in decision space.
    For dim>2: uses the same random 2D slice for all three plots.
    
    Optionally overlay optimization paths from path_df with columns:
      [run_id, eval, raw_f, perturbed_f, x0, x1, ..., branch]
    where branch = "true" or "gp" to distinguish the two optimization runs.
    For 2D case only.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Extract paths by branch for legend
    path_df_true = None
    path_df_gp = None
    if path_df is not None and not path_df.empty:
        if "branch" in path_df.columns:
            path_df_true = path_df[path_df["branch"] == "true"].sort_values("eval")
            path_df_gp = path_df[path_df["branch"] == "gp"].sort_values("eval")
        else:
            # Legacy: no branch column, treat as single path
            path_df = path_df.sort_values("eval")
            path_df_true = path_df
    
    if problem.dim == 2:
        # ========== 2D case: direct grid evaluation ==========
        # Plot 1: True objective
        X, Y, Z_true = evaluate_grid_2d(problem, disturbance=None, n_grid=n_grid, center_on_opt=False)
        cs1 = axes[0].contourf(X, Y, Z_true, levels=n_grid//10, cmap="viridis")
        fig.colorbar(cs1, ax=axes[0])
        axes[0].plot(problem.xopt[0], problem.xopt[1], "rx", markersize=8, label="optimum", zorder=5)
        
        # Overlay paths
        if path_df_true is not None and not path_df_true.empty:
            axes[0].plot(path_df_true["x0"], path_df_true["x1"], "c.-", linewidth=1.5, 
                        markersize=4, alpha=0.8, label="true optimizer path", zorder=4)
        if path_df_gp is not None and not path_df_gp.empty:
            axes[0].plot(path_df_gp["x0"], path_df_gp["x1"], "y.-", linewidth=1.5, 
                        markersize=4, alpha=0.8, label="GP optimizer path", zorder=4)
        
        axes[0].set_xlabel("x0")
        axes[0].set_ylabel("x1")
        axes[0].set_title("True Objective")
        axes[0].legend(loc="best", fontsize="small")
        
        # Plot 2: Disturbed objective
        X, Y, Z_dist = evaluate_grid_2d(problem, disturbance=disturbance, n_grid=n_grid, center_on_opt=False)
        cs2 = axes[1].contourf(X, Y, Z_dist, levels=n_grid//10, cmap="viridis")
        fig.colorbar(cs2, ax=axes[1])
        axes[1].plot(problem.xopt[0], problem.xopt[1], "rx", markersize=8, label="optimum", zorder=5)
        
        # Overlay paths
        if path_df_true is not None and not path_df_true.empty:
            axes[1].plot(path_df_true["x0"], path_df_true["x1"], "c.-", linewidth=1.5, 
                        markersize=4, alpha=0.8, label="true optimizer path", zorder=4)
        if path_df_gp is not None and not path_df_gp.empty:
            axes[1].plot(path_df_gp["x0"], path_df_gp["x1"], "y.-", linewidth=1.5, 
                        markersize=4, alpha=0.8, label="GP optimizer path", zorder=4)
        
        axes[1].set_xlabel("x0")
        axes[1].set_ylabel("x1")
        # Show both configured multiplier and the actual linear amplitude
        A_log10 = float(disturbance.amplitude)
        A_lin = 10 ** A_log10
        #if amp_multiplier is not None:
        axes[1].set_title(f"Disturbed Objective (amp_mult={amp_multiplier}, A={A_lin:.3g}, freq={disturbance.frequency:.2f})")
        #else:
        #    axes[1].set_title(f"Disturbed Objective (amp_log10={disturbance.amplitude:.2f}, freq={disturbance.frequency:.2f})")
        axes[1].legend(loc="best", fontsize="small")
        
        # Plot 3: Surrogate
        X, Y, Z_surr = evaluate_surrogate_grid_2d(gp_dict, problem, n_grid=n_grid, center_on_opt=False)
        cs3 = axes[2].contourf(X, Y, Z_surr, levels=n_grid//10, cmap="viridis")
        fig.colorbar(cs3, ax=axes[2])
        axes[2].plot(problem.xopt[0], problem.xopt[1], "rx", markersize=8, label="optimum", zorder=5)
        
        # Overlay GP path only on surrogate plot (this is what the GP optimizer optimizes)
        if path_df_gp is not None and not path_df_gp.empty:
            axes[2].plot(path_df_gp["x0"], path_df_gp["x1"], "y.-", linewidth=1.5, 
                        markersize=4, alpha=0.8, label="GP optimizer path", zorder=4)
        
        axes[2].set_xlabel("x0")
        axes[2].set_ylabel("x1")
        
        # Build surrogate title with RMSE metrics
        surrogate_title = "GP Surrogate"
        if gp_rmse_true is not None and gp_rmse_disturbed is not None:
            surrogate_title += f"\n(RMSE_true={gp_rmse_true:.3g}, RMSE_disturbed={gp_rmse_disturbed:.3g})"
        
        axes[2].set_title(surrogate_title)
        axes[2].legend(loc="best", fontsize="small")
    
    else:
        # ========== High-D case: random 2D slice ==========
        # Generate slice directions once, use for all three plots
        rng = np.random.default_rng(slice_seed)
        d = problem.dim
        v1 = rng.normal(size=d)
        v1 /= np.linalg.norm(v1)
        v2 = rng.normal(size=d)
        v2 -= v2.dot(v1) * v1
        v2 /= np.linalg.norm(v2)
        
        # Plot 1: True objective
        U, V, Z_true = evaluate_slice(
            problem,
            disturbance=None,
            n_grid=n_grid,
            slice_radius=slice_radius,
            v1=v1,
            v2=v2,
            seed=slice_seed,
        )
        cs1 = axes[0].contourf(U, V, Z_true, levels=n_grid//10, cmap="viridis")
        fig.colorbar(cs1, ax=axes[0])
        axes[0].plot(0, 0, "rx", markersize=8, label="optimum")
        axes[0].set_xlabel("slice coord 1")
        axes[0].set_ylabel("slice coord 2")
        axes[0].set_title("True Objective")
        
        # Plot 2: Disturbed objective
        U, V, Z_dist = evaluate_slice(
            problem,
            disturbance=disturbance,
            n_grid=n_grid,
            slice_radius=slice_radius,
            v1=v1,
            v2=v2,
            seed=slice_seed,
        )
        cs2 = axes[1].contourf(U, V, Z_dist, levels=n_grid//10, cmap="viridis")
        fig.colorbar(cs2, ax=axes[1])
        axes[1].plot(0, 0, "rx", markersize=8, label="optimum")
        axes[1].set_xlabel("slice coord 1")
        axes[1].set_ylabel("slice coord 2")
        A_log10 = float(disturbance.amplitude)
        A_lin = 10 ** A_log10
        if amp_multiplier is not None:
            axes[1].set_title(f"Disturbed Objective (amp_mult={amp_multiplier}, A={A_lin:.3g}, freq={disturbance.frequency:.2f})")
        else:
            axes[1].set_title(f"Disturbed Objective (amp_log10={disturbance.amplitude:.2f}, freq={disturbance.frequency:.2f})")
        
        # Plot 3: Surrogate
        U, V, Z_surr = evaluate_surrogate_slice(
            gp_dict,
            problem,
            n_grid=n_grid,
            slice_radius=slice_radius,
            v1=v1,
            v2=v2,
            seed=slice_seed,
        )
        cs3 = axes[2].contourf(U, V, Z_surr, levels=n_grid//10, cmap="viridis")
        fig.colorbar(cs3, ax=axes[2])
        axes[2].plot(0, 0, "rx", markersize=8, label="optimum")
        axes[2].set_xlabel("slice coord 1")
        axes[2].set_ylabel("slice coord 2")
        
        # Build surrogate title with RMSE metrics
        surrogate_title = "GP Surrogate"
        if gp_rmse_true is not None and gp_rmse_disturbed is not None:
            surrogate_title += f"\n(RMSE_true={gp_rmse_true:.3g}, RMSE_disturbed={gp_rmse_disturbed:.3g})"
        
        axes[2].set_title(surrogate_title)
    
    fig.suptitle(f"{problem.name} Comparison (dim={problem.dim})", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig, axes

def evaluate_slice(
    problem,
    disturbance=None,
    n_grid: int = 200,
    slice_radius: float = 0.5,
    v1: np.ndarray | None = None,
    v2: np.ndarray | None = None,
    seed: int = 0,
):
    """
    Evaluate a 2D slice through a high-D problem, centered at xopt.
    Returns U, V, Z where U,V are meshgrid coordinates in slice space.
    """
    d = problem.dim
    lo, hi = problem.bounds
    xopt = problem.xopt

    if v1 is None or v2 is None:
        # Generate two random orthonormal directions
        rng = np.random.default_rng(seed)
        v1 = rng.normal(size=d)
        v1 /= np.linalg.norm(v1)
        v2 = rng.normal(size=d)
        v2 -= v2.dot(v1) * v1
        v2 /= np.linalg.norm(v2)

    width = hi - lo
    base_radius = slice_radius * float(np.min(width))

    u = np.linspace(-base_radius, base_radius, n_grid)
    v = np.linspace(-base_radius, base_radius, n_grid)
    U, V = np.meshgrid(u, v)

    Z = np.empty_like(U)

    for i in range(n_grid):
        for j in range(n_grid):
            x = xopt + U[i, j] * v1 + V[i, j] * v2
            x = problem.clip(x)  # keep in bounds
            base_f = problem.f(x)
            if disturbance is not None:
                Z[i, j] = disturbance.apply(x, base_f, problem.fopt, problem.xopt)
            else:
                Z[i, j] = base_f

    return U, V, Z
