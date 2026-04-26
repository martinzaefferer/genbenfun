from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, Any, List, Optional
from datetime import datetime
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import gzip
import sys
import importlib


# ========== Optional dependencies detection ==========
def _try_import(module_name):
    """Safely import optional packages."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        print(f"Warning: {module_name} not available", file=sys.stderr)
        return None

scipy_opt = _try_import("scipy.optimize")
cma_pkg = _try_import("cma")
cocoex = _try_import("cocoex")

def _try_import_gpytorch():
    try:
        import torch
        import gpytorch
        from gpytorch.models import ExactGP
        from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean
        from gpytorch.distributions import MultivariateNormal
        from gpytorch.likelihoods import GaussianLikelihood
        return torch, gpytorch, ExactGP, RBFKernel, MaternKernel, ScaleKernel, ConstantMean, GaussianLikelihood, MultivariateNormal
    except Exception:
        return None

_GPYTORCH = _try_import_gpytorch()

# ========== Utilities ==========
@dataclass
class Problem:
    name: str
    dim: int
    bounds: Tuple[np.ndarray, np.ndarray]
    f: Callable[[np.ndarray], float]
    xopt: np.ndarray
    fopt: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clip(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds
        return np.minimum(np.maximum(x, lo), hi)

def rastrigin(x: np.ndarray) -> float:
    A = 10
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def ackley(x: np.ndarray) -> float:
    # Standard Ackley with a=20, b=0.2, c=2π
    a, b, c = 20, 0.2, 2*np.pi
    d = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c*x))
    term1 = -a * np.exp(-b*np.sqrt(s1/d))
    term2 = -np.exp(s2/d)
    return float(term1 + term2 + a + np.e)

# ========== Disturbance wrapper ==========
@dataclass
class SineDisturbance:
    amplitude: float
    frequency: float
    mode: str = "sin_x"  #or sin_f
    # "sin_f" keeps the optimum location (nonnegative disturbance)
    # perturbation is: amplitude * sin^2(2*pi*freq * (f(x) - fopt))
    # else for "sin_x":
    # perturbation is: amplitude * (1-cos(2*pi*freq * (||x - xopt||)))


    def apply(self, base_x: float, base_f: float, fopt: float, xopt: float) -> float:
        if self.mode == "sin_f":
            arg = 2*np.pi*(10**self.frequency)*(base_f - fopt)
            return base_f + (10**self.amplitude) * (np.sin(arg)**2)
        else:
            r =  np.linalg.norm(base_x - xopt)
            arg = 2*np.pi*(10**self.frequency)*r
            return base_f + (10**self.amplitude) * (1-np.cos(arg))

def create_gzipped_paths_writer(paths_path: str, max_dim: int) -> Tuple[csv.DictWriter, gzip.GzipFile]:
    """
    Create a gzipped CSV writer for paths.

    Columns:
      run_id, eval, raw_f, perturbed_f, x0..x{max_dim-1}
    """
    path = Path(paths_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = ["run_id", "eval", "raw_f", "perturbed_f"]
    x_cols = [f"x{i}" for i in range(max_dim)]
    fieldnames = base_cols + x_cols

    gz_file = gzip.open(path, mode="wt", newline="")
    writer = csv.DictWriter(gz_file, fieldnames=fieldnames)
    writer.writeheader()

    return writer, gz_file


class CSVPathLogger:
    """
    Streaming logger that acts as an objective function.

    - Computes raw and disturbed f(x)
    - Tracks best_so_far in memory (small)
    - Writes (run_id, eval, raw_f, perturbed_f, x0..x{max_dim-1}) to CSV
    - Keeps in-memory buffer for fast access

    If log_only_improvements=True, only evaluations that improve the
    best perturbed_f so far are written.
    """
    def __init__(
        self,
        problem,
        disturbance,
        budget: int,
        writer: csv.DictWriter,
        run_id: int,
        max_dim: int,
        flush_every: int = 1000,
        log_only_improvements: bool = False,   # NEW
    ):
        self.problem = problem
        self.disturbance = disturbance
        self.budget = budget
        self.writer = writer
        self.run_id = run_id
        self.max_dim = max_dim
        self.flush_every = flush_every
        self.log_only_improvements = log_only_improvements

        self.n_evals = 0
        self.best_f = float("inf")          # best perturbed_f so far
        self.best_x: Optional[np.ndarray] = None

        self._buffer: List[Dict[str, Any]] = []
        self._memory_buffer: List[Dict[str, Any]] = []  # Keep all logged rows in memory

        # cache bounds for clipping
        self.lo, self.hi = self.problem.bounds

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)

        # clip to bounds for evaluation + logging
        x_clipped = np.minimum(np.maximum(x, self.lo), self.hi)

        raw = self.problem.f(x_clipped)
        val = raw
        if self.disturbance is not None:
            val = self.disturbance.apply(x_clipped, raw, self.problem.fopt, self.problem.xopt)

        self.n_evals += 1

        improved = val < self.best_f

        if improved:
            self.best_f = float(val)
            self.best_x = x_clipped.copy()

        # Decide whether to log this evaluation
        if ((not self.log_only_improvements) or improved) and (self.n_evals <= self.budget):
            row: Dict[str, Any] = {
                "run_id": self.run_id,
                "eval": self.n_evals,
                "raw_f": float(raw),
                "perturbed_f": float(val),
            }
            for i in range(self.max_dim):
                if i < len(x_clipped):
                    row[f"x{i}"] = float(x_clipped[i])
                else:
                    row[f"x{i}"] = ""

            self._buffer.append(row)
            self._memory_buffer.append(row.copy())  # Keep copy in memory
            if len(self._buffer) >= self.flush_every:
                self.flush()

        return float(val)

    def flush(self):
        if not self._buffer:
            return
        self.writer.writerows(self._buffer)
        self._buffer.clear()

    def close(self):
        self.flush()

    def get_path_dataframe(self) -> pd.DataFrame:
        """Return accumulated path data as DataFrame."""
        if not self._memory_buffer:
            return pd.DataFrame()
        return pd.DataFrame(self._memory_buffer)

class CSVSurrogatePathLogger:
    """
    For surrogate optimization, logs:
      raw_f        := surrogate mean prediction
      perturbed_f  := true disturbed objective at x (reference)
    Writes rows into the GP paths file (Option A).
    Keeps in-memory buffer for fast access.
    """
    def __init__(
        self,
        problem,
        disturbance,
        surrogate_predict,
        budget,
        writer,
        run_id,
        max_dim,
        flush_every=1000,
        log_only_improvements=False,
    ):
        #import numpy as np
        #self.np = np
        self.problem = problem
        self.disturbance = disturbance
        self.surrogate_predict = surrogate_predict
        self.budget = budget
        self.writer = writer
        self.run_id = run_id
        self.max_dim = max_dim
        self.flush_every = flush_every
        self.log_only_improvements = log_only_improvements

        self.n_evals = 0
        self.best_f = float("inf")
        self.best_x = None
        self._buffer = []
        self._memory_buffer = []  # Keep all logged rows in memory

        self.lo, self.hi = self.problem.bounds

    def __call__(self, x):
        #np = self.np
        x = np.asarray(x, dtype=float)
        x = np.minimum(np.maximum(x, self.lo), self.hi)

        # value minimized by optimizer
        surr = float(self.surrogate_predict(x))

        # reference true disturbed value at same x
        raw = self.problem.f(x)
        true_dist = raw
        if self.disturbance is not None:
            true_dist = self.disturbance.apply(x, raw, self.problem.fopt, self.problem.xopt)

        self.n_evals += 1
        improved = surr < self.best_f
        if improved:
            self.best_f = surr
            self.best_x = x.copy()

        if self.n_evals <= self.budget and ((not self.log_only_improvements) or improved):
            row = {
                "run_id": self.run_id,
                "eval": self.n_evals,
                "raw_f": float(surr),
                "perturbed_f": float(true_dist),
            }
            for i in range(self.max_dim):
                row[f"x{i}"] = float(x[i]) if i < len(x) else ""
            self._buffer.append(row)
            self._memory_buffer.append(row.copy())  # Keep copy in memory

            if len(self._buffer) >= self.flush_every:
                self.flush()

        return surr

    def flush(self):
        if self._buffer:
            self.writer.writerows(self._buffer)
            self._buffer.clear()

    def close(self):
        self.flush()

    def get_path_dataframe(self) -> pd.DataFrame:
        """Return accumulated path data as DataFrame."""
        if not self._memory_buffer:
            return pd.DataFrame()
        return pd.DataFrame(self._memory_buffer)

# ===================== Surrogate modeling (Gaussian Process with GPyTorch) =====================
def latin_hypercube(rng, n: int, d: int):
    """Simple LHS in [0,1]^d with no external deps."""
    #import numpy as np
    X = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return X


@dataclass
class GPSurrogateConfig:
    # training set size heuristic: n_train_per_dim * dim
    n_train_per_dim: int = 50
    design: str = "lhs"  # "lhs" or "random"
    seed: int = 0

    # kernel choice
    kernel: str = "rbf"  # "rbf" or "matern52"

    # numeric stability / smoothing knobs
    alpha: float = 1e-10
    add_white_kernel: bool = True
    white_noise_level: float = 1e-4

    # GP options
    normalize_y: bool = True
    niter_optim_per_dim: int = 100


def fit_gp_surrogate(problem, disturbance, cfg: GPSurrogateConfig) -> Dict[str, Any]:
    """
    Fit GP to the *disturbed* objective values using GPyTorch.
    Returns dict with: model, predict_mean(x), X_train, y_train, n_train
    """
    if _GPYTORCH is None:
        raise RuntimeError(
            "GPyTorch and/or torch not available. "
            "Install torch and gpytorch to use GP surrogates. "
            "pip install torch gpytorch"
        )
    torch, gpytorch, ExactGP, RBFKernel, MaternKernel, ScaleKernel, ConstantMean, GaussianLikelihood, MultivariateNormal = _GPYTORCH

    lo, hi = problem.bounds
    d = problem.dim
    n_train = int(cfg.n_train_per_dim * d)

    rng = np.random.default_rng(cfg.seed)

    if cfg.design == "lhs":
        U = latin_hypercube(rng, n_train, d)
    elif cfg.design == "random":
        U = rng.random((n_train, d))
    else:
        raise ValueError(f"Unknown design: {cfg.design}")

    X = lo + (hi - lo) * U

    y = np.empty(n_train, dtype=float)
    for i in range(n_train):
        x = X[i]
        raw = problem.f(x)
        y[i] = disturbance.apply(x, raw, problem.fopt, problem.xopt) if disturbance is not None else raw

    # Convert to torch tensors
    X_train = torch.from_numpy(X.astype(np.float32)).float()
    y_train = torch.from_numpy(y.astype(np.float32)).float()

    # Normalize targets
    if cfg.normalize_y:
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std < 1e-10:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std
    else:
        y_mean = torch.tensor(0.0, dtype=torch.float32)
        y_std = torch.tensor(1.0, dtype=torch.float32)
        y_train_norm = y_train

    # Define the GP model
    class ExactGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel_type):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            
            if kernel_type == "rbf":
                base_kernel = RBFKernel()
            elif kernel_type == "matern52":
                base_kernel = MaternKernel(nu=2.5)
            else:
                raise ValueError(f"Unknown kernel: {kernel_type}")
            
            # Add noise kernel via likelihood, not explicitly in kernel
            self.covar_module = ScaleKernel(base_kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    # Initialize likelihood and model
    likelihood = GaussianLikelihood()
    model = ExactGPModel(X_train, y_train_norm, likelihood, cfg.kernel)

    # Set noise level
    with torch.no_grad():
        likelihood.noise = torch.tensor(cfg.white_noise_level, dtype=torch.float32)

    # Training mode
    model.train()
    likelihood.train()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Brief training loop (quick fit; not extensive optimization)
    n_epochs = max(10, min(10000, d * cfg.niter_optim_per_dim))
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train_norm)
        loss.backward()
        optimizer.step()

    # Set to eval mode
    model.eval()
    likelihood.eval()

    # Create prediction function
    def predict_mean(x):
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(1, -1)).float()
        with torch.no_grad():
            output = likelihood(model(x_tensor))
            pred_mean = output.mean
        # Denormalize
        pred_mean_denorm = float(pred_mean[0] * y_std + y_mean)
        return pred_mean_denorm

    # ========== Compute validation set errors ==========
    # Generate independent validation set
    n_val = max(20, int(cfg.n_train_per_dim * d / 2))  # Smaller than training set
    rng_val = np.random.default_rng(cfg.seed + 1)  # Different seed from training
    
    if cfg.design == "lhs":
        U_val = latin_hypercube(rng_val, n_val, d)
    else:
        U_val = rng_val.random((n_val, d))
    
    X_val = lo + (hi - lo) * U_val
    
    # Evaluate on validation set
    y_val_true = np.empty(n_val, dtype=float)
    y_val_disturbed = np.empty(n_val, dtype=float)
    y_val_pred = np.empty(n_val, dtype=float)
    
    for i in range(n_val):
        x = X_val[i]
        raw = problem.f(x)
        y_val_true[i] = raw
        y_val_disturbed[i] = disturbance.apply(x, raw, problem.fopt, problem.xopt) if disturbance is not None else raw
        y_val_pred[i] = predict_mean(x)
    
    # Compute RMSE metrics
    rmse_true = float(np.sqrt(np.mean((y_val_pred - y_val_true) ** 2)))
    rmse_disturbed = float(np.sqrt(np.mean((y_val_pred - y_val_disturbed) ** 2)))

    return {
        "model": model,
        "likelihood": likelihood,
        "predict_mean": predict_mean,
        "X_train": X,
        "y_train": y,
        "n_train": n_train,
        "y_mean": y_mean.item() if cfg.normalize_y else 0.0,
        "y_std": y_std.item() if cfg.normalize_y else 1.0,
        "rmse_true": rmse_true,
        "rmse_disturbed": rmse_disturbed,
    }




# ========== Amplitude estimation ==========

def estimate_amplitude_log10(
    problem,
    eps: float = 1.0,          # fraction of robust local/global scale
    n_samples: int = 2000,
    local: bool = True,
    local_radius_frac: float = 0.02,  # fraction of min box width
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    min_A: float = 1e-12,
):
    lo, hi = problem.bounds
    d = problem.dim
    rng = np.random.default_rng(0)

    if local:
        width = hi - lo
        r = local_radius_frac * float(np.min(width))

        # random directions on sphere
        U = rng.normal(size=(n_samples, d))
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-300
        radii = r * rng.random(n_samples) ** (1.0 / d)  # uniform in ball
        X = problem.xopt + U * radii[:, None]
        X = np.array([problem.clip(x) for x in X])
    else:
        X = lo + (hi - lo) * rng.random((n_samples, d))

    vals = np.array([problem.f(x) for x in X], dtype=float)
    deltas = vals - float(problem.fopt)

    # robust spread; add tiny constant to avoid 0 spread edge cases
    scale = float(np.quantile(deltas, q_hi) - np.quantile(deltas, q_lo)) + 1e-300

    A = max(min_A, eps * scale)

    # convert to logscale: 10**amplitude = A  => amplitude = log10(A)
    return float(np.log10(A))



# ========== Problem sources (COCO if available, else built-ins) ==========
def make_builtin_problems(dims: List[int]) -> List[Problem]:
    problems = []
    for d in dims:
        lo = -5*np.ones(d)
        hi = 5*np.ones(d)
        xopt = np.zeros(d)
        xopt_rosen = np.ones(d)
        problems += [
            Problem("Sphere", d, (lo, hi), sphere, xopt=xopt, fopt=0.0),
            Problem("Rastrigin", d, (lo, hi), rastrigin, xopt=xopt, fopt=0.0),
            Problem("Rosenbrock", d, (lo, hi), rosenbrock, xopt=xopt_rosen, fopt=0.0),
            Problem("Ackley", d, (lo, hi), ackley, xopt=xopt, fopt=0.0),
        ]
    return problems

def make_coco_bbob_problems(dims, functions=list(range(1,25)), instances=list(range(1,10))):
    problems = []
    for d in dims:
        for f_id in functions:
            for inst in instances:
                bp = cocoex.BareProblem("bbob", f_id, d, inst)
                x_opt = np.asarray(bp.best_parameter(), float)
                f_opt = float(bp.best_value())

                # bind bp at definition time
                def make_f_callable(bp):
                    def f_callable(x):
                        return float(bp([float(v) for v in x]))
                    return f_callable

                lo = -5.0 * np.ones(d)
                hi =  5.0 * np.ones(d)
                problems.append(
                    Problem(
                        name=f"COCO_bare_bbob_f{f_id}_i{inst}_d{d}",
                        dim=d,
                        bounds=(lo, hi),
                        f=make_f_callable(bp),   # <- now correctly bound
                        xopt=x_opt,
                        fopt=f_opt,
                        metadata={"xopt": x_opt}
                    )
                )
    return problems

# ========== Optimizers ==========
def de_optimize(problem: Problem, budget: int, seed: int, evaluator: Callable) -> Dict[str, Any]:
    """
    Differential Evolution optimizer using scipy.optimize.

    Parameters:
        problem (Problem): The optimization problem to solve.
        budget (int): Maximum number of function evaluations allowed.
        seed (int): Random seed for reproducibility.
        evaluator (Callable): Objective function wrapper for logging and disturbance.

    Returns:
        Dict[str, Any]: Dictionary containing the best solution 'x', its perturbed function value 'fun',
                        number of iterations 'nit', and number of function evaluations 'nfev'.
    """
    if scipy_opt is None:
        raise RuntimeError("scipy is not available for Differential Evolution.")
    rng = np.random.RandomState(seed)
    lo, hi = problem.bounds
    bounds = list(zip(lo, hi))
    # SciPy DE evaluations per iteration ≈ popsize * dim
    popsize = 15  # default; can be tuned
    per_iter = popsize * problem.dim
    maxiter = max(1, int(np.floor(budget / max(per_iter, 1))))
    res = scipy_opt.differential_evolution(
        evaluator,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=0.0,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=rng,
        polish=False,
        disp=False,
        updating="deferred",
        workers=1,
    )
    return {"x": res.x, "fun": res.fun, "nit": res.nit, "nfev": evaluator.n_evals}

def cma_optimize(problem: Problem, budget: int, seed: int, evaluator: Callable) -> Dict[str, Any]:
    if cma_pkg is None:
        raise RuntimeError("cma package is not available for CMA-ES.")
    lo, hi = problem.bounds
    # Start at center with sigma relative to range
    x0 = (lo + hi) / 2.0
    sigma0 = float(np.mean(hi - lo) / 6)  # heuristic
    opts = {
        "seed": seed,
        "bounds": [lo.tolist(), hi.tolist()],
        "verbose": -9,
        "maxfevals": int(budget),
        "tolfun": 0,
        "tolx": 0, #todo check stopping criteria
    }
    n_evals = 0
    best_x = x0.copy()
    best_f = float("inf")

    es = cma_pkg.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    es.optimize(evaluator)
    res = es.result  # (xbest, fbest, evals_best, evals, iterations, etc.)
    return {"x": np.array(res.xbest), "fun": float(res.fbest), "nit": int(res.iterations), "nfev": int(evaluator.n_evals)}

# Alternative optimizer: random search (baseline, should be unaffected by perturbation?)
def random_search(problem: Problem, budget: int, seed: int, evaluator: Callable) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    lo, hi = problem.bounds
    for _ in range(budget):
        x = lo + (hi - lo) * rng.rand(problem.dim)
        evaluator(x)
    return {"x": evaluator.best_x, "fun": evaluator.best_f, "nit": 0, "nfev": evaluator.n_evals}

# Registry
OPTIMIZERS: Dict[str, Callable[[Problem, int, int, Callable], Dict[str, Any]]] = {}
if scipy_opt is not None:
    OPTIMIZERS["DE"] = de_optimize
if cma_pkg is not None:
    OPTIMIZERS["CMA-ES"] = cma_optimize
# Always include random search as a fallback baseline
OPTIMIZERS["Random"] = random_search

# ========== Experiment runner ==========
@dataclass
class ExperimentConfig:
    dims: List[int] = field(default_factory=lambda: [2])
    budget_per_dim: int = 400  # total budget = budget_per_dim * dim
    seeds: List[int] = field(default_factory=lambda: [0])
    amplitude_list: List[float] = field(default_factory=lambda: [0.0])
    frequency_list: List[float] = field(default_factory=lambda: [1.0])
    use_coco: bool = True
    coco_functions: List[int] = field(default_factory=lambda: [1])
    coco_instances: List[int] = field(default_factory=lambda: [1])
    builtin_only: bool = False
    # --- Surrogate toggle ---
    fit_surrogate: bool = True
    surrogate_cfg: GPSurrogateConfig = field(default_factory=GPSurrogateConfig)

def build_problems(cfg: ExperimentConfig) -> List[Problem]:
    problems: List[Problem] = []
    if cfg.use_coco and cocoex is not None:
        problems += make_coco_bbob_problems(cfg.dims, cfg.coco_functions, cfg.coco_instances)
    ## Always include built-ins unless explicitly asked not to
    #if (not cfg.use_coco) or (not problems) or (not cfg.builtin_only):
    #    problems += make_builtin_problems(cfg.dims)
    return problems

def run_experiments(
    cfg,
    runs_path: Optional[str] = None,
    paths_path: Optional[str] = None,
    paths_gp_path: Optional[str] = None,
    flush_every: int = 1000,
    plot_during_experiments: bool = False,
    plot_output_dir: Optional[str] = None,
    plot_seed: int = 1,
):
    """
    Run all experiments from cfg and log:

      - runs_path: gzipped CSV with one row per run (df_results + run_id)
      - paths_path: gzipped CSV with one row per evaluation per run
                    (run_id, eval, raw_f, perturbed_f, x0..x{max_dim-1})

    If plot_during_experiments=True:
      - After each unique (problem, amplitude, frequency, algo) combo,
        generate comparison plots (true, disturbed, surrogate) for plot_seed
      - Saves to plot_output_dir

    Returns:
      df_results: pandas DataFrame (all runs, includes run_id)
      file_paths: dict with keys 'runs', 'paths'
    """

    # Import here to avoid circular import issues
    from surfplots import plot_true_disturbed_surrogate_comparison

    problems = build_problems(cfg)
    max_dim = max(p.dim for p in problems)


    # Default output filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]

    if runs_path is None:
        runs_path = f"runs_{timestamp}_{uid}.csv.gz"
    if paths_path is None:
        paths_path = f"paths_{timestamp}_{uid}.csv.gz"
    if paths_gp_path is None:
        paths_gp_path = f"paths_gp_{timestamp}_{uid}.csv.gz"

     # Paths writers (gzipped)
    paths_writer, gz_file = create_gzipped_paths_writer(paths_path, max_dim=max_dim)
    paths_gp_writer, gz_gp_file = create_gzipped_paths_writer(paths_gp_path, max_dim=max_dim)

    # In-memory buffers to track paths (for plotting during experiments)
    # Maps: run_id -> DataFrame with columns [run_id, eval, raw_f, perturbed_f, x0..x{max_dim-1}, branch]
    path_data_cache: Dict[int, pd.DataFrame] = {}

    results: List[Dict[str, Any]] = []
    run_id_true = 0
    run_id_gp = 0

    try:
        for prob in problems:
            print(f"\n=== Problem: {prob.name} (dim={prob.dim}) ===")
            total_budget = cfg.budget_per_dim * prob.dim

            for A in cfg.amplitude_list:
                A_log10 = estimate_amplitude_log10(prob, eps=A, local=True, local_radius_frac=0.05)

                for freq in cfg.frequency_list:

                    disturbance = SineDisturbance(
                        amplitude=A_log10,
                        frequency=freq,
                        mode="sin_x",
                    )

                    # Fit surrogate once per disturbance configuration (reuse across all algorithms)
                    gp_pack = None
                    if getattr(cfg, "fit_surrogate", False):
                        gp_pack = fit_gp_surrogate(prob, disturbance, cfg.surrogate_cfg)

                    for algo_name, algo_fn in OPTIMIZERS.items():
                        for seed in cfg.seeds:

                                                        # ===================== TRUE disturbed run =====================
                            this_run_id_true = run_id_true
                            run_id_true += 1

                            evaluator_true = CSVPathLogger(
                                problem=prob,
                                disturbance=disturbance,
                                budget=total_budget,
                                writer=paths_writer,
                                run_id=this_run_id_true,
                                max_dim=max_dim,
                                flush_every=flush_every,
                                log_only_improvements=True,
                            )

                            out_true = algo_fn(prob, total_budget, seed, evaluator_true)
                            evaluator_true.close()

                            best_x_true = out_true["x"]
                            best_fun_true = float(out_true["fun"]) if np.isfinite(out_true["fun"]) else np.nan
                            best_fun_undisturbed_true = float(prob.f(best_x_true))
                            best_gap_undisturbed_true = best_fun_undisturbed_true - float(prob.fopt)
                            best_xgap_true = float(np.linalg.norm(best_x_true - prob.xopt, ord=None))

                            # ===================== GP surrogate run (optional) =====================
                            this_run_id_gp = ""
                            out_gp = None
                            evaluator_gp = None  # Initialize before if block

                            best_fun_gp = ""
                            best_xgap_gp = ""
                            best_fun_true_at_gp = ""
                            best_fun_raw_at_gp = ""
                            nfev_gp = ""
                            nit_gp = ""
                            status_gp = ""
                            gp_n_train = ""
                            gp_kernel = ""
                            gp_rmse_true = ""
                            gp_rmse_disturbed = ""

                            if gp_pack is not None:
                                this_run_id_gp = run_id_gp
                                run_id_gp += 1

                                evaluator_gp = CSVSurrogatePathLogger(
                                    problem=prob,
                                    disturbance=disturbance,
                                    surrogate_predict=gp_pack["predict_mean"],
                                    budget=total_budget,
                                    writer=paths_gp_writer,
                                    run_id=this_run_id_gp,
                                    max_dim=max_dim,
                                    flush_every=flush_every,
                                    log_only_improvements=True,
                                )

                                out_gp = algo_fn(prob, total_budget, seed, evaluator_gp)
                                evaluator_gp.close()

                                best_x_gp = out_gp["x"]
                                best_fun_gp = float(out_gp["fun"]) if np.isfinite(out_gp["fun"]) else np.nan
                                best_xgap_gp = float(np.linalg.norm(best_x_gp - prob.xopt, ord=None))

                                # Evaluate GP optimum on true objectives (raw + disturbed)
                                best_fun_raw_at_gp = float(prob.f(best_x_gp))
                                best_fun_true_at_gp = best_fun_raw_at_gp
                                if disturbance is not None:
                                    best_fun_true_at_gp = float(
                                        disturbance.apply(best_x_gp, best_fun_raw_at_gp, prob.fopt, prob.xopt)
                                    )

                                nfev_gp = int(out_gp.get("nfev", evaluator_gp.n_evals))
                                nit_gp = int(out_gp.get("nit", -1))
                                status_gp = out_gp.get("status", "ok")

                                gp_n_train = int(gp_pack.get("n_train", ""))
                                gp_rmse_true = float(gp_pack.get("rmse_true", ""))
                                gp_rmse_disturbed = float(gp_pack.get("rmse_disturbed", ""))

                            # ===================== One combined runs row =====================
                            row = {
                                # run ids refer to DIFFERENT path files
                                "run_id_true": int(this_run_id_true),
                                "run_id_gp": this_run_id_gp,

                                "problem": prob.name,
                                "dim": prob.dim,
                                "algo": algo_name,
                                "amplitude": float(A),
                                "amplitude_actual": float(A_log10),
                                "frequency": float(freq),
                                "dist_mode": disturbance.mode,
                                "seed": int(seed),
                                "budget": int(total_budget),

                                # TRUE run metrics 
                                "best_fun_true": best_fun_true,
                                "best_gap_true": (best_fun_true - float(prob.fopt)) if np.isfinite(best_fun_true) else np.nan,
                                "best_fun_undisturbed_true": best_fun_undisturbed_true,
                                "best_gap_undisturbed_true": best_gap_undisturbed_true,
                                "best_xgap_true": best_xgap_true,
                                "nfev_true": int(out_true.get("nfev", evaluator_true.n_evals)),
                                "nit_true": int(out_true.get("nit", -1)),
                                "status_true": out_true.get("status", "ok"),

                                # GP run metrics (blank if surrogate disabled)
                                "gp_n_train": gp_n_train,
                                "gp_rmse_true": gp_rmse_true,
                                "gp_rmse_disturbed": gp_rmse_disturbed,
                                "best_fun_gp": best_fun_gp,
                                "best_xgap_gp": best_xgap_gp,
                                "best_fun_true_at_gp": best_fun_true_at_gp,
                                "best_fun_raw_at_gp": best_fun_raw_at_gp,
                                "best_gap_true_at_gp": (
                                    float(best_fun_true_at_gp) - float(prob.fopt)
                                ) if best_fun_true_at_gp != "" else "",
                                "nfev_gp": nfev_gp,
                                "nit_gp": nit_gp,
                                "status_gp": status_gp,

                                # optimum info
                                "fopt": float(prob.fopt),
                            }

                            results.append(row)

                        # ===================== Generate plots after all seeds for this (prob, amp, freq, algo) =====================
                        # Now supports both 2D and high-D cases (slices for dim > 2)
                        if plot_during_experiments and plot_output_dir:
                            # Find the run with matching seed
                            plot_seed_data = None
                            for row in results[-len(cfg.seeds):]:  # Last N results are this algo's runs
                                if row["seed"] == plot_seed:
                                    plot_seed_data = row
                                    break
                            
                            if plot_seed_data is None and len(cfg.seeds) > 0:
                                # If exact seed not found, use first seed
                                plot_seed_data = results[-len(cfg.seeds)]
                            
                            if plot_seed_data is not None and gp_pack is not None:
                                fig = None
                                try:
                                    # Get path data from in-memory buffers (no need to read from gzipped files)
                                    path_df = None
                                    frames = []
                                    
                                    # Get true run path data (if evaluator_true exists in scope)
                                    if evaluator_true is not None:
                                        df_true = evaluator_true.get_path_dataframe()
                                        if not df_true.empty:
                                            df_true["branch"] = "true"
                                            frames.append(df_true)
                                    
                                    # Get GP run path data (if evaluator_gp exists in scope)
                                    if evaluator_gp is not None:
                                        df_gp = evaluator_gp.get_path_dataframe()
                                        if not df_gp.empty:
                                            df_gp["branch"] = "gp"
                                            frames.append(df_gp)
                                    
                                    if frames:
                                        path_df = pd.concat(frames, ignore_index=True)
                                    
                                    # Create output filename
                                    safe_prob = prob.name.replace("/", "_")
                                    fname = f"{safe_prob}_amp{float(A):.2f}_freq{float(freq):.2f}_algo{algo_name}.png"
                                    fpath = str(Path(plot_output_dir) / fname)
                                    
                                    # Generate reproducible slice seed from problem/amplitude/frequency
                                    # (for dim > 2, ensures same slice visualization across runs)
                                    slice_seed = hash((prob.name, float(A), float(freq))) % (2**31)
                                    
                                             # Generate comparison plot (handles 2D directly, uses slices for dim > 2)
                                    fig, axes = plot_true_disturbed_surrogate_comparison(
                                        problem=prob,
                                        disturbance=disturbance,
                                        gp_dict=gp_pack,
                                        path_df=path_df,
                                        n_grid=150,
                                        output_path=fpath,
                                        dpi=100,
                                        slice_radius=0.5, #this radius is given ass a fraction of the range, meaning 0.5 covers roughly 100%
                                        slice_seed=slice_seed,
                                        amp_multiplier=float(A),
                                        gp_rmse_true=gp_rmse_true,
                                        gp_rmse_disturbed=gp_rmse_disturbed,
                                    )
                                    
                                except Exception as e:
                                    print(f"Warning: Could not plot {prob.name} | {algo_name}: {e}")
                                
                                finally:
                                    # Always close figure, even if exception occurred
                                    if fig is not None:
                                        plt.close(fig)


        # Build runs DataFrame
        df_results = pd.DataFrame(results)

        # Save runs as gzipped CSV
        df_results.to_csv(runs_path, index=False, compression="gzip")
        print(f"\nSaved runs summary to: {runs_path}")
        print(f"Saved paths TRUE (per-eval) to: {paths_path}")
        print(f"Saved paths GP   (per-eval) to: {paths_gp_path}")

        return df_results, {"runs": runs_path, "paths_true": paths_path, "paths_gp": paths_gp_path}


    finally:
        # Ensure gz files are closed
        gz_file.close()
        gz_gp_file.close()

