from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Loading & preparation
# -------------------------------

POSSIBLE_GAP_COLS = [
    "best_gap_undisturbed_true",
    "best_gap_undisturbed",
    "best_gap_true",
    "best_gap",
]
POSSIBLE_XGAP_COLS = [
    "best_xgap_true",
    "best_xgap",
]
POSSIBLE_BESTF_COLS = [
    "best_fun_true",
    "best_fun",
    "best_f",
    "best_value",
]
POSSIBLE_FOPT_COLS = ["fopt","f_opt","opt_value"]

REQUIRED_BASE_COLS = [
    "problem",
    "dim",
    "algo",
    "amplitude",
    "frequency",
    "best_gap_undisturbed",
    "best_xgap",
]

def prepare_results(data: "pd.DataFrame | str") -> pd.DataFrame:
    """
    Load and minimally clean runs data for xgap-based analysis.

    Expected columns in runs file (Option A):
      - best_xgap_true
      - best_xgap_gp (may be blank if surrogate disabled)
      - problem, dim, algo, amplitude, frequency, seed, budget
    """
    if isinstance(data, str):
        # allow both .csv and .csv.gz
        df = pd.read_csv(data, compression="infer")
    else:
        df = data.copy()

    # Drop duplicate columns if any (not strictly necessary for xgap)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Coerce core numeric inputs
    for c in ["dim", "seed", "budget"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in ["amplitude", "frequency", "best_xgap_true", "best_xgap_gp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived xgap metric: how much worse/better GP is in x-distance
    if "best_xgap_true" in df.columns and "best_xgap_gp" in df.columns:
        df["xgap_delta"] = df["best_xgap_gp"] - df["best_xgap_true"]

    # Keep only rows that can support "true" xgap analysis
    required = ["problem", "dim", "algo", "amplitude", "frequency", "best_xgap_true"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for xgap analysis: {missing}")

    df = df.dropna(subset=required)

    return df

def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def select_xgap_metric(df: pd.DataFrame, metric: str = "true") -> pd.DataFrame:
    """
    Compatibility helper used by notebooks: pick an xgap-style metric and
    return a DataFrame containing an `xgap_metric` column suitable for
    heatmap/plotting functions.

    metric: one of 'true', 'gp', 'delta'
    - 'true' -> uses `best_xgap_true` or falls back to `best_xgap`
    - 'gp'   -> uses `best_xgap_gp` or falls back to `best_xgap`
    - 'delta'-> uses `xgap_delta` if present, otherwise computes
                `best_xgap_gp - best_xgap_true` when possible
    """
    if metric not in ("true", "gp", "delta"):
        raise ValueError("metric must be one of 'true','gp','delta'")

    out = df.copy()

    if metric == "true":
        col = _first_present(out, ["best_xgap_true", "best_xgap"])
        if col is None:
            raise ValueError("No column found for 'true' xgap (expected best_xgap_true or best_xgap)")
        out["xgap_metric"] = pd.to_numeric(out[col], errors="coerce")

    elif metric == "gp":
        col = _first_present(out, ["best_xgap_gp", "best_xgap"])
        if col is None:
            raise ValueError("No column found for 'gp' xgap (expected best_xgap_gp or best_xgap)")
        out["xgap_metric"] = pd.to_numeric(out[col], errors="coerce")

    else:  # delta
        if "xgap_delta" in out.columns:
            out["xgap_metric"] = pd.to_numeric(out["xgap_delta"], errors="coerce")
        elif ("best_xgap_gp" in out.columns) and ("best_xgap_true" in out.columns):
            out["xgap_metric"] = pd.to_numeric(out["best_xgap_gp"], errors="coerce") - pd.to_numeric(out["best_xgap_true"], errors="coerce")
        else:
            raise ValueError("Cannot compute 'delta' xgap: need 'xgap_delta' or both 'best_xgap_gp' and 'best_xgap_true'")

    # drop rows without a numeric metric
    out = out.dropna(subset=["xgap_metric"]).copy()
    return out

# -------------------------------
# Slices & matrices (for heatmaps)
# -------------------------------

def make_matrix_xgap(df: pd.DataFrame, problem: str, dim: int, algo: str,
                     value_col: str = "xgap_metric", stat: str = "median"):
    """
    Build (A, F, Z) for heatmaps where Z[f_idx, a_idx] is the aggregated xgap metric.
    A and F are log10 values (consistent with your existing approach).
    """
    d = df[(df["problem"] == problem) & (df["dim"] == dim) & (df["algo"] == algo)].copy()
    if d.empty:
        raise ValueError("No data after filtering")

    A = np.sort(d["amplitude"].unique())
    F = np.sort(d["frequency"].unique())

    Z = np.full((len(F), len(A)), np.nan, dtype=float)

    for i, f in enumerate(F):
        for j, a in enumerate(A):
            cell = d[(d["frequency"] == f) & (d["amplitude"] == a)][value_col].dropna()
            if cell.empty:
                continue
            if stat == "median":
                Z[i, j] = float(np.median(cell))
            elif stat == "mean":
                Z[i, j] = float(np.mean(cell))
            else:
                raise ValueError("stat must be 'median' or 'mean'")

    return A, F, Z

# -------------------------------
# Plotting (matplotlib only, one chart per call)
# -------------------------------

def _edges_from_centers(centers: np.ndarray):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        # single bin: make a tiny symmetric pad
        pad = 0.5 if centers[0] == 0 else abs(centers[0]) * 0.05
        return np.array([centers[0] - pad, centers[0] + pad])
    diffs = np.diff(centers)
    interior = centers[:-1] + diffs / 2
    first = centers[0] - diffs[0] / 2
    last  = centers[-1] + diffs[-1] / 2
    return np.concatenate(([first], interior, [last]))

def plot_heatmap(
    df: pd.DataFrame,
    problem: str,
    dim: int,
    algo: str,
    value: str = "median",
    log10_axes: bool = True,
    swap_axes: bool = True,   # <--- NEW
    title: Optional[str] = None,
):
    """
    Heatmap of statistic across amplitude/frequency grid.

    make_matrix_xgap returns:
      A: amplitudes, F: frequencies, Z shape (len(F), len(A)) with Z[f_idx, a_idx]

    If swap_axes=True:
      x-axis becomes frequency, y-axis becomes amplitude, and we plot Z.T
    """
    A, F, Z = make_matrix_xgap(df, problem, dim, algo, value_col="xgap_metric", stat=value)

    # centers in plotting space
    amp_centers = 10**A if log10_axes else A
    freq_centers = 10**F if log10_axes else F

    if not swap_axes:
        # x=amplitude, y=frequency
        x_centers = np.asarray(amp_centers)
        y_centers = np.asarray(freq_centers)
        C = np.asarray(Z)              # shape (nF, nA)
        xlab = "amplitude"
        ylab = "frequency"
    else:
        # x=frequency, y=amplitude  ==> transpose Z
        x_centers = np.asarray(freq_centers)
        y_centers = np.asarray(amp_centers)
        C = np.asarray(Z).T            # shape (nA, nF)
        xlab = "frequency"
        ylab = "amplitude"

    x_edges = _edges_from_centers(x_centers)
    y_edges = _edges_from_centers(y_centers)

    # sanity check: C must be (len(y_centers), len(x_centers))
    if C.shape != (len(y_centers), len(x_centers)):
        raise ValueError(
            f"Heatmap shape mismatch: C{C.shape} vs "
            f"(len(y)={len(y_centers)}, len(x)={len(x_centers)}). "
            f"swap_axes={swap_axes}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(x_edges, y_edges, C, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"best_xgap ({value})")

    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)

    ax.set_xlabel(xlab + (" (linearized)" if log10_axes else ""))
    ax.set_ylabel(ylab + (" (linearized)" if log10_axes else ""))

    # optional minor grid on edges
    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", linewidth=0.5, alpha=0.3)

    if title is None:
        title = f"{problem} | dim={dim} | alg={algo}" + (" | swapped" if swap_axes else "")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


