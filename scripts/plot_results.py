"""
Learning-curve plotter for visual-rl-locomotion comparison runs.

Reads metrics.csv files produced by run_compare.py, aligns eval rows
across seeds by global_step, and plots mean ± std shaded curves for
state-based and pixel-based PPO.

Output:
    <out_dir>/compare_eval_return.png   — eval_return_mean ± std
    <out_dir>/compare_train_return.png  — episode_return ± std  (if available)

Usage:
    python scripts/plot_results.py \\
        --runs_dir runs/compare/demo \\
        --out_dir  reports/figures

Dependencies: numpy, pandas, matplotlib (all in requirements.txt).
No seaborn.
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean ± std learning curves from a compare run."
    )
    parser.add_argument(
        "--runs_dir", type=str, required=True,
        help="Root directory produced by run_compare.py, e.g. runs/compare/demo."
    )
    parser.add_argument(
        "--metric", type=str, default="eval_return_mean",
        help="CSV column to plot as the primary Y axis."
    )
    parser.add_argument(
        "--out_dir", type=str, default="reports/figures",
        help="Directory where PNG files are saved."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_eval_rows(seed_dir: str, metric: str) -> Optional[pd.DataFrame]:
    """
    Load metrics.csv for one seed and return only the eval rows.

    An 'eval row' is one where the requested metric column is non-empty
    and numeric.  These are the dedicated rows logged after each evaluation
    call in train_ppo.py.

    Returns None if the file is missing or has no valid eval rows.
    """
    csv_path = os.path.join(seed_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        print(f"  [warn] Missing: {csv_path}")
        return None

    df = pd.read_csv(csv_path, dtype=str)   # read everything as string first

    if metric not in df.columns or "global_step" not in df.columns:
        print(f"  [warn] Column '{metric}' not found in {csv_path}")
        return None

    # Keep only rows where metric is a non-empty, parseable float.
    mask = df[metric].notna() & (df[metric].str.strip() != "")
    df = df[mask].copy()

    if df.empty:
        print(f"  [warn] No eval rows for metric '{metric}' in {csv_path}")
        return None

    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    df[metric]        = pd.to_numeric(df[metric],        errors="coerce")
    df = df.dropna(subset=["global_step", metric])

    return df[["global_step", metric]].reset_index(drop=True)


def collect_mode_curves(
    runs_dir: str,
    mode: str,
    metric: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Aggregate per-seed eval curves for one obs_mode into mean ± std arrays.

    Seeds are discovered automatically by listing subdirectories of
    <runs_dir>/<mode>/ whose names start with "seed_".

    Returns:
        steps : (T,) sorted unique global_step values present in ≥ 1 seed.
        mean  : (T,) nan-mean across seeds at each step.
        std   : (T,) nan-std  across seeds at each step.

    Returns (None, None, None) if no data is found.
    """
    mode_dir = os.path.join(runs_dir, mode)
    if not os.path.isdir(mode_dir):
        print(f"  [warn] Mode directory not found: {mode_dir}")
        return None, None, None

    seed_dirs = sorted(
        d for d in os.listdir(mode_dir)
        if os.path.isdir(os.path.join(mode_dir, d)) and d.startswith("seed_")
    )
    if not seed_dirs:
        print(f"  [warn] No seed_* directories under {mode_dir}")
        return None, None, None

    seed_frames = []
    for sd in seed_dirs:
        df = _load_eval_rows(os.path.join(mode_dir, sd), metric)
        if df is not None:
            seed_frames.append(df.rename(columns={metric: sd}))

    if not seed_frames:
        return None, None, None

    # Outer-join all seeds on global_step so no step is dropped if one
    # seed has a missing entry.
    merged = seed_frames[0]
    for sf in seed_frames[1:]:
        merged = merged.merge(sf, on="global_step", how="outer")

    merged = merged.sort_values("global_step").reset_index(drop=True)

    steps  = merged["global_step"].to_numpy(dtype=float)
    vals   = merged.drop(columns="global_step").to_numpy(dtype=float)

    mean = np.nanmean(vals, axis=1)
    std  = np.nanstd(vals,  axis=1)

    n_seeds = vals.shape[1]
    print(f"  [{mode}] {n_seeds} seed(s), {len(steps)} eval step(s).")

    return steps, mean, std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour palette: accessible, distinct.
_COLOURS = {
    "state":       "#1976D2",   # blue
    "pixels":      "#E64A19",   # orange-red
    "pixels_fs4":  "#2E7D32",   # green
}

# Human-readable legend labels.
_LABELS = {
    "state":       "state",
    "pixels":      "pixels (no stack)",
    "pixels_fs4":  "pixels (stack=4)",
}


def _plot_curves(
    runs_dir: str,
    metric: str,
    out_dir: str,
    out_filename: str,
    title: str,
    ylabel: str,
) -> bool:
    """
    Build one comparison figure (two curves, shaded std bands) and save it.

    Returns True if at least one mode was successfully plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for mode in ["state", "pixels", "pixels_fs4"]:
        steps, mean, std = collect_mode_curves(runs_dir, mode, metric)
        if steps is None:
            continue

        colour = _COLOURS.get(mode, "grey")
        label  = _LABELS.get(mode, mode)
        ax.plot(steps, mean, label=label, color=colour, linewidth=2.0)
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            alpha=0.20,
            color=colour,
            linewidth=0,
        )
        any_data = True

    if not any_data:
        plt.close(fig)
        print(f"  [warn] Nothing to plot for metric '{metric}'.")
        return False

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.30, linestyle="--")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\nPlotting from: {args.runs_dir}")
    print(f"Output dir  : {args.out_dir}\n")

    # --- Primary plot: eval return ---
    _plot_curves(
        runs_dir     = args.runs_dir,
        metric       = args.metric,
        out_dir      = args.out_dir,
        out_filename = "compare_eval_return.png",
        title        = "Hopper-v4: Deterministic Eval Return — State vs. Pixels vs. Pixels+Stack\n(mean ± std across seeds)",
        ylabel       = "Eval Return (mean)",
    )

    # --- Optional secondary plot: episodic training return ---
    # Only generated if episode_return rows exist for at least one mode.
    print()
    _plot_curves(
        runs_dir     = args.runs_dir,
        metric       = "episode_return",
        out_dir      = args.out_dir,
        out_filename = "compare_train_return.png",
        title        = "Hopper-v4: Training Episode Return — State vs. Pixels vs. Pixels+Stack\n(mean ± std across seeds)",
        ylabel       = "Episode Return",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
