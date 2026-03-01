"""
Multi-seed results aggregator for visual-rl-locomotion.

Reads metrics.csv from a structured compare-run directory, aligns eval rows
across seeds by global_step, computes mean ± std, and outputs:
  1. A markdown summary table (printed to stdout, optionally written to file).
  2. An overlay plot (eval_return_mean ± std) saved to --out_dir.

Expected directory layout (produced by run_compare.py):
    <runs_dir>/
        state/
            seed_0/metrics.csv
            seed_1/metrics.csv
            seed_2/metrics.csv
        pixels/
            seed_0/metrics.csv
            ...

Usage:
    # Print table to stdout and generate overlay plot
    python scripts/aggregate_results.py \\
        --runs_dir runs/compare/20k_eval5 \\
        --out_dir  reports/figures \\
        --save_md  reports/results_hopper_v4.md

    # Just print, no files written
    python scripts/aggregate_results.py \\
        --runs_dir runs/compare/demo \\
        --no_plot
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed PPO results and produce a summary table."
    )
    p.add_argument("--runs_dir", type=str, required=True,
                   help="Root compare-run directory (e.g. runs/compare/20k_eval5).")
    p.add_argument("--metric",   type=str, default="eval_return_mean",
                   help="Primary metric column to aggregate.")
    p.add_argument("--out_dir",  type=str, default="reports/figures",
                   help="Directory for plot output.")
    p.add_argument("--save_md",  type=str, default=None,
                   help="If set, write the markdown summary to this file path.")
    p.add_argument("--no_plot",  action="store_true",
                   help="Skip generating plots.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_eval_rows(seed_dir: str, metric: str) -> Optional[pd.DataFrame]:
    """Return eval-only rows from one seed's metrics.csv, or None."""
    csv_path = os.path.join(seed_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path, dtype=str)
    if metric not in df.columns or "global_step" not in df.columns:
        return None
    mask = df[metric].notna() & (df[metric].str.strip() != "")
    df = df[mask].copy()
    if df.empty:
        return None
    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    df[metric]        = pd.to_numeric(df[metric],        errors="coerce")
    return df[["global_step", metric]].dropna().reset_index(drop=True)


def load_mode(
    runs_dir: str,
    mode: str,
    metric: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load per-seed eval DataFrames for one mode.

    Returns a dict {seed_name: DataFrame(global_step, metric)}.
    """
    mode_dir = os.path.join(runs_dir, mode)
    if not os.path.isdir(mode_dir):
        return {}

    frames: Dict[str, pd.DataFrame] = {}
    for sd in sorted(os.listdir(mode_dir)):
        full = os.path.join(mode_dir, sd)
        if not os.path.isdir(full):
            continue
        df = _load_eval_rows(full, metric)
        if df is not None:
            frames[sd] = df
    return frames


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    frames: Dict[str, pd.DataFrame],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Outer-join all seeds on global_step, compute nanmean / nanstd.

    Returns (steps, mean, std, n_seeds).
    """
    if not frames:
        return np.array([]), np.array([]), np.array([]), 0

    merged = None
    for name, df in frames.items():
        renamed = df.rename(columns={metric: name})
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on="global_step", how="outer")

    merged = merged.sort_values("global_step").reset_index(drop=True)
    steps  = merged["global_step"].to_numpy(dtype=float)
    vals   = merged.drop(columns="global_step").to_numpy(dtype=float)

    mean = np.nanmean(vals, axis=1)
    std  = np.nanstd(vals,  axis=1)
    return steps, mean, std, vals.shape[1]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary_table(
    runs_dir: str,
    metric: str,
) -> str:
    """
    Build a markdown table showing final-step mean ± std for each mode.

    'Final step' = the highest eval global_step present across all seeds.
    """
    rows: List[str] = []

    for mode in ["state", "pixels"]:
        frames = load_mode(runs_dir, mode, metric)
        steps, mean, std, n_seeds = aggregate(frames, metric)

        if len(steps) == 0:
            rows.append(f"| {mode:<12} | N/A — no data found | — | — |")
            continue

        # Use the last eval step as the summary point.
        final_mean = mean[-1]
        final_std  = std[-1]
        final_step = int(steps[-1])

        rows.append(
            f"| {mode:<12} | {final_mean:>8.1f} ± {final_std:<6.1f} "
            f"| {n_seeds} | {final_step:,} |"
        )

    runs_name = os.path.basename(os.path.normpath(runs_dir))

    lines = [
        f"## Aggregated Results — `{runs_name}`",
        "",
        f"Metric: `{metric}`  |  Source: `{runs_dir}`",
        "",
        "| Mode         | Final Eval Return (mean ± std) | Seeds | Final Step |",
        "|:-------------|:-------------------------------|------:|----------:|",
    ] + rows + [
        "",
        "*Mean and std computed across seeds at the final logged eval step.*",
        "*Std = 0 when only one seed is present.*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLOURS = {"state": "#1976D2", "pixels": "#E64A19"}


def plot_overlay(
    runs_dir: str,
    metric: str,
    out_dir: str,
) -> Optional[str]:
    """
    Generate mean ± std overlay plot for all modes and save as PNG.

    Returns the saved file path, or None if no data was found.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for mode in ["state", "pixels"]:
        frames = load_mode(runs_dir, mode, metric)
        steps, mean, std, n_seeds = aggregate(frames, metric)

        if len(steps) == 0:
            print(f"  [aggregate] No {mode} data — skipping curve.")
            continue

        colour = _COLOURS.get(mode, "grey")
        label  = f"{mode} (n={n_seeds})"
        ax.plot(steps, mean, label=label, color=colour, linewidth=2.0)
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.20, color=colour, linewidth=0)
        any_data = True

    if not any_data:
        plt.close(fig)
        return None

    runs_name = os.path.basename(os.path.normpath(runs_dir))
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(
        f"Hopper-v4 · {metric}\nState vs. Pixels — {runs_name} (mean ± std)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.30, linestyle="--")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "compare_eval_return.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [aggregate] Plot saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\n[aggregate] runs_dir : {args.runs_dir}")
    print(f"[aggregate] metric   : {args.metric}\n")

    # --- Build and print summary table ---
    table_md = build_summary_table(args.runs_dir, args.metric)
    print(table_md)

    # --- Optionally write markdown to file ---
    if args.save_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_md)), exist_ok=True)
        with open(args.save_md, "w") as f:
            f.write(f"# Results — Hopper-v4\n\n")
            f.write(f"*Generated by `scripts/aggregate_results.py` from `{args.runs_dir}`.*\n\n")
            f.write(table_md + "\n")
        print(f"\n  [aggregate] Markdown → {args.save_md}")

    # --- Generate plot ---
    if not args.no_plot:
        print()
        plot_overlay(args.runs_dir, args.metric, args.out_dir)

    print("\n[aggregate] Done.\n")


if __name__ == "__main__":
    main()
