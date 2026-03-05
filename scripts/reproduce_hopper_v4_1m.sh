#!/usr/bin/env bash
# reproduce_hopper_v4_1m.sh
#
# Full 1M-step benchmark: PPO state vs. pixels vs. pixels+stack=4
# on Hopper-v4 for 1 000 000 environment steps, 3 seeds, 5 eval episodes.
#
# Outputs (separate from 20k run, no overwrites):
#   runs/compare/1m_eval10/{state,pixels,pixels_fs4}/seed_{0,1,2}/
#     config.json      — full hyperparameter record
#     metrics.csv      — per-update + per-eval rows
#     checkpoints/     — .pt files at each eval boundary
#   reports/figures/1m/compare_eval_return.png
#   reports/figures/1m/compare_train_return.png
#   reports/results_hopper_v4_1m.md
#
# Runtime estimates (CPU, single machine):
#   State mode           : ~4–8 h per seed
#   Pixel mode           : ~12–24 h per seed (CNN forward + backward per batch)
#   Pixel + stack=4 mode : ~12–24 h per seed (same CNN, 4× input channels)
#   Total (all 9 runs)   : 1–3 days on CPU; 4–8 h with a GPU
#
# Usage:
#   bash scripts/reproduce_hopper_v4_1m.sh
#   DEVICE=cuda bash scripts/reproduce_hopper_v4_1m.sh
#   OUT_DIR=runs/compare/my_1m_run bash scripts/reproduce_hopper_v4_1m.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via env vars or --flags)
# ---------------------------------------------------------------------------
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-runs/compare/1m_eval10}"
ENV_ID="${ENV_ID:-Hopper-v4}"
TOTAL_TIMESTEPS=1000000
EVAL_EVERY=10000
N_STEPS=2048
EPOCHS=10
BATCH_SIZE=64
IMG_SIZE=64
N_EVAL_EPISODES=5
SEEDS="0,1,2"

# Allow --device / --out_dir / --seeds overrides from CLI
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2";         shift 2 ;;
        --out_dir)  OUT_DIR="$2";        shift 2 ;;
        --seeds)    SEEDS="$2";          shift 2 ;;
        *) echo "[warn] Unknown argument: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "[error] .venv not found at ${REPO_ROOT}/.venv"
    echo "        Run: python -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# Figures output directory (separate from 20k figures to avoid overwriting)
FIGURES_DIR="${REPO_ROOT}/reports/figures/1m"
mkdir -p "${FIGURES_DIR}"

echo "============================================================"
echo "  visual-rl-locomotion — Hopper-v4 1M Reproduction Run"
echo "  Conditions: state | pixels | pixels_fs4"
echo "============================================================"
echo "  repo      : ${REPO_ROOT}"
echo "  python    : ${PYTHON}"
echo "  device    : ${DEVICE}"
echo "  out_dir   : ${OUT_DIR}"
echo "  seeds     : ${SEEDS}"
echo "  steps     : ${TOTAL_TIMESTEPS}"
echo "  eval@     : every ${EVAL_EVERY} steps × ${N_EVAL_EPISODES} episodes"
echo "  figures   : ${FIGURES_DIR}"
echo "============================================================"
echo ""
echo "  NOTE: This is a long-running job."
echo "  Estimated wall time on CPU: 1–3 days for all 9 runs."
echo "  Use DEVICE=cuda to significantly reduce runtime."
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1a — state + pixels (no frame stacking)
# ---------------------------------------------------------------------------
echo ""
echo "[1a/4] Launching state + pixels (no stack) via run_compare.py ..."
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/run_compare.py" \
    --env_id          "${ENV_ID}"            \
    --total_timesteps "${TOTAL_TIMESTEPS}"   \
    --eval_every      "${EVAL_EVERY}"        \
    --n_steps         "${N_STEPS}"           \
    --epochs          "${EPOCHS}"            \
    --batch_size      "${BATCH_SIZE}"        \
    --device          "${DEVICE}"            \
    --img_size        "${IMG_SIZE}"          \
    --n_eval_episodes "${N_EVAL_EPISODES}"   \
    --seeds           "${SEEDS}"             \
    --out_dir         "${REPO_ROOT}/${OUT_DIR}"

# ---------------------------------------------------------------------------
# Step 1b — pixels with frame_stack=4
# ---------------------------------------------------------------------------
echo ""
echo "[1b/4] Launching pixels + frame_stack=4 via run_compare.py ..."
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/run_compare.py" \
    --env_id          "${ENV_ID}"            \
    --total_timesteps "${TOTAL_TIMESTEPS}"   \
    --eval_every      "${EVAL_EVERY}"        \
    --n_steps         "${N_STEPS}"           \
    --epochs          "${EPOCHS}"            \
    --batch_size      "${BATCH_SIZE}"        \
    --device          "${DEVICE}"            \
    --img_size        "${IMG_SIZE}"          \
    --n_eval_episodes "${N_EVAL_EPISODES}"   \
    --seeds           "${SEEDS}"             \
    --frame_stack     4                      \
    --out_dir         "${REPO_ROOT}/${OUT_DIR}"

# ---------------------------------------------------------------------------
# Step 2 — Learning-curve plots
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Generating learning-curve plots → ${FIGURES_DIR} ..."
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/plot_results.py" \
    --runs_dir "${REPO_ROOT}/${OUT_DIR}"  \
    --out_dir  "${FIGURES_DIR}"

# ---------------------------------------------------------------------------
# Step 3 — Aggregate + write results markdown
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Aggregating results into reports/results_hopper_v4_1m.md ..."
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/aggregate_results.py" \
    --runs_dir "${REPO_ROOT}/${OUT_DIR}"               \
    --out_dir  "${FIGURES_DIR}"                        \
    --save_md  "${REPO_ROOT}/reports/results_hopper_v4_1m.md"

echo ""
echo "============================================================"
echo "  Reproduction complete.  3 conditions × 3 seeds = 9 runs."
echo ""
echo "  Results    : ${OUT_DIR}/{state,pixels,pixels_fs4}/"
echo "  Plots      : reports/figures/1m/compare_eval_return.png"
echo "             : reports/figures/1m/compare_train_return.png"
echo "  Summary    : reports/results_hopper_v4_1m.md"
echo "============================================================"
