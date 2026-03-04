#!/usr/bin/env bash
# reproduce_hopper_v4_20k.sh
#
# Fully reproducible benchmark: PPO state vs. pixels vs. pixels+stack=4
# on Hopper-v4 for 20 000 environment steps, 3 seeds, 5 eval episodes.
#
# Outputs:
#   runs/compare/20k_eval5/{state,pixels,pixels_fs4}/seed_{0,1,2}/
#     config.json      — full hyperparameter record
#     metrics.csv      — per-update + per-eval rows
#     checkpoints/     — .pt files at each eval boundary
#   reports/figures/compare_eval_return.png
#   reports/results_hopper_v4.md
#
# Runtime (CPU, single machine):
#   State mode           : ~5–15 min per seed
#   Pixel mode           : ~15–40 min per seed (CNN forward + backward per batch)
#   Pixel + stack=4 mode : ~15–40 min per seed (same CNN, 4× input channels)
#
# Usage:
#   bash scripts/reproduce_hopper_v4_20k.sh
#   bash scripts/reproduce_hopper_v4_20k.sh --device cuda   # override device

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via env vars or positional --flags below)
# ---------------------------------------------------------------------------
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-runs/compare/20k_eval5}"
ENV_ID="${ENV_ID:-Hopper-v4}"
TOTAL_TIMESTEPS=20000
EVAL_EVERY=2000
N_STEPS=2048
EPOCHS=10
BATCH_SIZE=64
IMG_SIZE=64
N_EVAL_EPISODES=5
SEEDS="0,1,2"

# Allow --device override from CLI without breaking set -u
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2";         shift 2 ;;
        --out_dir)  OUT_DIR="$2";        shift 2 ;;
        --seeds)    SEEDS="$2";          shift 2 ;;
        *) echo "[warn] Unknown argument: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root (works from any working directory)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "[error] .venv not found at ${REPO_ROOT}/.venv"
    echo "        Run: python -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

echo "============================================================"
echo "  visual-rl-locomotion — Hopper-v4 20k Reproduction Run"
echo "  Conditions: state | pixels | pixels_fs4"
echo "============================================================"
echo "  repo    : ${REPO_ROOT}"
echo "  python  : ${PYTHON}"
echo "  device  : ${DEVICE}"
echo "  out_dir : ${OUT_DIR}"
echo "  seeds   : ${SEEDS}"
echo "  steps   : ${TOTAL_TIMESTEPS}"
echo "  eval@   : every ${EVAL_EVERY} steps × ${N_EVAL_EPISODES} episodes"
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

echo ""
echo "[2/4] Generating learning-curve plots ..."
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/plot_results.py" \
    --runs_dir "${REPO_ROOT}/${OUT_DIR}"             \
    --out_dir  "${REPO_ROOT}/reports/figures"

# ---------------------------------------------------------------------------
# Step 3 — Aggregate + write results markdown
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Aggregating results into reports/results_hopper_v4.md ..."
echo ""

"$PYTHON" "${REPO_ROOT}/scripts/aggregate_results.py" \
    --runs_dir "${REPO_ROOT}/${OUT_DIR}"               \
    --out_dir  "${REPO_ROOT}/reports/figures"          \
    --save_md  "${REPO_ROOT}/reports/results_hopper_v4.md"

echo ""
echo "============================================================"
echo "  Reproduction complete.  3 conditions × 3 seeds = 9 runs."
echo ""
echo "  Results    : ${OUT_DIR}/{state,pixels,pixels_fs4}/"
echo "  Plot       : reports/figures/compare_eval_return.png"
echo "  Summary    : reports/results_hopper_v4.md"
echo "============================================================"
