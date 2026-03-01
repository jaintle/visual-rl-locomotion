"""
Multi-seed comparison runner for visual-rl-locomotion.

Trains PPO in both obs_mode=state and obs_mode=pixels over a fixed set of
seeds by launching train_ppo.py as a subprocess for each (mode, seed) pair.
Using subprocesses avoids shared state and makes each run fully isolated.

Output layout:
    <out_dir>/
        state/
            seed_0/  config.json  metrics.csv  checkpoints/
            seed_1/  ...
            seed_2/  ...
        pixels/
            seed_0/  ...
            ...

Usage (quick smoke run):
    python scripts/run_compare.py \\
        --env_id Hopper-v4 \\
        --total_timesteps 2000 --eval_every 1000 \\
        --n_steps 256 --epochs 2 --batch_size 64 \\
        --device cpu --img_size 64 \\
        --seeds 0,1,2 --out_dir runs/compare/demo
"""

import argparse
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO for multiple seeds in state and pixel modes."
    )

    # Environment
    parser.add_argument("--env_id",           type=str,   default="Hopper-v4")
    parser.add_argument("--img_size",         type=int,   default=64)

    # Training budget
    parser.add_argument("--total_timesteps",  type=int,   default=100_000)
    parser.add_argument("--eval_every",       type=int,   default=10_000)
    parser.add_argument("--n_steps",          type=int,   default=2048)
    parser.add_argument("--epochs",           type=int,   default=10)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--n_eval_episodes",  type=int,   default=3,
                        help="Deterministic eval episodes per checkpoint.")

    # Hardware
    parser.add_argument("--device",           type=str,   default="cpu",
                        choices=["cpu", "cuda"])

    # Experiment configuration
    parser.add_argument("--seeds",            type=str,   default="0,1,2",
                        help="Comma-separated seed integers, e.g. '0,1,2'.")
    parser.add_argument("--out_dir",          type=str,   default="runs/compare/demo")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single-run launcher
# ---------------------------------------------------------------------------

def run_one(
    mode: str,
    seed: int,
    save_dir: str,
    args: argparse.Namespace,
    train_script: str,
) -> None:
    """
    Launch one train_ppo.py subprocess for a given (mode, seed) pair.

    Args:
        mode:         "state" or "pixels".
        seed:         Integer random seed.
        save_dir:     Directory for this run's outputs.
        args:         Parsed namespace from run_compare's arg parser.
        train_script: Absolute path to scripts/train_ppo.py.
    """
    cmd = [
        sys.executable, train_script,
        "--env_id",          args.env_id,
        "--obs_mode",        mode,
        "--seed",            str(seed),
        "--total_timesteps", str(args.total_timesteps),
        "--eval_every",      str(args.eval_every),
        "--n_steps",         str(args.n_steps),
        "--epochs",          str(args.epochs),
        "--batch_size",      str(args.batch_size),
        "--device",          args.device,
        "--img_size",        str(args.img_size),
        "--n_eval_episodes", str(args.n_eval_episodes),
        "--save_dir",        save_dir,
    ]

    print(f"\n{'=' * 64}")
    print(f"  [run_compare] mode={mode}  seed={seed}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 64}")

    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"  [run_compare] Finished in {elapsed:.1f}s → {save_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    modes = ["state", "pixels"]

    # Resolve train_ppo.py path relative to this script's directory.
    scripts_dir   = os.path.dirname(os.path.abspath(__file__))
    train_script  = os.path.join(scripts_dir, "train_ppo.py")

    if not os.path.isfile(train_script):
        print(f"[ERROR] train_ppo.py not found at: {train_script}", file=sys.stderr)
        sys.exit(1)

    total_runs = len(modes) * len(seeds)
    run_idx    = 0

    wall_start = time.time()

    for mode in modes:
        for seed in seeds:
            run_idx += 1
            save_dir = os.path.join(args.out_dir, mode, f"seed_{seed}")
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n[{run_idx}/{total_runs}] mode={mode}  seed={seed}")
            run_one(mode, seed, save_dir, args, train_script)

    wall_elapsed = time.time() - wall_start
    print(f"\n{'=' * 64}")
    print(f"  All {total_runs} runs complete in {wall_elapsed:.1f}s.")
    print(f"  Results → {args.out_dir}/")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
