"""
PPO training script for visual-rl-locomotion — Phase 3 (state + pixel observations).

Usage:
    # State mode
    python scripts/train_ppo.py \\
        --env_id Hopper-v4 --obs_mode state \\
        --total_timesteps 2000 --eval_every 1000 --seed 0 \\
        --save_dir runs/smoke_state --device cpu \\
        --n_steps 256 --batch_size 64 --epochs 2

    # Pixel mode
    python scripts/train_ppo.py \\
        --env_id Hopper-v4 --obs_mode pixels --img_size 64 \\
        --total_timesteps 2000 --eval_every 1000 --seed 0 \\
        --save_dir runs/smoke_pixels --device cpu \\
        --n_steps 256 --batch_size 64 --epochs 2

Produces under save_dir/:
    config.json            — full hyperparameter record
    metrics.csv            — one row per update cycle + one per eval
    checkpoints/           — .pt files at each eval boundary
"""

import argparse
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a Gymnasium locomotion task (state mode)."
    )

    # Environment
    parser.add_argument("--env_id",           type=str,   default="Hopper-v4")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--obs_mode",         type=str,   default="state",
                        choices=["state", "pixels"],
                        help="Observation mode: 'state' (MLP policy) or 'pixels' (CNN policy).")
    parser.add_argument("--img_size",         type=int,   default=64,
                        help="Pixel frame size (ignored in state mode).")
    parser.add_argument("--frame_stack",      type=int,   default=1,
                        help="Frames to stack along channel axis (pixels only). "
                             "1 = no stacking (default). Use 4 for Phase 6.")

    # Training
    parser.add_argument("--total_timesteps",  type=int,   default=200_000)
    parser.add_argument("--n_steps",          type=int,   default=2048,
                        help="Steps collected per rollout before each update.")
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--epochs",           type=int,   default=10,
                        help="PPO update epochs per rollout.")
    parser.add_argument("--lr",              type=float,  default=3e-4)
    parser.add_argument("--gamma",           type=float,  default=0.99)
    parser.add_argument("--gae_lambda",      type=float,  default=0.95)
    parser.add_argument("--clip_range",      type=float,  default=0.2)
    parser.add_argument("--ent_coef",        type=float,  default=0.0)
    parser.add_argument("--vf_coef",         type=float,  default=0.5)
    parser.add_argument("--max_grad_norm",   type=float,  default=0.5)

    # Evaluation & I/O
    parser.add_argument("--eval_every",       type=int,   default=10_000,
                        help="Evaluate every this many environment steps.")
    parser.add_argument("--n_eval_episodes",  type=int,   default=5,
                        help="Number of deterministic episodes per evaluation.")
    parser.add_argument("--save_dir",         type=str,   default="runs/run1")
    parser.add_argument("--device",           type=str,   default="cpu",
                        choices=["cpu", "cuda", "mps"])

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(agent, env_id: str, seed: int, obs_mode: str, img_size: int,
             n_episodes: int, device: torch.device,
             frame_stack: int = 1) -> list:
    """
    Run n_episodes deterministic episodes and return the list of episode returns.

    Works for both PPOAgent (state) and VisionPPOAgent (pixels): both expose
    agent.policy.get_action_and_logp(obs_t, deterministic=True).

    A separate environment is created so the training env's state is never
    perturbed.  obs_t is unsqueezed to (1, *obs.shape) which handles both
    flat state vectors and CHW pixel tensors correctly.
    """
    from visual_rl_locomotion.envs.make_env import make_env

    eval_env = make_env(env_id, seed=seed + 1000, obs_mode=obs_mode,
                        img_size=img_size, frame_stack=frame_stack)
    returns = []

    agent.policy.eval()

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + 1000 + ep)
        done = False
        ep_return = 0.0

        while not done:
            # Works for state (1, obs_dim) and pixels (1, 3, H, W).
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.policy.get_action_and_logp(
                    obs_t, deterministic=True
                )
            obs, reward, terminated, truncated, _ = eval_env.step(
                action.squeeze(0).cpu().numpy()
            )
            done = bool(terminated or truncated)
            ep_return += float(reward)

        returns.append(ep_return)

    eval_env.close()
    return returns


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Imports (inside main so import errors are surfaced cleanly) ---
    from visual_rl_locomotion.algo.ppo import PPOAgent, compute_gae
    from visual_rl_locomotion.envs.make_env import make_env
    from visual_rl_locomotion.models.vision_policy import VisionPPOAgent
    from visual_rl_locomotion.utils.config import args_to_dict, save_config
    from visual_rl_locomotion.utils.logger import CSVLogger
    from visual_rl_locomotion.utils.seed import set_seed

    # --- Device validation ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "[error] --device cuda requested but CUDA is not available.\n"
            "        On Apple Silicon use --device mps for GPU acceleration,\n"
            "        or --device cpu to run on the CPU."
        )
        raise SystemExit(1)
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(
            "[error] --device mps requested but MPS is not available.\n"
            "        Requires macOS 12.3+ with an Apple Silicon or AMD GPU.\n"
            "        Fall back with --device cpu."
        )
        raise SystemExit(1)

    # --- Setup ---
    set_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Save config ---
    config_path = os.path.join(args.save_dir, "config.json")
    save_config(args_to_dict(args), config_path)

    # --- Logger ---
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    logger = CSVLogger(csv_path)

    # --- Environment ---
    env = make_env(args.env_id, seed=args.seed, obs_mode=args.obs_mode,
                   img_size=args.img_size, frame_stack=args.frame_stack)

    obs_shape  = env.observation_space.shape   # (obs_dim,) or (3, H, W)
    action_dim = env.action_space.shape[0]

    print("=" * 60)
    print(f"  env_id          : {args.env_id}")
    print(f"  obs_mode        : {args.obs_mode}")
    if args.obs_mode == "pixels":
        print(f"  frame_stack     : {args.frame_stack}")
    print(f"  obs_shape       : {obs_shape}")
    print(f"  action_dim      : {action_dim}")
    print(f"  total_timesteps : {args.total_timesteps}")
    print(f"  n_steps         : {args.n_steps}")
    print(f"  device          : {device}")
    print(f"  save_dir        : {args.save_dir}")
    print("=" * 60)

    # --- Agent: branch on obs_mode ---
    if args.obs_mode == "state":
        agent = PPOAgent(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            lr=args.lr,
            device=device,
        )
    else:
        # pixels: obs_shape is (3, H, W)
        agent = VisionPPOAgent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            lr=args.lr,
            device=device,
        )
        print(f"  [pixels] CNN encoder + VisionPolicy instantiated.")

    # --- Training loop ---
    global_step     = 0
    next_eval_step  = args.eval_every

    # Reset environment; keep the obs live between rollout calls.
    obs, _ = env.reset(seed=args.seed)

    while global_step < args.total_timesteps:

        # ---- Rollout collection ----
        buffer, obs, last_value, ep_returns, ep_lengths = agent.collect_rollout(
            env, args.n_steps, obs
        )
        global_step += args.n_steps

        # ---- GAE ----
        advantages, returns = compute_gae(
            buffer["rewards"],
            buffer["values"],
            buffer["dones"],
            last_value,
            args.gamma,
            args.gae_lambda,
        )
        buffer["advantages"] = advantages
        buffer["returns"]    = returns

        # ---- PPO update ----
        update_metrics = agent.update(
            buffer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
        )

        # ---- Log train metrics ----
        # Log one row per update cycle.  Include episode info from the most
        # recently completed episode in this rollout (if any).
        row = {
            "global_step":  global_step,
            "policy_loss":  round(update_metrics["policy_loss"], 6),
            "value_loss":   round(update_metrics["value_loss"],  6),
            "entropy":      round(update_metrics["entropy"],     6),
            "approx_kl":    round(update_metrics["approx_kl"],  6),
        }
        if ep_returns:
            row["episode_return"] = round(ep_returns[-1], 4)
            row["episode_length"] = ep_lengths[-1]
            mean_ep_return = np.mean(ep_returns)
            print(
                f"  step {global_step:>8d} | "
                f"ep_return {mean_ep_return:>8.2f} | "
                f"policy_loss {update_metrics['policy_loss']:>7.4f} | "
                f"value_loss {update_metrics['value_loss']:>7.4f} | "
                f"kl {update_metrics['approx_kl']:>7.4f}"
            )
        else:
            print(
                f"  step {global_step:>8d} | "
                f"ep_return {'(none)':>8s} | "
                f"policy_loss {update_metrics['policy_loss']:>7.4f} | "
                f"value_loss {update_metrics['value_loss']:>7.4f} | "
                f"kl {update_metrics['approx_kl']:>7.4f}"
            )
        logger.log(row)

        # ---- Evaluation ----
        if global_step >= next_eval_step:
            print(f"\n  [eval] Running {args.n_eval_episodes} deterministic episodes...")
            eval_returns = evaluate(
                agent,
                args.env_id,
                seed=args.seed,
                obs_mode=args.obs_mode,
                img_size=args.img_size,
                n_episodes=args.n_eval_episodes,
                device=device,
                frame_stack=args.frame_stack,
            )
            eval_mean = float(np.mean(eval_returns))
            eval_std  = float(np.std(eval_returns))
            print(
                f"  [eval] step {global_step:>8d} | "
                f"return {eval_mean:.2f} ± {eval_std:.2f}  "
                f"(over {args.n_eval_episodes} episodes)\n"
            )

            logger.log({
                "global_step":      global_step,
                "eval_return_mean": round(eval_mean, 4),
                "eval_return_std":  round(eval_std,  4),
            })

            # Save checkpoint.
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:08d}.pt")
            agent.save_checkpoint(ckpt_path, global_step)

            next_eval_step += args.eval_every

    # ---- Final checkpoint ----
    final_ckpt = os.path.join(ckpt_dir, f"step_{global_step:08d}_final.pt")
    agent.save_checkpoint(final_ckpt, global_step)

    env.close()

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  config   : {config_path}")
    print(f"  metrics  : {csv_path}")
    print(f"  ckpts    : {ckpt_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
