"""
Phase 1 smoke test: environment creation and pixel wrapper.

Verifies:
    - make_env() constructs a working environment in both modes.
    - State observations have the expected shape.
    - Pixel observations are (3, img_size, img_size), float32, in [0, 1].
    - Ten random steps complete without error.
    - In pixel mode, the first frame is saved to assets/smoke_frame.png.

Usage:
    # State mode
    python scripts/smoke_env.py --obs_mode state

    # Pixel mode
    python scripts/smoke_env.py --obs_mode pixels --img_size 64
"""

import argparse
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 environment smoke test.")
    parser.add_argument("--env_id", type=str, default="Hopper-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="state",
        choices=["state", "pixels"],
    )
    parser.add_argument("--img_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Phase 1 Smoke Test")
    print(f"  env_id   : {args.env_id}")
    print(f"  seed     : {args.seed}")
    print(f"  obs_mode : {args.obs_mode}")
    if args.obs_mode == "pixels":
        print(f"  img_size : {args.img_size}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Import here so import errors surface with a clear message.
    # ------------------------------------------------------------------ #
    try:
        from visual_rl_locomotion.envs.make_env import make_env
    except ImportError as exc:
        print(f"[ERROR] Could not import make_env: {exc}")
        print("Ensure the package is installed: pip install -e .")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Create environment.
    # ------------------------------------------------------------------ #
    print("\n[1] Creating environment...")
    env = make_env(
        env_id=args.env_id,
        seed=args.seed,
        obs_mode=args.obs_mode,
        img_size=args.img_size,
    )
    print(f"    observation_space : {env.observation_space}")
    print(f"    action_space      : {env.action_space}")

    # ------------------------------------------------------------------ #
    # Reset.
    # ------------------------------------------------------------------ #
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=args.seed)
    print(f"    initial obs shape : {obs.shape}")
    print(f"    initial obs dtype : {obs.dtype}")

    # ------------------------------------------------------------------ #
    # Mode-specific assertions on initial observation.
    # ------------------------------------------------------------------ #
    if args.obs_mode == "state":
        assert obs.ndim == 1, f"State obs should be 1-D, got shape {obs.shape}."
        print(f"    [OK] State observation shape: {obs.shape}")

    elif args.obs_mode == "pixels":
        expected_shape = (3, args.img_size, args.img_size)
        assert obs.shape == expected_shape, (
            f"Pixel obs shape mismatch: expected {expected_shape}, got {obs.shape}."
        )
        assert obs.dtype == np.float32, (
            f"Pixel obs dtype must be float32, got {obs.dtype}."
        )
        assert obs.min() >= 0.0 and obs.max() <= 1.0, (
            f"Pixel obs values out of [0, 1]: min={obs.min():.4f}, max={obs.max():.4f}."
        )
        print(f"    [OK] Pixel observation shape : {obs.shape}")
        print(f"    [OK] Pixel observation dtype : {obs.dtype}")
        print(f"    [OK] Pixel value range       : [{obs.min():.4f}, {obs.max():.4f}]")

        # Save first frame to assets/.
        _save_pixel_frame(obs, args.img_size)

    # ------------------------------------------------------------------ #
    # Step 10 random actions.
    # ------------------------------------------------------------------ #
    print("\n[3] Stepping 10 random actions...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(
            f"    step {step + 1:2d} | obs shape: {obs.shape}"
            f" | reward: {reward:+.4f}"
            f" | done: {done}"
        )

        # Per-step assertions.
        if args.obs_mode == "pixels":
            assert obs.shape == (3, args.img_size, args.img_size), (
                f"Pixel shape changed mid-episode at step {step + 1}: {obs.shape}."
            )
            assert obs.dtype == np.float32
            assert 0.0 <= obs.min() and obs.max() <= 1.0

        if done:
            obs, info = env.reset()

    # ------------------------------------------------------------------ #
    # Cleanup.
    # ------------------------------------------------------------------ #
    env.close()
    print("\n[4] Environment closed cleanly.")
    print("\n" + "=" * 60)
    print("Smoke test PASSED.")
    print("=" * 60)


def _save_pixel_frame(obs_chw: np.ndarray, img_size: int) -> None:
    """Save first pixel observation as PNG for visual inspection."""
    from PIL import Image

    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
    )
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, "smoke_frame.png")

    # CHW float32 [0,1] -> HWC uint8 [0,255]
    frame_hwc = np.transpose(obs_chw, (1, 2, 0))
    frame_uint8 = (frame_hwc * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(frame_uint8)
    img.save(save_path)
    print(f"    [OK] Saved smoke frame: {save_path}  ({img_size}x{img_size} px)")


if __name__ == "__main__":
    main()
