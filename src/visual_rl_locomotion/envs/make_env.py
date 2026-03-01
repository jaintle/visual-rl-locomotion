"""
Environment factory for visual-rl-locomotion.

Supports two observation modes:
    state  - raw proprioceptive vector (default Gymnasium output)
    pixels - RGB frames via PixelObservationWrapper
"""

import gymnasium as gym

from visual_rl_locomotion.envs.pixels import PixelObservationWrapper


def make_env(
    env_id: str,
    seed: int,
    obs_mode: str,
    img_size: int = 64,
) -> gym.Env:
    """
    Create and configure a Gymnasium environment.

    Args:
        env_id:   Gymnasium environment ID, e.g. "Hopper-v4".
        seed:     Integer seed used for env.reset() for reproducibility.
        obs_mode: "state" for proprioceptive observations,
                  "pixels" for pixel observations.
        img_size: Square pixel resolution when obs_mode="pixels".

    Returns:
        A configured gym.Env.  The seed is applied immediately via reset()
        so the first episode is reproducible.  Callers should call
        env.reset(seed=seed) again if they need a fresh episode.

    Raises:
        ValueError: If obs_mode is not one of {"state", "pixels"}.
    """
    if obs_mode not in ("state", "pixels"):
        raise ValueError(
            f"obs_mode must be 'state' or 'pixels', got '{obs_mode}'."
        )

    # Always create with rgb_array so render() works in both modes.
    env = gym.make(env_id, render_mode="rgb_array")

    # Seed the environment.  This call also performs the first reset
    # internally so the RNG is properly initialised.
    env.reset(seed=seed)

    if obs_mode == "pixels":
        env = PixelObservationWrapper(env, img_size=img_size)

    return env
