"""
Environment factory for visual-rl-locomotion.

Supports two observation modes:
    state  - raw proprioceptive vector (default Gymnasium output)
    pixels - RGB frames via PixelObservationWrapper

Phase 6: optional frame stacking applied on top of pixel mode via
FrameStackWrapper (pixels mode only, frame_stack >= 2).
"""

import gymnasium as gym

from visual_rl_locomotion.envs.pixels import PixelObservationWrapper
from visual_rl_locomotion.envs.frame_stack import FrameStackWrapper


def make_env(
    env_id: str,
    seed: int,
    obs_mode: str,
    img_size: int = 64,
    frame_stack: int = 1,
) -> gym.Env:
    """
    Create and configure a Gymnasium environment.

    Args:
        env_id:      Gymnasium environment ID, e.g. "Hopper-v4".
        seed:        Integer seed used for env.reset() for reproducibility.
        obs_mode:    "state" for proprioceptive observations,
                     "pixels" for pixel observations.
        img_size:    Square pixel resolution when obs_mode="pixels".
        frame_stack: Number of consecutive frames to stack along the channel
                     axis (pixels mode only).  1 = no stacking (default).
                     Must be >= 1.  Values >= 2 apply FrameStackWrapper,
                     changing obs shape from (3, H, W) to (frame_stack*3, H, W).

    Returns:
        A configured gym.Env.  The seed is applied immediately via reset()
        so the first episode is reproducible.  Callers should call
        env.reset(seed=seed) again if they need a fresh episode.

    Raises:
        ValueError: If obs_mode is not one of {"state", "pixels"}.
        ValueError: If frame_stack < 1.
        ValueError: If frame_stack > 1 and obs_mode == "state".
    """
    if obs_mode not in ("state", "pixels"):
        raise ValueError(
            f"obs_mode must be 'state' or 'pixels', got '{obs_mode}'."
        )
    if frame_stack < 1:
        raise ValueError(f"frame_stack must be >= 1, got {frame_stack}.")
    if frame_stack > 1 and obs_mode == "state":
        raise ValueError(
            "frame_stack > 1 is only supported in pixel mode."
        )

    # Always create with rgb_array so render() works in both modes.
    env = gym.make(env_id, render_mode="rgb_array")

    # Seed the environment.  This call also performs the first reset
    # internally so the RNG is properly initialised.
    env.reset(seed=seed)

    if obs_mode == "pixels":
        env = PixelObservationWrapper(env, img_size=img_size)
        if frame_stack >= 2:
            env = FrameStackWrapper(env, n_frames=frame_stack)

    return env
