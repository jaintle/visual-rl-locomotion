"""
Frame-stacking wrapper for pixel-based Gymnasium environments.

Stacks the last N consecutive CHW pixel frames along the channel axis,
giving the policy access to motion information without modifying the
encoder architecture.

For a single-frame observation of shape (C, H, W) and n_frames=4, the
stacked observation is (4*C, H, W) — i.e. (12, H, W) for RGB input.

On reset(), the internal deque is pre-filled with n_frames copies of the
initial frame so the agent always receives a full stack from the very first
step.  No zero-padding is used; the initial frame is duplicated instead,
which gives the policy a valid (if redundant) input at the episode start.

Phase 6 restriction: frame stacking is applied only in pixel mode.
The wrapper must be applied on top of PixelObservationWrapper, never
directly on a state-mode environment.
"""

import collections
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FrameStackWrapper(gym.Wrapper):
    """
    Stack the last n_frames pixel observations along the channel axis.

    The wrapped environment must return observations with shape (C, H, W)
    (i.e. CHW float32 as produced by PixelObservationWrapper).

    Stacked observation shape:
        (n_frames * C, H, W)  — oldest frame at channel index 0.

    Args:
        env:      A pixel environment with obs shape (C, H, W).
        n_frames: Number of frames to stack.  Must be >= 2.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)

        if n_frames < 2:
            raise ValueError(
                f"n_frames must be >= 2 for FrameStackWrapper (got {n_frames}). "
                "For n_frames=1 (no stacking), do not apply this wrapper."
            )

        old_space = env.observation_space
        if len(old_space.shape) != 3:
            raise ValueError(
                f"FrameStackWrapper expects a 3-D (C, H, W) observation space, "
                f"got shape {old_space.shape}."
            )

        C, H, W = old_space.shape
        self.n_frames = n_frames
        self._C = C

        # Replace the observation space to reflect the stacked shape.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_frames * C, H, W),
            dtype=np.float32,
        )

        # Ring buffer holding individual frames.  maxlen enforces capacity.
        # Index 0 = oldest frame; index -1 = most recent frame.
        self._frames: collections.deque = collections.deque(maxlen=n_frames)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Reset environment and pre-fill the frame buffer with the initial frame.

        Duplicates the initial observation n_frames times so the first
        call to _get_obs() returns a valid (non-zero-padded) stack.
        """
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step environment, push new frame into buffer, return stacked obs."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Concatenate all buffered frames along the channel axis.

        Returns:
            np.ndarray of shape (n_frames * C, H, W), dtype float32.
            Oldest frame occupies channels [0 : C], newest occupies [-C :].
        """
        assert len(self._frames) == self.n_frames, (
            f"Frame buffer has {len(self._frames)} frames, expected {self.n_frames}."
        )
        return np.concatenate(list(self._frames), axis=0)
