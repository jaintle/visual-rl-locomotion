"""
Pixel observation wrapper for Gymnasium environments.

Converts raw RGB renders into float32 CHW tensors in [0, 1].
No frame stacking; that is deferred to a later phase.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image


class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a Gymnasium environment so that observations are pixel frames
    instead of state vectors.

    The underlying environment must support render_mode="rgb_array".

    Observation format:
        - Shape : (3, img_size, img_size)
        - Dtype : float32
        - Range : [0.0, 1.0]
        - Order : CHW  (channels-first, compatible with PyTorch Conv2d)
    """

    def __init__(self, env: gym.Env, img_size: int = 64):
        """
        Args:
            env:      A Gymnasium env created with render_mode="rgb_array".
            img_size: Height and width of the square output frame (pixels).
        """
        super().__init__(env)

        if env.render_mode != "rgb_array":
            raise ValueError(
                "PixelObservationWrapper requires render_mode='rgb_array', "
                f"got render_mode='{env.render_mode}'."
            )

        self.img_size = img_size

        # Replace the observation space to reflect the pixel shape.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, img_size, img_size),
            dtype=np.float32,
        )

    def observation(self, obs):
        """
        Discard the state observation; render and return pixel frame.

        Args:
            obs: The original state observation (ignored).

        Returns:
            np.ndarray of shape (3, img_size, img_size), dtype float32, [0, 1].
        """
        # Render to an (H, W, 3) uint8 RGB array.
        frame = self.env.render()  # ndarray uint8, shape (H, W, 3)

        # Resize to (img_size, img_size) using PIL for deterministic results.
        img = Image.fromarray(frame)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        frame_resized = np.asarray(img, dtype=np.float32)  # (H, W, 3)

        # Normalise to [0, 1].
        frame_normalised = frame_resized / 255.0

        # Convert HWC -> CHW.
        frame_chw = np.transpose(frame_normalised, (2, 0, 1))  # (3, H, W)

        return frame_chw
