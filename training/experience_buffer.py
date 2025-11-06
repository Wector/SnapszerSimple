"""Experience buffer for Neural CFR training."""

import numpy as np
import jax.numpy as jnp
from typing import Tuple


class ExperienceBuffer:
    """
    Circular buffer for storing CFR training samples.

    Stores (observation, legal_mask, strategy) tuples collected during
    CFR traversals. Used to train the policy network.
    """

    def __init__(self, capacity: int, obs_size: int, num_actions: int):
        """
        Initialize buffer.

        Args:
            capacity: Maximum number of samples to store
            obs_size: Size of observation vector
            num_actions: Number of possible actions
        """
        self.capacity = capacity
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Preallocate arrays
        self.observations = np.zeros((capacity, obs_size), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.strategies = np.zeros((capacity, num_actions), dtype=np.float32)

        self.size = 0  # Current number of samples
        self.ptr = 0   # Write pointer

    def add(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        strategy: np.ndarray
    ):
        """
        Add a single sample to the buffer.

        Args:
            observation: [obs_size] observation vector
            legal_mask: [num_actions] boolean mask
            strategy: [num_actions] strategy distribution
        """
        self.observations[self.ptr] = observation
        self.legal_masks[self.ptr] = legal_mask
        self.strategies[self.ptr] = strategy

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        observations: np.ndarray,
        legal_masks: np.ndarray,
        strategies: np.ndarray
    ):
        """
        Add multiple samples at once.

        Args:
            observations: [batch, obs_size]
            legal_masks: [batch, num_actions]
            strategies: [batch, num_actions]
        """
        batch_size = len(observations)

        for i in range(batch_size):
            self.add(observations[i], legal_masks[i], strategies[i])

    def sample(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            (observations, legal_masks, strategies) as JAX arrays
        """
        # Sample random indices
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        # Convert to JAX arrays
        obs = jnp.array(self.observations[indices])
        masks = jnp.array(self.legal_masks[indices])
        strats = jnp.array(self.strategies[indices])

        return obs, masks, strats

    def get_all(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get all samples in the buffer.

        Returns:
            (observations, legal_masks, strategies) as JAX arrays
        """
        obs = jnp.array(self.observations[:self.size])
        masks = jnp.array(self.legal_masks[:self.size])
        strats = jnp.array(self.strategies[:self.size])

        return obs, masks, strats

    def clear(self):
        """Clear the buffer."""
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.size >= self.capacity
