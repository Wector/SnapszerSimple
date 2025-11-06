"""Policy network for Neural CFR using JAX/Flax."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Neural network that maps observations to action probabilities.

    Architecture:
    - Input: observation tensor (80 features)
    - Hidden layers: MLP with residual connections
    - Output: action logits (22 actions)

    The network learns to approximate the CFR strategy across all info sets.
    """

    hidden_sizes: Tuple[int, ...] = (256, 256, 128)

    @nn.compact
    def __call__(self, observation: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            observation: [batch, 80] observation tensor
            legal_mask: [batch, 22] boolean mask of legal actions

        Returns:
            strategy: [batch, 22] probability distribution over actions
        """
        x = observation

        # Initial projection
        x = nn.Dense(self.hidden_sizes[0])(x)
        x = nn.relu(x)

        # Residual blocks
        for hidden_size in self.hidden_sizes[1:]:
            residual = x

            # Two-layer residual block
            x = nn.Dense(hidden_size)(x)
            x = nn.relu(x)
            x = nn.Dense(hidden_size)(x)

            # Residual connection (with projection if size changes)
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size)(residual)
            x = nn.relu(x + residual)

        # Output layer (logits)
        logits = nn.Dense(22)(x)

        # Mask illegal actions with large negative value
        masked_logits = jnp.where(
            legal_mask,
            logits,
            jnp.full_like(logits, -1e9)
        )

        # Convert to probabilities
        strategy = jax.nn.softmax(masked_logits, axis=-1)

        return strategy

    def compute_loss(
        self,
        params,
        observations: jnp.ndarray,
        legal_masks: jnp.ndarray,
        target_strategies: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute cross-entropy loss between predicted and target strategies.

        Args:
            params: Network parameters
            observations: [batch, 80] observations
            legal_masks: [batch, 22] legal action masks
            target_strategies: [batch, 22] target strategy distributions

        Returns:
            loss: Scalar loss value
        """
        # Forward pass
        predicted_strategies = self.apply(params, observations, legal_masks)

        # Cross-entropy loss (only on legal actions)
        # -sum(target * log(predicted))
        epsilon = 1e-8  # Numerical stability
        loss = -jnp.sum(
            target_strategies * jnp.log(predicted_strategies + epsilon) * legal_masks,
            axis=-1
        )

        return jnp.mean(loss)


def create_policy_network(rng_key: jax.random.PRNGKey, hidden_sizes: Tuple[int, ...] = (256, 256, 128)):
    """
    Create and initialize a policy network.

    Args:
        rng_key: JAX random key
        hidden_sizes: Sizes of hidden layers

    Returns:
        (network, params): Network module and initialized parameters
    """
    from snapszer import jax_optimized as game

    network = PolicyNetwork(hidden_sizes=hidden_sizes)

    # Initialize with dummy input
    dummy_obs = jnp.zeros((1, game.OBSERVATION_SIZE), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, game.TOTAL_ACTIONS), dtype=jnp.bool_)

    params = network.init(rng_key, dummy_obs, dummy_mask)

    return network, params
