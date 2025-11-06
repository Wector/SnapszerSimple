"""Information set utilities for CFR."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class InformationSet:
    """
    Stores regret and strategy data for an information set.

    Attributes:
        regret_sum: Cumulative regrets for each action
        strategy_sum: Cumulative strategy weights
        num_updates: Number of times this info set was updated
    """

    regret_sum: np.ndarray  # [num_actions]
    strategy_sum: np.ndarray  # [num_actions]
    num_updates: int = 0

    @staticmethod
    def create(num_actions: int):
        """Create new information set with zeros."""
        return InformationSet(
            regret_sum=np.zeros(num_actions, dtype=np.float64),
            strategy_sum=np.zeros(num_actions, dtype=np.float64),
            num_updates=0
        )


def get_info_set_key(state, player: int) -> bytes:
    """
    Convert game state to information set identifier for player.

    The information set includes everything the player can observe:
    - Their own cards
    - Public information (trump, trick cards, scores, etc.)
    - Action history (implicit in observation_tensor)

    Args:
        state: Game state (SnapszerState)
        player: Player index (0 or 1)

    Returns:
        bytes: Hashable key for dictionary lookup
    """
    from snapszer import jax_optimized

    # Use observation tensor as information set representation
    # This encodes all information visible to the player
    obs = jax_optimized.observation_tensor(state, player)

    # Convert to hashable bytes
    return obs.tobytes()


def get_strategy(regret_sum: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """
    Regret matching: convert regrets to strategy probabilities.

    Args:
        regret_sum: [num_actions] cumulative regrets
        legal_mask: [num_actions] boolean mask of legal actions

    Returns:
        strategy: [num_actions] probability distribution over actions
    """
    # Only use positive regrets
    positive_regrets = np.maximum(0.0, regret_sum)

    # Mask out illegal actions
    masked_regrets = np.where(legal_mask, positive_regrets, 0.0)

    # Normalize to probability distribution
    total = np.sum(masked_regrets)
    if total > 0:
        strategy = masked_regrets / total
    else:
        # If no positive regrets, use uniform over legal actions
        num_legal = np.sum(legal_mask)
        if num_legal > 0:
            strategy = np.where(legal_mask, 1.0 / num_legal, 0.0)
        else:
            # Edge case: no legal actions (shouldn't happen)
            strategy = np.zeros_like(regret_sum)

    return strategy


def get_average_strategy(info_set: InformationSet) -> np.ndarray:
    """
    Extract average strategy from accumulated strategy weights.

    This is the Nash equilibrium approximation.

    Args:
        info_set: Information set with strategy_sum

    Returns:
        Average strategy as probability distribution
    """
    total = np.sum(info_set.strategy_sum)

    if total > 0:
        return info_set.strategy_sum / total
    else:
        # If never updated, return uniform
        num_actions = len(info_set.strategy_sum)
        return np.ones(num_actions) / num_actions
