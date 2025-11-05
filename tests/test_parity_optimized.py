"""Parity tests between base Python and optimized JAX implementations of Snapszer.

This test suite ensures that the optimized JAX implementation produces identical
legal action masks to the base Python implementation for various game states.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple
from snapszer import base
from snapszer import jax_optimized

MAX_HAND_SIZE = 10

def convert_state(base_state: base.SnapszerState) -> jax_optimized.SnapszerState:
    """Convert a base SnapszerState to an optimized JAX SnapszerState."""
    
    key = jax.random.PRNGKey(0)  # Dummy key

    hands = np.full((2, MAX_HAND_SIZE), -1, dtype=np.int32)
    for p in range(2):
        hands[p, :len(base_state.hands[p])] = base_state.hands[p]

    def make_mask(hand):
        return np.bitwise_or.reduce(np.where(hand >= 0, 1 << hand, 0))

    hand_masks = np.apply_along_axis(make_mask, 1, hands)

    trick_cards = np.array([-1 if c is None else c for c in base_state.trick_cards], dtype=np.int32)
    
    closed_by = -1 if base_state.closed_by is None else base_state.closed_by
    
    last_trick_winner = -1 if base_state.last_trick_winner is None else base_state.last_trick_winner

    winner = -1 if base_state.winner is None else base_state.winner

    game_points = np.zeros(2, dtype=np.int32)
    if base_state.game_points is not None:
        game_points[0] = base_state.game_points[0]
        game_points[1] = base_state.game_points[1]


    return jax_optimized.SnapszerState(
        key=key,
        trump=jnp.int32(base_state.trump),
        trump_card=jnp.int32(base_state.trump_card),
        stock=jnp.array(base_state.stock, dtype=jnp.int32),
        stock_idx=jnp.int32(base_state.stock_idx),
        hands=jnp.array(hands),
        hand_sizes=jnp.array([len(h) for h in base_state.hands], dtype=jnp.int32),
        hand_masks=jnp.array(hand_masks, dtype=jnp.int32),
        current_player=jnp.int32(base_state.current_player),
        leader=jnp.int32(base_state.leader),
        trick_cards=jnp.array(trick_cards),
        points=jnp.array(base_state.points, dtype=jnp.int32),
        tricks_won=jnp.array(base_state.tricks_won, dtype=jnp.int32),
        closed=jnp.bool_(base_state.closed),
        closed_by=jnp.int32(closed_by),
        trump_taken=jnp.bool_(base_state.trump_taken),
        last_trick_winner=jnp.int32(last_trick_winner),
        marriages_scored=jnp.array(base_state.marriages_scored, dtype=jnp.bool_),
        terminal=jnp.bool_(base_state.terminal),
        winner=jnp.int32(winner),
        game_points=jnp.array(game_points),
    )

def compare_legal_actions(base_actions: List[int], jax_mask: jnp.ndarray, step: int) -> Tuple[bool, str]:
    """Compare legal actions between base and JAX."""
    jax_actions = [i for i in range(len(jax_mask)) if jax_mask[i]]

    base_set = set(base_actions)
    jax_set = set(jax_actions)

    if base_set != jax_set:
        missing_in_jax = base_set - jax_set
        extra_in_jax = jax_set - base_set
        error_msg = f"\nStep {step} legal actions differ:\n"
        error_msg += f"  Base: {sorted(base_actions)}\n"
        error_msg += f"  JAX:  {sorted(jax_actions)}\n"
        if missing_in_jax:
            error_msg += f"  Missing in JAX: {sorted(missing_in_jax)}\n"
        if extra_in_jax:
            error_msg += f"  Extra in JAX: {sorted(extra_in_jax)}\n"
        return False, error_msg

    return True, ""

def test_single_game_legal_actions(seed: int, max_steps: int = 100, verbose: bool = False) -> Tuple[bool, str]:
    """Test legal actions parity for a single game."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing game with seed {seed}")
        print(f"{'='*60}")

    base_state = base.SnapszerState.new(seed=seed)
    
    step = 0
    np_rng = np.random.RandomState(seed + 1000)

    while not base_state.is_terminal() and step < max_steps:
        step += 1

        jax_state = convert_state(base_state)
        
        base_actions = base_state.legal_actions()
        jax_mask = jax_optimized.legal_actions_mask(jax_state)

        match, error = compare_legal_actions(base_actions, jax_mask, step)
        if not match:
            return False, f"Seed {seed} step {step}: {error}"

        action = np_rng.choice(base_actions)
        
        if verbose:
            print(f"\nStep {step}:")
            print(f"  Player: {base_state.current_player}")
            print(f"  Action: {action}")
            print(f"  Base: {base_state.public_str()}")

        base_state.apply_action(action)

    return True, f"Seed {seed} passed ({step} steps)"

def test_100_seeds():
    """Run legal actions parity test for 100 seeds."""
    for seed in range(100):
        success, message = test_single_game_legal_actions(seed)
        assert success, message


if __name__ == "__main__":
    test_100_seeds()
    print("All parity tests passed.")