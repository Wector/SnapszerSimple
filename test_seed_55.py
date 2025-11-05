"""Debug test for seed 55 failure."""

import numpy as np
import jax.numpy as jnp
import snapszer_base as base
import snapszer_jax as jax_impl


def test_seed_55_debug():
    """Debug seed 55 step by step."""
    seed = 55

    print(f"Testing seed {seed} step by step...")
    print("="*60)

    # Initialize both
    base_state = base.SnapszerState.new(seed=seed)
    jax_state = jax_impl.new_game(jnp.int32(seed))

    print(f"Initial state:")
    print(f"  Base P0 hand: {base_state.hands[0]}")
    print(f"  JAX P0 hand:  {[int(x) for x in jax_state.hands[0][:jax_state.hand_sizes[0]]]}")
    print(f"  Base P1 hand: {base_state.hands[1]}")
    print(f"  JAX P1 hand:  {[int(x) for x in jax_state.hands[1][:jax_state.hand_sizes[1]]]}")
    print()

    # Play through manually with deterministic action selection
    np_rng = np.random.RandomState(seed + 1000)

    for step in range(1, 11):
        if base_state.is_terminal():
            break

        base_actions = base_state.legal_actions()
        action = np_rng.choice(base_actions)

        print(f"Step {step}: Player {base_state.current_player} plays action {action}")
        print(f"  Before - Base P0: {base_state.hands[0]} (size {len(base_state.hands[0])})")
        print(f"  Before - JAX P0:  {[int(x) for x in jax_state.hands[0][:jax_state.hand_sizes[0]]]} (size {int(jax_state.hand_sizes[0])})")
        print(f"  Before - Base P1: {base_state.hands[1]} (size {len(base_state.hands[1])})")
        print(f"  Before - JAX P1:  {[int(x) for x in jax_state.hands[1][:jax_state.hand_sizes[1]]]} (size {int(jax_state.hand_sizes[1])})")

        # Apply action
        base_state.apply_action(action)
        jax_state = jax_impl.apply_action(jax_state, jnp.int32(action))

        print(f"  After  - Base P0: {base_state.hands[0]} (size {len(base_state.hands[0])})")
        print(f"  After  - JAX P0:  {[int(x) for x in jax_state.hands[0][:jax_state.hand_sizes[0]]]} (size {int(jax_state.hand_sizes[0])})")
        print(f"  After  - Base P1: {base_state.hands[1]} (size {len(base_state.hands[1])})")
        print(f"  After  - JAX P1:  {[int(x) for x in jax_state.hands[1][:jax_state.hand_sizes[1]]]} (size {int(jax_state.hand_sizes[1])})")

        # Check if they match
        base_p0 = base_state.hands[0]
        jax_p0 = [int(x) for x in jax_state.hands[0][:jax_state.hand_sizes[0]]]
        base_p1 = base_state.hands[1]
        jax_p1 = [int(x) for x in jax_state.hands[1][:jax_state.hand_sizes[1]]]

        if base_p0 != jax_p0:
            print(f"  ❌ MISMATCH P0!")
            print(f"     Missing in JAX: {set(base_p0) - set(jax_p0)}")
            print(f"     Extra in JAX: {set(jax_p0) - set(base_p0)}")
            break

        if base_p1 != jax_p1:
            print(f"  ❌ MISMATCH P1!")
            print(f"     Missing in JAX: {set(base_p1) - set(jax_p1)}")
            print(f"     Extra in JAX: {set(jax_p1) - set(base_p1)}")
            break

        print(f"  ✓ Match!")
        print()


if __name__ == "__main__":
    test_seed_55_debug()
