"""Test MT19937 state initialization."""

import numpy as np
import jax.numpy as jnp
import snapszer_base as base
import snapszer_jax as jax_impl

def test_mt_state():
    """Test that MT19937 state is initialized correctly."""
    seed = 42

    print(f"Testing MT19937 state initialization with seed {seed}")
    print("="*60)

    # Base implementation
    base_rng = base._MT19937(seed)
    print("\nBase MT state (first 10 values):")
    for i in range(10):
        print(f"  mt[{i}] = {base_rng._mt[i]}")

    # JAX implementation
    mt = jax_impl.mt19937_init(jnp.int32(seed))
    print("\nJAX MT state (first 10 values):")
    for i in range(10):
        print(f"  mt[{i}] = {int(mt[i])}")

    # Check if they match
    print("\n" + "="*60)
    print("Checking for differences:")
    all_match = True
    for i in range(624):
        if base_rng._mt[i] != int(mt[i]):
            print(f"  MISMATCH at index {i}: base={base_rng._mt[i]}, jax={int(mt[i])}")
            all_match = False
            if i > 20:  # Only show first few mismatches
                print("  ... (more mismatches)")
                break

    if all_match:
        print("  ✓ All MT state values match!")
    else:
        print("  ✗ MT state does not match")

if __name__ == "__main__":
    test_mt_state()
