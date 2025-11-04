"""
Test MT19937 implementation to verify it matches the base.
"""

import jax.numpy as jnp
import snapszer_base as base
import snapszer_jax as jax_env


def test_mt19937_extract():
    """Test that MT19937 extracts match base."""
    seed = 42

    # Base MT19937
    base_rng = base._MT19937(seed)

    # JAX MT19937
    jax_mt = jax_env.mt19937_init(jnp.uint32(seed))
    jax_index = jnp.uint32(624)

    # Extract 10 random numbers
    print("Testing MT19937 extraction:")
    for i in range(10):
        base_val = base_rng.rand_uint32()

        jax_val, jax_mt, jax_index = jax_env.mt19937_extract(jax_mt, jax_index)
        jax_val = int(jax_val)

        print(f"  {i}: base={base_val}, jax={jax_val}, match={base_val == jax_val}")

        if base_val != jax_val:
            print(f"    MISMATCH at step {i}!")
            return False

    print("  All extractions match!")
    return True


def test_mt19937_random_double():
    """Test random double generation."""
    seed = 42

    base_rng = base._MT19937(seed)

    jax_mt = jax_env.mt19937_init(jnp.uint32(seed))
    jax_index = jnp.uint32(624)

    print("\nTesting MT19937 random_double:")
    for i in range(5):
        base_val = base_rng.random_double()

        jax_val, jax_mt, jax_index = jax_env.mt19937_random_double(jax_mt, jax_index)
        jax_val = float(jax_val)

        print(f"  {i}: base={base_val:.15f}, jax={jax_val:.15f}, diff={abs(base_val - jax_val):.2e}")

        if abs(base_val - jax_val) > 1e-10:
            print(f"    MISMATCH at step {i}!")
            return False

    print("  All random doubles match!")
    return True


def test_shuffle():
    """Test deck shuffling."""
    seed = 42

    # Base shuffle
    base_deck = list(range(20))
    base._mt_shuffle(base_deck, seed)

    # JAX shuffle
    jax_deck = jax_env.mt19937_shuffle(jnp.uint32(seed))
    jax_deck = [int(x) for x in jax_deck]

    print("\nTesting shuffle:")
    print(f"  Base: {base_deck}")
    print(f"  JAX:  {jax_deck}")
    print(f"  Match: {base_deck == jax_deck}")

    return base_deck == jax_deck


if __name__ == "__main__":
    test_mt19937_extract()
    test_mt19937_random_double()
    test_shuffle()
