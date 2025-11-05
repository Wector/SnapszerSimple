"""Test MT19937 RNG implementation."""

import numpy as np
import jax.numpy as jnp
from snapszer import base
from snapszer import jax_impl

def test_mt19937_outputs():
    """Test that MT19937 generates the same random numbers."""
    seed = 42

    print(f"Testing MT19937 with seed {seed}")
    print("="*60)

    # Base implementation
    base_rng = base._MT19937(seed)
    print("\nBase MT19937 outputs:")
    for i in range(10):
        val = base_rng.rand_uint32()
        print(f"  {i}: {val}")

    # JAX implementation
    mt = jax_impl.mt19937_init(jnp.int32(seed))
    index = jnp.int32(0)
    print("\nJAX MT19937 outputs:")
    for i in range(10):
        mt, index, val = jax_impl.mt19937_extract(mt, index)
        print(f"  {i}: {int(val)}")

    # Test random doubles
    print("\n" + "="*60)
    print("Testing random_double:")
    print("="*60)

    base_rng = base._MT19937(seed)
    print("\nBase random_double outputs:")
    for i in range(10):
        val = base_rng.random_double()
        print(f"  {i}: {val}")

    mt = jax_impl.mt19937_init(jnp.int32(seed))
    index = jnp.int32(0)
    print("\nJAX random_double outputs:")
    for i in range(10):
        mt, index, val = jax_impl.mt19937_random_double(mt, index)
        print(f"  {i}: {float(val)}")

if __name__ == "__main__":
    test_mt19937_outputs()
