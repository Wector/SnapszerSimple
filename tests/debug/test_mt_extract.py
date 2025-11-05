"""Test MT19937 extraction."""

import numpy as np
import jax.numpy as jnp
from snapszer import base
from snapszer import jax_impl

def test_mt_extract():
    """Test MT19937 value extraction."""
    seed = 42

    print(f"Testing MT19937 extraction with seed {seed}")
    print("="*60)

    # Base implementation - manual extraction
    base_rng = base._MT19937(seed)

    print("\nBase implementation:")
    print(f"Initial index: {base_rng._index}")

    # Extract first value
    val1 = base_rng.rand_uint32()
    print(f"First value: {val1}")
    print(f"Index after first: {base_rng._index}")

    # JAX implementation
    print("\nJAX implementation:")
    mt = jax_impl.mt19937_init(jnp.int32(seed))
    index = jnp.int32(0)
    print(f"Initial index: {int(index)}")

    # Extract first value
    mt_new, index_new, val1_jax = jax_impl.mt19937_extract(mt, index)
    print(f"First value: {int(val1_jax)}")
    print(f"Index after first: {int(index_new)}")

    # Let's manually check the extraction logic
    print("\n" + "="*60)
    print("Manual extraction check:")

    # Get MT value at index 0
    base_y = base_rng._mt[0]
    jax_y = mt[0]

    print(f"MT[0] base: {base_y}, jax: {int(jax_y)}")

    # Apply tempering
    y = base_y
    print(f"Initial y: {y}")

    y ^= (y >> 11)
    print(f"After >> 11: {y}")

    y ^= (y << 7) & 0x9D2C5680
    print(f"After << 7 & 0x9D2C5680: {y}")

    y ^= (y << 15) & 0xEFC60000
    print(f"After << 15 & 0xEFC60000: {y}")

    y ^= (y >> 18)
    print(f"After >> 18: {y}")

    print(f"\nExpected base result: {val1}")
    print(f"Manual result: {y & 0xFFFFFFFF}")

    # Now do the same for JAX
    print("\n" + "="*60)
    print("JAX tempering check:")

    y_jax = jnp.uint32(jax_y)
    print(f"Initial y: {int(y_jax)}")

    y_jax = y_jax ^ (y_jax >> jnp.uint32(11))
    print(f"After >> 11: {int(y_jax)}")

    y_jax = y_jax ^ ((y_jax << jnp.uint32(7)) & jnp.uint32(0x9D2C5680))
    print(f"After << 7 & 0x9D2C5680: {int(y_jax)}")

    y_jax = y_jax ^ ((y_jax << jnp.uint32(15)) & jnp.uint32(0xEFC60000))
    print(f"After << 15 & 0xEFC60000: {int(y_jax)}")

    y_jax = y_jax ^ (y_jax >> jnp.uint32(18))
    print(f"After >> 18: {int(y_jax)}")

    print(f"\nExpected JAX result: {int(val1_jax)}")
    print(f"Manual result: {int(y_jax)}")

if __name__ == "__main__":
    test_mt_extract()
