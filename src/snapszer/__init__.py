"""Hungarian Snapszer card game implementation with JAX support.

This package provides:
- base: Pure Python reference implementation
- jax_impl: JAX implementation (parity-tested with base)
- jax_optimized: Optimized JAX implementation for training (~3-40x faster)

Example:
    >>> from snapszer import jax_optimized
    >>> import jax.random as random
    >>>
    >>> key = random.PRNGKey(42)
    >>> state = jax_optimized.new_game(key)
    >>> print(state.terminal)
    False
"""

from . import base
from . import jax_impl
from . import jax_optimized

__version__ = "0.1.0"
__all__ = ["base", "jax_impl", "jax_optimized"]
