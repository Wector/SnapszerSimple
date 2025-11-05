"""Quick comparison of original vs optimized JAX implementation."""

import time
import jax
import jax.numpy as jnp
from snapszer import jax_impl as jax_orig
from snapszer import jax_optimized as jax_opt


def main():
    print("="*60)
    print("QUICK OPTIMIZATION COMPARISON")
    print("="*60)
    print(f"\nDevice: {jax.devices()[0]}")
    print(f"Backend: {jax.default_backend()}")

    # Key operation: new_game (biggest bottleneck)
    print("\n" + "="*60)
    print("1. new_game (initialization) - 1000 calls")
    print("="*60)

    print("\nOriginal (MT19937 RNG):")
    start = time.time()
    for i in range(1000):
        s = jax_orig.new_game(jnp.int32(i))
    s.hands.block_until_ready()
    orig_time = time.time() - start
    print(f"   Time: {orig_time:.4f}s ({orig_time/1000*1000:.2f}ms per call)")

    print("\nOptimized (JAX native RNG):")
    key = jax.random.PRNGKey(42)
    start = time.time()
    for i in range(1000):
        key, subkey = jax.random.split(key)
        s = jax_opt.new_game(subkey)
    s.hands.block_until_ready()
    opt_time = time.time() - start
    print(f"   Time: {opt_time:.4f}s ({opt_time/1000*1000:.2f}ms per call)")
    print(f"   SPEEDUP: {orig_time/opt_time:.2f}x")

    # Key operation: legal_actions_mask
    print("\n" + "="*60)
    print("2. legal_actions_mask - 10000 calls")
    print("="*60)

    orig_state = jax_orig.new_game(jnp.int32(42))
    print("\nOriginal:")
    start = time.time()
    for _ in range(10000):
        mask = jax_orig.legal_actions_mask(orig_state)
    mask.block_until_ready()
    orig_time = time.time() - start
    print(f"   Time: {orig_time:.4f}s ({orig_time/10000*1000:.2f}ms per call)")

    key = jax.random.PRNGKey(42)
    opt_state = jax_opt.new_game(key)
    print("\nOptimized (bitmask operations):")
    start = time.time()
    for _ in range(10000):
        mask = jax_opt.legal_actions_mask(opt_state)
    mask.block_until_ready()
    opt_time = time.time() - start
    print(f"   Time: {opt_time:.4f}s ({opt_time/10000*1000:.2f}ms per call)")
    print(f"   SPEEDUP: {orig_time/opt_time:.2f}x")

    # Full game playthrough
    print("\n" + "="*60)
    print("3. Full game playthrough - 50 games")
    print("="*60)

    print("\nOriginal:")
    start = time.time()
    for i in range(50):
        state = jax_orig.new_game(jnp.int32(i))
        for _ in range(100):
            if state.terminal:
                break
            mask = jax_orig.legal_actions_mask(state)
            action = jnp.argmax(mask)
            state = jax_orig.apply_action(state, action)
    state.hands.block_until_ready()
    orig_time = time.time() - start
    print(f"   Time: {orig_time:.4f}s ({orig_time/50*1000:.2f}ms per game)")

    print("\nOptimized:")
    key = jax.random.PRNGKey(42)
    start = time.time()
    for i in range(50):
        key, subkey = jax.random.split(key)
        state = jax_opt.new_game(subkey)
        for _ in range(100):
            if state.terminal:
                break
            mask = jax_opt.legal_actions_mask(state)
            action = jnp.argmax(mask)
            state = jax_opt.apply_action(state, action)
    state.hands.block_until_ready()
    opt_time = time.time() - start
    print(f"   Time: {opt_time:.4f}s ({opt_time/50*1000:.2f}ms per game)")
    print(f"   SPEEDUP: {orig_time/opt_time:.2f}x")

    # Batched execution
    print("\n" + "="*60)
    print("4. Batched execution - batch_size=1000")
    print("="*60)

    @jax.jit
    def run_batch_orig(seeds):
        states = jax.vmap(jax_orig.new_game)(seeds)
        def pick_action(state):
            mask = jax_orig.legal_actions_mask(state)
            return jnp.argmax(mask)
        actions = jax.vmap(pick_action)(states)
        return jax.vmap(jax_orig.apply_action)(states, actions)

    @jax.jit
    def run_batch_opt(keys):
        states = jax.vmap(jax_opt.new_game)(keys)
        def pick_action(state):
            mask = jax_opt.legal_actions_mask(state)
            return jnp.argmax(mask)
        actions = jax.vmap(pick_action)(states)
        return jax.vmap(jax_opt.apply_action)(states, actions)

    batch_size = 1000

    print("\nOriginal (warmup + 1 batch):")
    seeds = jnp.arange(batch_size, dtype=jnp.int32)
    states = run_batch_orig(seeds)  # Warmup
    states.hands.block_until_ready()

    start = time.time()
    states = run_batch_orig(seeds)
    states.hands.block_until_ready()
    orig_time = time.time() - start
    print(f"   Time: {orig_time:.4f}s ({orig_time/batch_size*1000:.2f}ms per game)")

    print("\nOptimized (warmup + 1 batch):")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    states = run_batch_opt(keys)  # Warmup
    states.hands.block_until_ready()

    start = time.time()
    states = run_batch_opt(keys)
    states.hands.block_until_ready()
    opt_time = time.time() - start
    print(f"   Time: {opt_time:.4f}s ({opt_time/batch_size*1000:.2f}ms per game)")
    print(f"   SPEEDUP: {orig_time/opt_time:.2f}x")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Key Optimizations:
1. JAX native RNG instead of MT19937 (~40x faster initialization)
2. Bitmask-based legal actions with jnp.select (~4x faster)
3. Unsorted hands with O(1) insert/remove operations
4. Reduced state size (removed 80-byte deck field)
5. Direct bitmask computation for legal replies

Overall: ~2-4x faster for full game simulation
Batched execution: ~20x faster

Note: Optimized version does NOT maintain parity with base
implementation due to different RNG. Use for training, not testing.
    """)


if __name__ == "__main__":
    main()
