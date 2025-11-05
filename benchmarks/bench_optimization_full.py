"""Compare performance of original vs optimized JAX implementation."""

import time
import jax
import jax.numpy as jnp
from snapszer import jax_impl as jax_orig
from snapszer import jax_optimized as jax_opt


def benchmark_single(n_games=100):
    """Benchmark single game execution."""
    print("\n" + "="*60)
    print("SINGLE GAME BENCHMARK")
    print("="*60)

    # Original implementation
    print("\n1. Original (MT19937 RNG, sorted hands):")
    start = time.time()
    for i in range(n_games):
        state = jax_orig.new_game(jnp.int32(i))
        for _ in range(100):
            if state.terminal:
                break
            mask = jax_orig.legal_actions_mask(state)
            action = jnp.argmax(mask)
            state = jax_orig.apply_action(state, action)
    orig_time = time.time() - start
    print(f"   {n_games} games: {orig_time:.4f}s ({orig_time/n_games*1000:.2f}ms per game)")

    # Optimized implementation
    print("\n2. Optimized (JAX RNG, unsorted hands):")
    key = jax.random.PRNGKey(42)
    start = time.time()
    for i in range(n_games):
        key, subkey = jax.random.split(key)
        state = jax_opt.new_game(subkey)
        for _ in range(100):
            if state.terminal:
                break
            mask = jax_opt.legal_actions_mask(state)
            action = jnp.argmax(mask)
            state = jax_opt.apply_action(state, action)
    opt_time = time.time() - start
    print(f"   {n_games} games: {opt_time:.4f}s ({opt_time/n_games*1000:.2f}ms per game)")

    speedup = orig_time / opt_time
    print(f"\n   Speedup: {speedup:.2f}x")
    return speedup


def benchmark_batch(batch_size=1000, n_batches=10):
    """Benchmark batched execution."""
    print("\n" + "="*60)
    print(f"BATCHED BENCHMARK (batch_size={batch_size})")
    print("="*60)

    # Original implementation
    print("\n1. Original implementation:")

    @jax.jit
    def run_batch_orig(seeds):
        states = jax.vmap(jax_orig.new_game)(seeds)

        def step_batch(states):
            def pick_action(state):
                mask = jax_orig.legal_actions_mask(state)
                return jnp.argmax(mask)
            actions = jax.vmap(pick_action)(states)
            return jax.vmap(jax_orig.apply_action)(states, actions)

        for _ in range(100):
            states = step_batch(states)
        return states

    # Warmup
    seeds = jnp.arange(batch_size, dtype=jnp.int32)
    states = run_batch_orig(seeds)
    states.hands.block_until_ready()

    # Benchmark
    start = time.time()
    for i in range(n_batches):
        seeds = jnp.arange(i * batch_size, (i + 1) * batch_size, dtype=jnp.int32)
        states = run_batch_orig(seeds)
        states.hands.block_until_ready()
    orig_time = time.time() - start

    total_games = n_batches * batch_size
    print(f"   {total_games} games: {orig_time:.4f}s ({total_games/orig_time:.2f} games/sec)")

    # Optimized implementation
    print("\n2. Optimized implementation:")

    @jax.jit
    def run_batch_opt(keys):
        states = jax.vmap(jax_opt.new_game)(keys)

        def step_batch(states):
            def pick_action(state):
                mask = jax_opt.legal_actions_mask(state)
                return jnp.argmax(mask)
            actions = jax.vmap(pick_action)(states)
            return jax.vmap(jax_opt.apply_action)(states, actions)

        for _ in range(100):
            states = step_batch(states)
        return states

    # Warmup
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    states = run_batch_opt(keys)
    states.hands.block_until_ready()

    # Benchmark
    start = time.time()
    for i in range(n_batches):
        key = jax.random.PRNGKey(i)
        keys = jax.random.split(key, batch_size)
        states = run_batch_opt(keys)
        states.hands.block_until_ready()
    opt_time = time.time() - start

    print(f"   {total_games} games: {opt_time:.4f}s ({total_games/opt_time:.2f} games/sec)")

    speedup = orig_time / opt_time
    print(f"\n   Speedup: {speedup:.2f}x")
    return speedup


def benchmark_operations():
    """Benchmark individual operations."""
    print("\n" + "="*60)
    print("OPERATION-LEVEL BENCHMARK")
    print("="*60)

    # Setup test states
    orig_state = jax_orig.new_game(jnp.int32(42))
    key = jax.random.PRNGKey(42)
    opt_state = jax_opt.new_game(key)

    operations = []

    # Test new_game
    print("\n1. new_game")
    start = time.time()
    for i in range(1000):
        s = jax_orig.new_game(jnp.int32(i))
    orig_time = time.time() - start
    print(f"   Original: {orig_time:.4f}s ({orig_time/1000*1000:.3f}ms per call)")

    key = jax.random.PRNGKey(42)
    start = time.time()
    for i in range(1000):
        key, subkey = jax.random.split(key)
        s = jax_opt.new_game(subkey)
    opt_time = time.time() - start
    print(f"   Optimized: {opt_time:.4f}s ({opt_time/1000*1000:.3f}ms per call)")
    print(f"   Speedup: {orig_time/opt_time:.2f}x")
    operations.append(("new_game", orig_time/opt_time))

    # Test remove_card_from_hand
    print("\n2. remove_card_from_hand")
    start = time.time()
    for _ in range(10000):
        s = jax_orig.remove_card_from_hand(orig_state, jnp.int32(0), jnp.int32(5))
    orig_time = time.time() - start
    print(f"   Original: {orig_time:.4f}s ({orig_time/10000*1000:.3f}ms per call)")

    start = time.time()
    for _ in range(10000):
        s = jax_opt.remove_card_from_hand(opt_state, jnp.int32(0), jnp.int32(5))
    opt_time = time.time() - start
    print(f"   Optimized: {opt_time:.4f}s ({opt_time/10000*1000:.3f}ms per call)")
    print(f"   Speedup: {orig_time/opt_time:.2f}x")
    operations.append(("remove_card", orig_time/opt_time))

    # Test insert_card_to_hand
    print("\n3. insert_card_to_hand")
    start = time.time()
    for _ in range(10000):
        s = jax_orig.insert_card_to_hand(orig_state, jnp.int32(0), jnp.int32(10))
    orig_time = time.time() - start
    print(f"   Original: {orig_time:.4f}s ({orig_time/10000*1000:.3f}ms per call)")

    start = time.time()
    for _ in range(10000):
        s = jax_opt.insert_card_to_hand(opt_state, jnp.int32(0), jnp.int32(10))
    opt_time = time.time() - start
    print(f"   Optimized: {opt_time:.4f}s ({opt_time/10000*1000:.3f}ms per call)")
    print(f"   Speedup: {orig_time/opt_time:.2f}x")
    operations.append(("insert_card", orig_time/opt_time))

    # Test legal_actions_mask
    print("\n4. legal_actions_mask")
    start = time.time()
    for _ in range(10000):
        mask = jax_orig.legal_actions_mask(orig_state)
    orig_time = time.time() - start
    print(f"   Original: {orig_time:.4f}s ({orig_time/10000*1000:.3f}ms per call)")

    start = time.time()
    for _ in range(10000):
        mask = jax_opt.legal_actions_mask(opt_state)
    opt_time = time.time() - start
    print(f"   Optimized: {opt_time:.4f}s ({opt_time/10000*1000:.3f}ms per call)")
    print(f"   Speedup: {orig_time/opt_time:.2f}x")
    operations.append(("legal_actions_mask", orig_time/opt_time))

    # Test apply_action
    print("\n5. apply_action")
    mask = jax_orig.legal_actions_mask(orig_state)
    action = jnp.argmax(mask)
    start = time.time()
    for _ in range(10000):
        s = jax_orig.apply_action(orig_state, action)
    orig_time = time.time() - start
    print(f"   Original: {orig_time:.4f}s ({orig_time/10000*1000:.3f}ms per call)")

    mask = jax_opt.legal_actions_mask(opt_state)
    action = jnp.argmax(mask)
    start = time.time()
    for _ in range(10000):
        s = jax_opt.apply_action(opt_state, action)
    opt_time = time.time() - start
    print(f"   Optimized: {opt_time:.4f}s ({opt_time/10000*1000:.3f}ms per call)")
    print(f"   Speedup: {orig_time/opt_time:.2f}x")
    operations.append(("apply_action", orig_time/opt_time))

    return operations


def main():
    print("="*60)
    print("OPTIMIZATION SPEEDUP COMPARISON")
    print("="*60)
    print(f"\nDevice: {jax.devices()[0]}")
    print(f"Backend: {jax.default_backend()}")

    # Run benchmarks
    operations = benchmark_operations()
    single_speedup = benchmark_single(n_games=100)

    batch_speedups = []
    for batch_size in [100, 1000, 10000]:
        speedup = benchmark_batch(batch_size=batch_size, n_batches=10)
        batch_speedups.append((batch_size, speedup))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nOperation-level speedups:")
    for op_name, speedup in operations:
        print(f"  {op_name:20s}: {speedup:5.2f}x")

    print(f"\nSingle game speedup: {single_speedup:.2f}x")

    print("\nBatch speedups:")
    for batch_size, speedup in batch_speedups:
        print(f"  Batch {batch_size:5d}: {speedup:5.2f}x")

    print("\n" + "="*60)
    print("KEY OPTIMIZATIONS:")
    print("="*60)
    print("""
1. JAX native RNG instead of MT19937 (huge speedup on new_game)
2. Unsorted hands with O(1) insert (vs O(n) sorted insert)
3. Swap-and-pop for remove (vs sorted array shift)
4. Reduced state size (removed deck field, 80 bytes saved)
5. More bitmask operations (faster membership tests)

Note: Optimized version does NOT maintain parity with base
implementation due to different RNG and hand ordering.
Use for training only, not for parity testing.
    """)


if __name__ == "__main__":
    main()
