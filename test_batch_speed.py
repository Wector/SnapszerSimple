"""Test batch performance for JAX Snapszer implementation."""

import time
import jax
import jax.numpy as jnp
import snapszer_jax as jax_impl


def test_single_game(seed: int = 42, max_steps: int = 100):
    """Play a single game to completion."""
    state = jax_impl.new_game(jnp.int32(seed))

    for _ in range(max_steps):
        if state.terminal:
            break

        # Get legal actions and pick first one
        legal_mask = jax_impl.legal_actions_mask(state)
        action = jnp.argmax(legal_mask)

        # Apply action
        state = jax_impl.apply_action(state, action)

    return state


def test_batched_games(batch_size: int, max_steps: int = 100):
    """Play multiple games in parallel using vmap."""
    # Create batch of seeds
    seeds = jnp.arange(batch_size, dtype=jnp.int32)

    # Initialize all games
    @jax.jit
    def init_batch(seeds):
        return jax.vmap(jax_impl.new_game)(seeds)

    states = init_batch(seeds)

    # Play all games step by step
    @jax.jit
    def step_batch(states):
        def pick_action(state):
            legal_mask = jax_impl.legal_actions_mask(state)
            action = jnp.argmax(legal_mask)
            return action

        actions = jax.vmap(pick_action)(states)
        new_states = jax.vmap(jax_impl.apply_action)(states, actions)
        return new_states

    for _ in range(max_steps):
        states = step_batch(states)
        # Check if all games are done
        if jnp.all(states.terminal):
            break

    return states


def benchmark_single():
    """Benchmark single game execution."""
    print("\n" + "="*60)
    print("SINGLE GAME BENCHMARK")
    print("="*60)

    # Warmup
    print("\nWarming up...")
    for i in range(5):
        state = test_single_game(seed=i)

    # Benchmark
    n_games = 100
    print(f"\nRunning {n_games} games sequentially...")
    start = time.time()

    for i in range(n_games):
        state = test_single_game(seed=i)

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Games/sec: {n_games/elapsed:.2f}")
    print(f"  Time per game: {elapsed/n_games*1000:.2f}ms")

    return n_games / elapsed


def benchmark_batch(batch_size: int):
    """Benchmark batched game execution."""
    print("\n" + "="*60)
    print(f"BATCHED GAMES BENCHMARK (batch_size={batch_size})")
    print("="*60)

    # Warmup
    print("\nWarming up JIT compilation...")
    states = test_batched_games(batch_size, max_steps=10)
    print(f"  Warmup batch completed")

    # Benchmark
    n_batches = 10
    total_games = n_batches * batch_size

    print(f"\nRunning {n_batches} batches ({total_games} total games)...")
    start = time.time()

    for i in range(n_batches):
        seed_offset = i * batch_size
        seeds = jnp.arange(seed_offset, seed_offset + batch_size, dtype=jnp.int32)

        # Initialize
        @jax.jit
        def init_batch(seeds):
            return jax.vmap(jax_impl.new_game)(seeds)

        states = init_batch(seeds)

        # Play
        @jax.jit
        def step_batch(states):
            def pick_action(state):
                legal_mask = jax_impl.legal_actions_mask(state)
                action = jnp.argmax(legal_mask)
                return action

            actions = jax.vmap(pick_action)(states)
            new_states = jax.vmap(jax_impl.apply_action)(states, actions)
            return new_states

        for _ in range(100):
            states = step_batch(states)
            if jnp.all(states.terminal):
                break

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Games/sec: {total_games/elapsed:.2f}")
    print(f"  Time per game: {elapsed/total_games*1000:.2f}ms")
    print(f"  Speedup vs single: {(total_games/elapsed) / single_rate:.2f}x")

    return total_games / elapsed


def benchmark_compilation_overhead():
    """Measure JIT compilation overhead."""
    print("\n" + "="*60)
    print("COMPILATION OVERHEAD TEST")
    print("="*60)

    batch_size = 32

    # First run (includes compilation)
    print("\nFirst run (with compilation)...")
    start = time.time()
    states = test_batched_games(batch_size, max_steps=100)
    first_run = time.time() - start
    print(f"  Time: {first_run:.4f}s")

    # Second run (compiled)
    print("\nSecond run (already compiled)...")
    start = time.time()
    states = test_batched_games(batch_size, max_steps=100)
    second_run = time.time() - start
    print(f"  Time: {second_run:.4f}s")

    # Compilation overhead
    overhead = first_run - second_run
    print(f"\nCompilation overhead: {overhead:.4f}s ({overhead/first_run*100:.1f}% of first run)")


def main():
    """Run all benchmarks."""
    print("="*60)
    print("JAX SNAPSZER BATCH PERFORMANCE TEST")
    print("="*60)

    # Check device
    print(f"\nDevice: {jax.devices()[0]}")
    print(f"Backend: {jax.default_backend()}")

    # Single game benchmark
    global single_rate
    single_rate = benchmark_single()

    # Compilation overhead
    benchmark_compilation_overhead()

    # Batched benchmarks
    batch_sizes = [10, 32, 100, 1000, 10000]
    rates = []

    for batch_size in batch_sizes:
        rate = benchmark_batch(batch_size)
        rates.append(rate)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nSingle game rate: {single_rate:.2f} games/sec")
    print(f"\nBatched rates:")
    for batch_size, rate in zip(batch_sizes, rates):
        speedup = rate / single_rate
        print(f"  Batch {batch_size:5d}: {rate:8.2f} games/sec ({speedup:5.2f}x speedup)")

    # Find optimal batch size
    max_idx = rates.index(max(rates))
    optimal_batch = batch_sizes[max_idx]
    print(f"\nOptimal batch size: {optimal_batch} ({rates[max_idx]:.2f} games/sec)")


if __name__ == "__main__":
    main()
