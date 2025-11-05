"""Profile JAX Snapszer to identify optimization opportunities."""

import jax
import jax.numpy as jnp
from snapszer import jax_impl
import time


def profile_operations():
    """Profile individual operations to find bottlenecks."""
    print("="*60)
    print("PROFILING INDIVIDUAL OPERATIONS")
    print("="*60)

    # Setup test state
    state = jax_impl.new_game(jnp.int32(42))

    # Profile new_game
    print("\n1. new_game (initialization)")
    start = time.time()
    for _ in range(1000):
        s = jax_impl.new_game(jnp.int32(42))
    elapsed = time.time() - start
    print(f"   1000 calls: {elapsed:.4f}s ({elapsed/1000*1000:.3f}ms per call)")

    # Profile legal_actions_mask
    print("\n2. legal_actions_mask")
    start = time.time()
    for _ in range(10000):
        mask = jax_impl.legal_actions_mask(state)
    elapsed = time.time() - start
    print(f"   10000 calls: {elapsed:.4f}s ({elapsed/10000*1000:.3f}ms per call)")

    # Profile apply_action (card play)
    print("\n3. apply_action (card play)")
    mask = jax_impl.legal_actions_mask(state)
    action = jnp.argmax(mask)
    start = time.time()
    for _ in range(10000):
        new_state = jax_impl.apply_action(state, action)
    elapsed = time.time() - start
    print(f"   10000 calls: {elapsed:.4f}s ({elapsed/10000*1000:.3f}ms per call)")

    # Profile observation_tensor
    print("\n4. observation_tensor")
    start = time.time()
    for _ in range(10000):
        obs = jax_impl.observation_tensor(state, jnp.int32(0))
    elapsed = time.time() - start
    print(f"   10000 calls: {elapsed:.4f}s ({elapsed/10000*1000:.3f}ms per call)")

    # Profile remove_card_from_hand
    print("\n5. remove_card_from_hand")
    start = time.time()
    for _ in range(10000):
        s = jax_impl.remove_card_from_hand(state, jnp.int32(0), jnp.int32(5))
    elapsed = time.time() - start
    print(f"   10000 calls: {elapsed:.4f}s ({elapsed/10000*1000:.3f}ms per call)")

    # Profile insert_card_to_hand
    print("\n6. insert_card_to_hand")
    start = time.time()
    for _ in range(10000):
        s = jax_impl.insert_card_to_hand(state, jnp.int32(0), jnp.int32(10))
    elapsed = time.time() - start
    print(f"   10000 calls: {elapsed:.4f}s ({elapsed/10000*1000:.3f}ms per call)")


def profile_batch_sizes():
    """Profile different batch sizes to understand scaling."""
    print("\n" + "="*60)
    print("BATCH SIZE SCALING ANALYSIS")
    print("="*60)

    batch_sizes = [1, 10, 100, 1000, 10000]

    for batch_size in batch_sizes:
        seeds = jnp.arange(batch_size, dtype=jnp.int32)

        # Warmup
        @jax.jit
        def init_batch(seeds):
            return jax.vmap(jax_impl.new_game)(seeds)

        states = init_batch(seeds)
        states.hands.block_until_ready()

        # Time initialization
        start = time.time()
        states = init_batch(seeds)
        states.hands.block_until_ready()
        init_time = time.time() - start

        # Time single step
        @jax.jit
        def step_once(states):
            def pick_action(state):
                legal_mask = jax_impl.legal_actions_mask(state)
                action = jnp.argmax(legal_mask)
                return action
            actions = jax.vmap(pick_action)(states)
            new_states = jax.vmap(jax_impl.apply_action)(states, actions)
            return new_states

        # Warmup
        states = step_once(states)
        states.hands.block_until_ready()

        # Time
        start = time.time()
        for _ in range(10):
            states = step_once(states)
        states.hands.block_until_ready()
        step_time = (time.time() - start) / 10

        print(f"\nBatch size {batch_size:5d}:")
        print(f"  Init:  {init_time*1000:7.3f}ms ({init_time/batch_size*1000000:.2f}us per game)")
        print(f"  Step:  {step_time*1000:7.3f}ms ({step_time/batch_size*1000000:.2f}us per game)")


def analyze_state_size():
    """Analyze memory usage of state."""
    print("\n" + "="*60)
    print("STATE MEMORY ANALYSIS")
    print("="*60)

    state = jax_impl.new_game(jnp.int32(42))

    total_bytes = 0
    print("\nState fields:")
    for field in state._fields:
        val = getattr(state, field)
        if hasattr(val, 'nbytes'):
            print(f"  {field:20s}: {val.nbytes:6d} bytes  (shape={val.shape}, dtype={val.dtype})")
            total_bytes += val.nbytes
        elif hasattr(val, '__len__'):
            # Array-like
            item_bytes = sum(getattr(v, 'nbytes', 0) for v in val)
            print(f"  {field:20s}: {item_bytes:6d} bytes  (array)")
            total_bytes += item_bytes
        else:
            print(f"  {field:20s}: scalar")

    print(f"\nTotal state size: {total_bytes} bytes ({total_bytes/1024:.2f} KB)")
    print(f"Batch of 10000: {total_bytes*10000/1024/1024:.2f} MB")


def profile_compilation():
    """Profile compilation overhead."""
    print("\n" + "="*60)
    print("COMPILATION ANALYSIS")
    print("="*60)

    # Clear compilation cache
    jax.clear_caches()

    # Time first compilation
    print("\nCompiling new_game...")
    start = time.time()
    state = jax_impl.new_game(jnp.int32(42))
    state.hands.block_until_ready()
    compile_time = time.time() - start
    print(f"  First call (with compilation): {compile_time*1000:.2f}ms")

    start = time.time()
    state = jax_impl.new_game(jnp.int32(43))
    state.hands.block_until_ready()
    exec_time = time.time() - start
    print(f"  Second call (cached): {exec_time*1000:.2f}ms")
    print(f"  Compilation overhead: {(compile_time-exec_time)*1000:.2f}ms")


def main():
    print("="*60)
    print("JAX SNAPSZER PROFILING")
    print("="*60)
    print(f"\nDevice: {jax.devices()[0]}")
    print(f"Backend: {jax.default_backend()}")

    profile_operations()
    profile_batch_sizes()
    analyze_state_size()
    profile_compilation()

    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    print("""
Based on profiling, potential optimizations:

1. HOT PATH OPTIMIZATION:
   - Focus on legal_actions_mask and apply_action (called most frequently)
   - Optimize hand operations (remove/insert cards)

2. REDUCE STATE SIZE:
   - Current state is quite large for batching
   - Consider using more compact representations

3. BATCHING:
   - Sweet spot is batch_size >= 1000
   - Ensure training loop uses large batches

4. REDUCE BRANCHES:
   - JAX prefers branchless code
   - Consider replacing some jax.lax.cond with jnp.where

5. SPECIALIZED OPERATIONS:
   - Hand operations could use bitmask-only approach for speed
   - MT19937 is slow - consider JAX native RNG if parity not needed
    """)


if __name__ == "__main__":
    main()
