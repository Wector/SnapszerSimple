# JAX Snapszer Optimization Summary

## Performance Improvements

### Key Speedups Achieved:
- **new_game (initialization)**: **41.8x faster** (25ms → 0.6ms)
- **legal_actions_mask**: **3.6x faster** (0.22ms → 0.06ms)
- **Full game simulation**: **2.15x faster** (67ms → 31ms per game)
- **Batched execution (1000 games)**: **18.8x faster**

## Optimizations Implemented

### 1. **JAX Native RNG Instead of MT19937** (~41x speedup on initialization)
**Original:**
```python
# Uses custom MT19937 implementation for parity with base
mt = mt19937_init(seed)
index = jnp.int32(624)
deck = mt_shuffle(deck, mt, index)  # Slow!
```

**Optimized:**
```python
# Uses JAX's optimized built-in RNG
deck = jax.random.permutation(key, jnp.arange(NUM_CARDS))  # Fast!
```

**Impact:** MT19937 shuffle was the biggest bottleneck. JAX's native RNG is GPU-optimized and much faster.

**Tradeoff:** Loses parity with base implementation, but that's okay for training.

### 2. **Unsorted Hands** (slight speedup on insert/remove)
**Original:**
```python
def insert_card_to_hand(state, player, cid):
    # Find insertion position (sorted order)
    insert_pos = jnp.sum(counts)  # O(n) scan
    # Shift cards right from insert_pos
    new_hand = jax.vmap(shift_right)(...)  # O(n) shift
```

**Optimized:**
```python
def insert_card_to_hand(state, player, cid):
    # Simply append to end - O(1)!
    new_hand = hand.at[hand_size].set(cid)
```

**For removal:**
```python
def remove_card_from_hand(state, player, cid):
    # Swap-and-pop (move last card to removed position)
    # O(1) instead of O(n) shift
    last_card = hand[last_idx]
    new_hand = jnp.where(idx == card_idx, last_card, hand)
```

**Impact:** Hands don't need to be sorted for correctness, only for display.

### 3. **Bitmask-Based Legal Actions** (~3.6x speedup)
**Original:**
```python
def legal_reply_cards(hand, hand_size, lead_cid, trump):
    # Uses fori_loop to iterate over hand
    def process_card(i):
        card = hand_full[i]
        is_valid = card >= 0
        ...
    obs = jax.lax.fori_loop(0, MAX_HAND_SIZE, process_card, obs)
```

**Optimized:**
```python
def legal_reply_cards_mask(hand_mask, hand_size, lead_cid, trump):
    # Uses bitmask directly
    def check_card(cid):
        has_card = mask_contains(hand_mask, cid)
        ...
    all_cids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    has_card, same_suit, beating, is_trump = jax.vmap(check_card)(all_cids)
```

**Impact:** Vectorized operations over all 20 possible cards instead of iterating over hand.

### 4. **Reduced State Size**
**Removed fields:**
- `deck`: 80 bytes (was only used during initialization, not needed in state)

**Before:** 295 bytes per state
**After:** ~215 bytes per state

**Impact:** Faster state copying, better cache utilization in batched execution.

### 5. **More Branchless Operations**
Replaced some `jax.lax.cond` with `jnp.where` for simpler branching patterns:

```python
# Before (nested cond)
return jax.lax.cond(terminal, lambda s: s, continue_fn, s)

# After (branchless where possible)
new_winner = jnp.where(should_finalize, winner, state.winner)
```

**Impact:** Reduced compilation overhead and slightly faster execution.

## Usage

### For Parity Testing (use original):
```python
import snapszer_jax as jax_impl

# Uses MT19937, maintains exact parity with snapszer_base.py
state = jax_impl.new_game(jnp.int32(seed))
```

### For Training (use optimized):
```python
import snapszer_jax_optimized as jax_impl

# Faster RNG, optimized for speed
key = jax.random.PRNGKey(42)
state = jax_impl.new_game(key)
```

## Benchmark Results

### Sequential Execution (50 games):
- **Original**: 66.99ms per game
- **Optimized**: 31.22ms per game
- **Speedup**: 2.15x

### Batched Execution (1000 games):
- **Original**: 0.028ms per game
- **Optimized**: 0.0015ms per game
- **Speedup**: 18.75x

### Projected Training Performance

With batch_size=10,000 (optimal from previous benchmarks):
- **Original**: ~6,600 games/second
- **Optimized (estimated)**: ~125,000 games/second

For Nash Equilibrium training:
- 1 million games: **8 seconds** (vs ~2.5 minutes with original)
- 1 billion games: **~2.2 hours** (vs ~42 hours with original)

## Files

- `snapszer_jax.py` - Original implementation (parity-tested, slower)
- `snapszer_jax_optimized.py` - Optimized implementation (faster, for training)
- `test_parity.py` - Parity tests (use original implementation only)
- `test_quick_comparison.py` - Performance comparison
- `test_batch_speed.py` - Batch performance benchmarks

## Recommendations

1. **Use `snapszer_jax_optimized.py` for all training** - significantly faster
2. **Keep `snapszer_jax.py` for correctness verification** - parity-tested
3. **Use batch sizes >= 1000** for optimal GPU utilization
4. **For SF-OCR training**, the optimized version will allow much faster iteration

## Next Steps for Training

With the optimized implementation ready, you can now:

1. Implement SF-OCR algorithm using batched game execution
2. Leverage ~125k games/second throughput for rapid training
3. Use large batch sizes (10k+) for best performance
4. Run extensive training runs to find Nash Equilibrium
