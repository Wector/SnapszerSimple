# Optimized JAX Implementation - Code Review

## Summary

The user made excellent additional optimizations to `jax_optimized.py`:
- ✅ Direct bitmask operations
- ✅ `jnp.select` instead of nested conditionals
- ✅ 0.050ms legal_actions_mask (even faster!)
- ✅ Parity test passes 100/100 seeds

## Issues Found

### 1. Dead Code in `can_exchange_trump_jack` (PERFORMANCE)

**Location:** [src/snapszer/jax_optimized.py:229-240](src/snapszer/jax_optimized.py#L229-L240)

**Problem:** Code computes `higher_count` but never uses it

**Current Code:**
```python
def count_higher(i):
    card = state.hands[player, i]
    is_valid = (i < state.hand_sizes[player]) & (card >= 0)
    is_trump_suit = cid_suit(card) == trump_suit
    is_higher = cid_rank(card) < trump_value
    return is_valid & is_trump_suit & is_higher

higher_count = jnp.sum(jax.vmap(count_higher)(jnp.arange(MAX_HAND_SIZE)))
# higher_count is never used!
```

**Fix:** Remove lines 229-240 entirely

**Corrected Function:**
```python
@jax.jit
def can_exchange_trump_jack(state: SnapszerState, player: jnp.ndarray) -> jnp.ndarray:
    """Check if player can exchange trump jack."""
    jack_cid = state.trump * NUM_RANKS + TRUMP_JACK_RANK
    has_jack = mask_contains(state.hand_masks[player], jack_cid)

    return (
        has_jack &
        ~state.closed &
        ~state.trump_taken &
        (state.trick_cards[0] == -1) &
        (state.leader == player) &
        (state.stock_idx < len(state.stock))
    )
```

**Impact:** Faster trump jack check (removes unnecessary vmap loop)

### 2. Parity Test Could Be More Comprehensive

**Location:** [tests/test_parity_optimized.py](tests/test_parity_optimized.py)

**Current Approach:**
- ✅ Tests legal actions match base implementation
- ✅ Uses random actions to explore game tree
- ✅ 100 seeds for good coverage

**Missing:**
- ⚠️ Doesn't explicitly test trump jack exchange scenarios
- ⚠️ Doesn't verify state transitions match (only actions)
- ⚠️ No edge case tests (e.g., closing talon with exactly 3 cards left)

**Recommendation:**
Add targeted test cases for:
1. Trump jack exchange conditions
2. Closing talon edge cases
3. Marriage scoring in various states

**Example Addition:**
```python
def test_trump_jack_exchange():
    """Test trump jack exchange is available when expected."""
    # Setup state where player has trump jack, is leader, stock not empty
    base_state = base.SnapszerState.new(seed=42)

    # Force player 0 to be leader with trump jack
    # ... setup code ...

    base_actions = base_state.legal_actions()
    jax_state = convert_state(base_state)
    jax_mask = jax_optimized.legal_actions_mask(jax_state)

    base_has_exchange = EXCHANGE_TRUMP_ACTION in base_actions
    jax_has_exchange = jax_mask[EXCHANGE_TRUMP_ACTION]

    assert base_has_exchange == jax_has_exchange, \
        f"Trump jack exchange mismatch: base={base_has_exchange}, jax={jax_has_exchange}"
```

## Performance Impact

### Before Dead Code Removal:
- `legal_actions_mask`: 0.050ms per call
- `can_exchange_trump_jack`: includes unnecessary vmap loop

### After Dead Code Removal (estimated):
- `legal_actions_mask`: ~0.045-0.048ms per call (5-10% faster)
- Cleaner code, easier to maintain

## Overall Assessment

**Grade: A-** (would be A+ after fixing dead code)

The optimizations are well-designed and effective. The main issue is the dead code that wastes computation cycles. The parity test is functional but could be more thorough for edge cases.

## Recommended Actions

1. **High Priority:** Remove dead code in `can_exchange_trump_jack` (lines 229-240)
2. **Medium Priority:** Add edge case tests to parity suite
3. **Low Priority:** Consider adding pytest parametrize for more systematic testing

## Final Thoughts

These optimizations successfully improve upon the already-fast JAX implementation. The bitmask operations and `jnp.select` usage show good understanding of JAX performance patterns. Well done!
