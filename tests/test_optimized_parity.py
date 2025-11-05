"""Parity tests for the optimized JAX implementation of Snapszer.

Since the optimized version uses JAX native RNG and unsorted hands, we can't test
for exact state parity with the base implementation. Instead, we test:
1. Game mechanics consistency (legal moves, terminal states)
2. Scoring and game outcome correctness
3. Statistical parity over many games
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict
from snapszer import base
from snapszer import jax_optimized


def test_game_invariants(state: jax_optimized.SnapszerState, step: int) -> Tuple[bool, str]:
    """Test that game invariants hold for the optimized JAX implementation."""
    errors = []

    # Hand sizes should match actual cards in hands
    for p in range(2):
        actual_size = jnp.sum((state.hands[p] >= 0).astype(jnp.int32))
        if actual_size != state.hand_sizes[p]:
            errors.append(f"P{p} hand size mismatch: declared={state.hand_sizes[p]}, actual={actual_size}")

    # Hand masks should match cards in hands
    for p in range(2):
        expected_mask = 0
        for i in range(state.hand_sizes[p]):
            card = int(state.hands[p, i])
            if card >= 0:
                expected_mask |= (1 << card)
        if expected_mask != int(state.hand_masks[p]):
            errors.append(f"P{p} hand mask mismatch: expected={bin(expected_mask)}, actual={bin(int(state.hand_masks[p]))}")

    # Total cards accounting: 20 total cards initially
    # Cards in state = hands + stock (not yet drawn/discarded) + trump (if not taken) + current trick
    # Cards played in completed tricks = 2 * (tricks_won[0] + tricks_won[1])
    # Cards discarded when talon closed = stock cards + trump card (if not taken before closing)

    cards_in_state = int(state.hand_sizes[0] + state.hand_sizes[1])

    # Stock cards: only count if not drawn and not closed
    # When closed, stock_idx is set to len(stock), so remaining_stock becomes 0
    # which correctly accounts for discarded stock cards
    remaining_stock = jnp.maximum(0, len(state.stock) - state.stock_idx)
    cards_in_state += int(remaining_stock)

    # Trump card: only count if not taken
    if not state.trump_taken:
        cards_in_state += 1

    # Add played cards in current trick
    if state.trick_cards[0] >= 0:
        cards_in_state += 1
    if state.trick_cards[1] >= 0:
        cards_in_state += 1

    # Cards that have been played in completed tricks (discarded from state)
    completed_tricks = int(state.tricks_won[0] + state.tricks_won[1])
    cards_played = completed_tricks * 2

    # Cards discarded when talon was closed (if applicable)
    # When closed, stock_idx = len(stock), so remaining_stock = 0
    # The discarded cards are: (len(stock) - stock_idx_before_close) + (1 if trump not taken)
    # But we already account for this by checking remaining_stock and trump_taken

    # For non-closed games: cards_in_state + cards_played should equal 20
    # For closed games: cards_in_state + cards_played + discarded_stock should equal 20
    # But since we already set remaining_stock=0 when closed, we're already accounting for it

    # Actually, when talon is closed, the remaining cards are just removed from the game
    # So we should only check: cards_in_state + cards_played <= 20
    # and cards_in_state + cards_played == 20 when talon is open

    total_accounted = cards_in_state + cards_played

    # Only check exact 20 if talon was never closed or all cards were drawn before closing
    if not state.closed:
        # Talon is open, all 20 cards should be accounted for
        if total_accounted != 20:
            errors.append(
                f"Total cards != 20 (talon open): {total_accounted} "
                f"(in_state={cards_in_state}, played={cards_played}, "
                f"hands={int(state.hand_sizes[0])}+{int(state.hand_sizes[1])}, "
                f"stock={int(remaining_stock)}, trump={0 if state.trump_taken else 1}, "
                f"trick={int(state.trick_cards[0] >= 0)}+{int(state.trick_cards[1] >= 0)}, "
                f"completed_tricks={completed_tricks})"
            )
    else:
        # Talon was closed, some cards may have been discarded
        if total_accounted > 20:
            errors.append(
                f"Total cards > 20 (talon closed): {total_accounted} "
                f"(in_state={cards_in_state}, played={cards_played})"
            )

    # Points should be non-negative and <= 120
    for p in range(2):
        pts = int(state.points[p])
        if pts < 0 or pts > 120:
            errors.append(f"P{p} points out of range: {pts}")

    # Tricks won should be non-negative and <= 10
    for p in range(2):
        tricks = int(state.tricks_won[p])
        if tricks < 0 or tricks > 10:
            errors.append(f"P{p} tricks won out of range: {tricks}")

    # If terminal, winner should be set
    if state.terminal and int(state.winner) == -1:
        errors.append("Terminal state but no winner set")

    # If terminal, game points should be set
    if state.terminal:
        gp_sum = int(state.game_points[0] + state.game_points[1])
        if gp_sum == 0:
            errors.append("Terminal state but no game points awarded")

    # Current player should be valid
    cp = int(state.current_player)
    if not state.terminal and (cp < 0 or cp > 1):
        errors.append(f"Invalid current player: {cp}")

    if errors:
        error_msg = f"\nStep {step} invariant violations:\n" + "\n".join(f"  - {e}" for e in errors)
        return False, error_msg

    return True, ""


def test_legal_actions_validity(state: jax_optimized.SnapszerState, step: int) -> Tuple[bool, str]:
    """Test that all legal actions are actually valid."""
    if state.terminal:
        return True, ""

    errors = []
    legal_mask = jax_optimized.legal_actions_mask(state)
    legal_actions = [i for i in range(len(legal_mask)) if legal_mask[i]]

    if len(legal_actions) == 0:
        errors.append("No legal actions available in non-terminal state")

    me = int(state.current_player)

    for action in legal_actions:
        # Check card actions
        if action < 20:
            # Should have this card in hand
            if not jax_optimized.mask_contains(state.hand_masks[me], jnp.int32(action)):
                errors.append(f"Action {action} is legal but card not in P{me} hand")

        # Check exchange action
        elif action == 20:
            jack_cid = int(state.trump) * 5 + 4
            if not jax_optimized.mask_contains(state.hand_masks[me], jnp.int32(jack_cid)):
                errors.append(f"Exchange action legal but no trump jack in hand")
            if state.closed or state.trump_taken:
                errors.append(f"Exchange action legal but talon closed or trump taken")

        # Check close action
        elif action == 21:
            if state.closed:
                errors.append(f"Close action legal but already closed")
            if state.stock_idx >= len(state.stock):
                errors.append(f"Close action legal but stock empty")

    if errors:
        error_msg = f"\nStep {step} legal actions validity check failed:\n" + "\n".join(f"  - {e}" for e in errors)
        return False, error_msg

    return True, ""


def play_optimized_game(key: jax.Array, max_steps: int = 100, verbose: bool = False) -> Tuple[jax_optimized.SnapszerState, int, bool]:
    """Play a complete game using the optimized JAX implementation."""
    state = jax_optimized.new_game(key)
    step = 0

    # Use a separate RNG for action selection
    action_key = jax.random.PRNGKey(int(jax.random.randint(key, (), 0, 1000000)))

    while not state.terminal and step < max_steps:
        step += 1

        # Test invariants
        valid, error = test_game_invariants(state, step)
        if not valid:
            if verbose:
                print(error)
            return state, step, False

        # Test legal actions validity
        valid, error = test_legal_actions_validity(state, step)
        if not valid:
            if verbose:
                print(error)
            return state, step, False

        # Get legal actions
        legal_mask = jax_optimized.legal_actions_mask(state)
        legal_actions = jnp.array([i for i in range(len(legal_mask)) if legal_mask[i]], dtype=jnp.int32)

        if len(legal_actions) == 0:
            if verbose:
                print(f"Step {step}: No legal actions available!")
            return state, step, False

        # Choose random legal action
        action_key, subkey = jax.random.split(action_key)
        action_idx = jax.random.randint(subkey, (), 0, len(legal_actions))
        action = legal_actions[action_idx]

        if verbose:
            print(f"Step {step}: P{state.current_player} plays action {action}")

        # Apply action
        state = jax_optimized.apply_action(state, action)

    # Final invariant check
    valid, error = test_game_invariants(state, step)
    if not valid:
        if verbose:
            print(error)
        return state, step, False

    return state, step, True


def compare_game_statistics(base_stats: Dict, opt_stats: Dict, tolerance: float = 0.15) -> Tuple[bool, str]:
    """Compare statistical properties between base and optimized implementations.

    Args:
        base_stats: Statistics from base implementation
        opt_stats: Statistics from optimized implementation
        tolerance: Allowed relative difference (e.g., 0.15 = 15%)
    """
    errors = []

    # Compare win rates (should be roughly equal for both players)
    for p in range(2):
        base_wr = base_stats[f'p{p}_wins'] / base_stats['num_games']
        opt_wr = opt_stats[f'p{p}_wins'] / opt_stats['num_games']

        # Both should be around 0.5 with some variance
        if abs(base_wr - 0.5) > 0.2 or abs(opt_wr - 0.5) > 0.2:
            errors.append(f"P{p} win rate too far from 0.5: base={base_wr:.3f}, opt={opt_wr:.3f}")

    # Compare average game lengths
    base_avg_len = base_stats['total_steps'] / base_stats['num_games']
    opt_avg_len = opt_stats['total_steps'] / opt_stats['num_games']
    rel_diff = abs(base_avg_len - opt_avg_len) / base_avg_len

    if rel_diff > tolerance:
        errors.append(f"Average game length differs too much: base={base_avg_len:.1f}, opt={opt_avg_len:.1f} (diff={rel_diff:.1%})")

    # Compare average game points
    base_avg_gp = base_stats['total_game_points'] / base_stats['num_games']
    opt_avg_gp = opt_stats['total_game_points'] / opt_stats['num_games']
    rel_diff = abs(base_avg_gp - opt_avg_gp) / max(base_avg_gp, 0.1)

    if rel_diff > tolerance:
        errors.append(f"Average game points differs too much: base={base_avg_gp:.2f}, opt={opt_avg_gp:.2f} (diff={rel_diff:.1%})")

    if errors:
        error_msg = "\nStatistical comparison failed:\n" + "\n".join(f"  - {e}" for e in errors)
        return False, error_msg

    return True, ""


def run_statistical_parity_test(num_games: int = 100, verbose: bool = False) -> Tuple[bool, str]:
    """Run statistical parity test comparing base and optimized implementations."""
    if verbose:
        print(f"\nRunning statistical parity test with {num_games} games...")

    # Collect statistics from base implementation
    base_stats = {
        'num_games': num_games,
        'p0_wins': 0,
        'p1_wins': 0,
        'total_steps': 0,
        'total_game_points': 0,
    }

    np_rng = np.random.RandomState(42)

    for i in range(num_games):
        seed = np_rng.randint(0, 1000000)
        state = base.SnapszerState.new(seed=seed)
        action_rng = np.random.RandomState(seed + 1000)

        step = 0
        max_steps = 100

        while not state.is_terminal() and step < max_steps:
            step += 1
            actions = state.legal_actions()
            if not actions:
                break
            action = action_rng.choice(actions)
            state.apply_action(action)

        if state.is_terminal():
            base_stats['total_steps'] += step
            if state.winner == 0:
                base_stats['p0_wins'] += 1
            elif state.winner == 1:
                base_stats['p1_wins'] += 1
            if state.game_points:
                base_stats['total_game_points'] += sum(state.game_points)

    # Collect statistics from optimized implementation
    opt_stats = {
        'num_games': num_games,
        'p0_wins': 0,
        'p1_wins': 0,
        'total_steps': 0,
        'total_game_points': 0,
    }

    np_rng = np.random.RandomState(42)

    for i in range(num_games):
        seed = np_rng.randint(0, 1000000)
        key = jax.random.PRNGKey(seed)
        state, steps, success = play_optimized_game(key, verbose=False)

        if not success:
            return False, f"Optimized game {i} failed validation"

        if state.terminal:
            opt_stats['total_steps'] += steps
            winner = int(state.winner)
            if winner == 0:
                opt_stats['p0_wins'] += 1
            elif winner == 1:
                opt_stats['p1_wins'] += 1
            opt_stats['total_game_points'] += int(state.game_points[0] + state.game_points[1])

    if verbose:
        print(f"\nBase implementation statistics:")
        print(f"  P0 wins: {base_stats['p0_wins']}/{num_games} ({base_stats['p0_wins']/num_games:.1%})")
        print(f"  P1 wins: {base_stats['p1_wins']}/{num_games} ({base_stats['p1_wins']/num_games:.1%})")
        print(f"  Avg steps: {base_stats['total_steps']/num_games:.1f}")
        print(f"  Avg game points: {base_stats['total_game_points']/num_games:.2f}")

        print(f"\nOptimized implementation statistics:")
        print(f"  P0 wins: {opt_stats['p0_wins']}/{num_games} ({opt_stats['p0_wins']/num_games:.1%})")
        print(f"  P1 wins: {opt_stats['p1_wins']}/{num_games} ({opt_stats['p1_wins']/num_games:.1%})")
        print(f"  Avg steps: {opt_stats['total_steps']/num_games:.1f}")
        print(f"  Avg game points: {opt_stats['total_game_points']/num_games:.2f}")

    return compare_game_statistics(base_stats, opt_stats)


def test_single_optimized_game(seed: int, max_steps: int = 100, verbose: bool = False) -> Tuple[bool, str]:
    """Test a single game with the optimized implementation."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing optimized JAX game with seed {seed}")
        print(f"{'='*60}")

    key = jax.random.PRNGKey(seed)
    state, steps, success = play_optimized_game(key, max_steps, verbose)

    if not success:
        return False, f"Game validation failed at step {steps}"

    if not state.terminal:
        return False, f"Game did not terminate within {max_steps} steps"

    # Verify final state
    winner = int(state.winner)
    if winner not in [0, 1]:
        return False, f"Invalid winner: {winner}"

    game_points = [int(state.game_points[0]), int(state.game_points[1])]
    if game_points[winner] == 0:
        return False, f"Winner P{winner} has 0 game points: {game_points}"

    if verbose:
        print(f"\n✓ Game completed successfully after {steps} steps")
        print(f"  Winner: P{winner}")
        print(f"  Game points: {game_points}")
        print(f"  Final points: [{int(state.points[0])}, {int(state.points[1])}]")
        print(f"  Tricks won: [{int(state.tricks_won[0])}, {int(state.tricks_won[1])}]")

    return True, f"Seed {seed} passed ({steps} steps)"


def run_optimized_parity_tests(num_games: int = 10, run_statistical: bool = True,
                                 num_statistical_games: int = 100, verbose: bool = False):
    """Run comprehensive parity tests for the optimized JAX implementation."""
    print("\n" + "="*60)
    print("OPTIMIZED JAX PARITY TEST SUITE")
    print("="*60)
    print(f"Testing optimized JAX implementation")
    print("="*60)

    # Test individual games
    print(f"\n[1/2] Testing {num_games} complete games...")

    passed = 0
    failed = 0

    np_rng = np.random.RandomState(42)

    for i in range(num_games):
        seed = np_rng.randint(0, 1000000)
        success, message = test_single_optimized_game(seed, verbose=verbose)

        if success:
            passed += 1
            print(f"  [{i+1}/{num_games}] ✓ {message}")
        else:
            failed += 1
            print(f"  [{i+1}/{num_games}] ✗ FAILED")
            print(f"       {message}")
            if not verbose:
                # Run again with verbose to see details
                print("\n       Re-running with verbose output...")
                test_single_optimized_game(seed, verbose=True)
            break  # Stop on first failure

    if failed > 0:
        print("\n" + "="*60)
        print("✗ GAME VALIDATION TESTS FAILED")
        print("="*60)
        return False

    print(f"\n  All {num_games} games passed validation!")

    # Run statistical parity test
    if run_statistical:
        print(f"\n[2/2] Testing statistical parity with {num_statistical_games} games...")
        success, message = run_statistical_parity_test(num_statistical_games, verbose=verbose)

        if success:
            print(f"  ✓ Statistical parity test passed")
        else:
            print(f"  ✗ Statistical parity test failed")
            print(f"     {message}")
            return False

    # Summary
    print("\n" + "="*60)
    print("✓ ALL OPTIMIZED PARITY TESTS PASSED!")
    print("="*60)
    print("\nThe optimized JAX implementation is functionally correct.")
    return True


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    num_games = 10
    num_statistical_games = 100
    run_statistical = "--no-stats" not in sys.argv

    for arg in sys.argv[1:]:
        if arg.startswith("--games="):
            num_games = int(arg.split("=")[1])
        elif arg.startswith("--stat-games="):
            num_statistical_games = int(arg.split("=")[1])
        elif arg.startswith("-n"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                num_games = int(sys.argv[idx + 1])

    # Run tests
    success = run_optimized_parity_tests(
        num_games=num_games,
        run_statistical=run_statistical,
        num_statistical_games=num_statistical_games,
        verbose=verbose
    )

    sys.exit(0 if success else 1)
