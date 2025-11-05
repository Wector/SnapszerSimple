"""Parity tests between base Python and JAX implementations of Snapszer.

This test suite ensures that the JAX implementation produces identical results
to the base Python implementation for the same seeds and action sequences.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from snapszer import base
from snapszer import jax_impl


def compare_hands(base_hand: List[int], jax_hand: jnp.ndarray, jax_size: int, player: int) -> Tuple[bool, str]:
    """Compare hands between base and JAX implementations."""
    jax_hand_list = [int(x) for x in jax_hand[:jax_size] if x >= 0]

    if base_hand != jax_hand_list:
        return False, f"Player {player} hands differ: base={base_hand}, jax={jax_hand_list}"

    return True, ""


def compare_states(base_state: base.SnapszerState, jax_state: jax_impl.SnapszerState, step: int) -> Tuple[bool, str]:
    """Compare all relevant state components between base and JAX."""
    errors = []

    # Trump
    if base_state.trump != int(jax_state.trump):
        errors.append(f"Trump differs: base={base_state.trump}, jax={int(jax_state.trump)}")

    # Trump card
    if base_state.trump_card != int(jax_state.trump_card):
        errors.append(f"Trump card differs: base={base_state.trump_card}, jax={int(jax_state.trump_card)}")

    # Deck
    base_deck = base_state.deck
    jax_deck = [int(x) for x in jax_state.deck]
    if base_deck != jax_deck:
        errors.append(f"Decks differ: base={base_deck}, jax={jax_deck}")

    # Stock index
    if base_state.stock_idx != int(jax_state.stock_idx):
        errors.append(f"Stock index differs: base={base_state.stock_idx}, jax={int(jax_state.stock_idx)}")

    # Hands
    for p in range(2):
        match, msg = compare_hands(base_state.hands[p], jax_state.hands[p], int(jax_state.hand_sizes[p]), p)
        if not match:
            errors.append(msg)

    # Hand sizes
    base_sizes = [len(base_state.hands[0]), len(base_state.hands[1])]
    jax_sizes = [int(x) for x in jax_state.hand_sizes]
    if base_sizes != jax_sizes:
        errors.append(f"Hand sizes differ: base={base_sizes}, jax={jax_sizes}")

    # Hand masks
    base_masks = [base_state.hand_masks[0], base_state.hand_masks[1]]
    jax_masks = [int(x) for x in jax_state.hand_masks]
    if base_masks != jax_masks:
        errors.append(f"Hand masks differ: base={base_masks}, jax={jax_masks}")

    # Current player
    if base_state.current_player != int(jax_state.current_player):
        errors.append(f"Current player differs: base={base_state.current_player}, jax={int(jax_state.current_player)}")

    # Leader
    if base_state.leader != int(jax_state.leader):
        errors.append(f"Leader differs: base={base_state.leader}, jax={int(jax_state.leader)}")

    # Trick cards
    base_trick = [x if x is not None else -1 for x in base_state.trick_cards]
    jax_trick = [int(x) for x in jax_state.trick_cards]
    if base_trick != jax_trick:
        errors.append(f"Trick cards differ: base={base_trick}, jax={jax_trick}")

    # Points
    base_points = list(base_state.points)
    jax_points = [int(x) for x in jax_state.points]
    if base_points != jax_points:
        errors.append(f"Points differ: base={base_points}, jax={jax_points}")

    # Tricks won
    base_tricks = list(base_state.tricks_won)
    jax_tricks = [int(x) for x in jax_state.tricks_won]
    if base_tricks != jax_tricks:
        errors.append(f"Tricks won differ: base={base_tricks}, jax={jax_tricks}")

    # Closed
    if base_state.closed != bool(jax_state.closed):
        errors.append(f"Closed differs: base={base_state.closed}, jax={bool(jax_state.closed)}")

    # Closed by
    base_closed_by = base_state.closed_by if base_state.closed_by is not None else -1
    jax_closed_by = int(jax_state.closed_by)
    if base_closed_by != jax_closed_by:
        errors.append(f"Closed by differs: base={base_closed_by}, jax={jax_closed_by}")

    # Trump taken
    if base_state.trump_taken != bool(jax_state.trump_taken):
        errors.append(f"Trump taken differs: base={base_state.trump_taken}, jax={bool(jax_state.trump_taken)}")

    # Last trick winner
    base_ltw = base_state.last_trick_winner if base_state.last_trick_winner is not None else -1
    jax_ltw = int(jax_state.last_trick_winner)
    if base_ltw != jax_ltw:
        errors.append(f"Last trick winner differs: base={base_ltw}, jax={jax_ltw}")

    # Marriages scored
    for p in range(2):
        for s in range(4):
            base_marriage = base_state.marriages_scored[p][s]
            jax_marriage = bool(jax_state.marriages_scored[p, s])
            if base_marriage != jax_marriage:
                errors.append(f"Marriage P{p} S{s} differs: base={base_marriage}, jax={jax_marriage}")

    # Terminal
    if base_state.terminal != bool(jax_state.terminal):
        errors.append(f"Terminal differs: base={base_state.terminal}, jax={bool(jax_state.terminal)}")

    # Winner
    base_winner = base_state.winner if base_state.winner is not None else -1
    jax_winner = int(jax_state.winner)
    if base_winner != jax_winner:
        errors.append(f"Winner differs: base={base_winner}, jax={jax_winner}")

    # Game points
    if base_state.game_points is not None:
        base_gp = list(base_state.game_points)
    else:
        base_gp = [0, 0]
    jax_gp = [int(x) for x in jax_state.game_points]
    if base_gp != jax_gp:
        errors.append(f"Game points differ: base={base_gp}, jax={jax_gp}")

    if errors:
        error_msg = f"\nStep {step} state comparison failed:\n" + "\n".join(f"  - {e}" for e in errors)
        return False, error_msg

    return True, ""


def compare_legal_actions(base_actions: List[int], jax_mask: jnp.ndarray, step: int) -> Tuple[bool, str]:
    """Compare legal actions between base and JAX."""
    jax_actions = [i for i in range(len(jax_mask)) if jax_mask[i]]

    base_set = set(base_actions)
    jax_set = set(jax_actions)

    if base_set != jax_set:
        missing_in_jax = base_set - jax_set
        extra_in_jax = jax_set - base_set
        error_msg = f"\nStep {step} legal actions differ:\n"
        error_msg += f"  Base: {sorted(base_actions)}\n"
        error_msg += f"  JAX:  {sorted(jax_actions)}\n"
        if missing_in_jax:
            error_msg += f"  Missing in JAX: {sorted(missing_in_jax)}\n"
        if extra_in_jax:
            error_msg += f"  Extra in JAX: {sorted(extra_in_jax)}\n"
        return False, error_msg

    return True, ""


def compare_observations(base_obs: List[float], jax_obs: jnp.ndarray, player: int, step: int) -> Tuple[bool, str]:
    """Compare observation tensors."""
    jax_obs_list = [float(x) for x in jax_obs]

    # Allow small floating point differences
    for i in range(len(base_obs)):
        if abs(base_obs[i] - jax_obs_list[i]) > 1e-6:
            return False, f"\nStep {step} observation P{player}[{i}] differs: base={base_obs[i]}, jax={jax_obs_list[i]}"

    return True, ""


def test_single_game(seed: int, max_steps: int = 100, verbose: bool = False) -> Tuple[bool, str]:
    """Test a single game with deterministic seed."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing game with seed {seed}")
        print(f"{'='*60}")

    # Initialize both implementations
    base_state = base.SnapszerState.new(seed=seed)
    jax_state = jax_impl.new_game(jnp.int32(seed))

    # Compare initial states
    match, error = compare_states(base_state, jax_state, 0)
    if not match:
        return False, f"Initial state mismatch for seed {seed}: {error}"

    # Compare initial legal actions
    base_actions = base_state.legal_actions()
    jax_mask = jax_impl.legal_actions_mask(jax_state)
    match, error = compare_legal_actions(base_actions, jax_mask, 0)
    if not match:
        return False, f"Initial legal actions mismatch for seed {seed}: {error}"

    # Compare initial observations
    for p in range(2):
        base_obs = base_state.observation_tensor(p)
        jax_obs = jax_impl.observation_tensor(jax_state, jnp.int32(p))
        match, error = compare_observations(base_obs, jax_obs, p, 0)
        if not match:
            return False, f"Initial observation mismatch for seed {seed}: {error}"

    # Play through the game
    step = 0
    np_rng = np.random.RandomState(seed + 1000)  # Different seed for action selection

    while not base_state.is_terminal() and step < max_steps:
        step += 1

        # Get legal actions
        base_actions = base_state.legal_actions()
        jax_mask = jax_impl.legal_actions_mask(jax_state)

        # Compare legal actions
        match, error = compare_legal_actions(base_actions, jax_mask, step)
        if not match:
            return False, f"Seed {seed} step {step}: {error}"

        # Choose a random legal action
        action = np_rng.choice(base_actions)

        if verbose:
            print(f"\nStep {step}:")
            print(f"  Player: {base_state.current_player}")
            print(f"  Action: {action}")
            print(f"  Base: {base_state.public_str()}")

        # Apply action to both
        base_state.apply_action(action)
        jax_state = jax_impl.apply_action(jax_state, jnp.int32(action))

        # Compare states after action
        match, error = compare_states(base_state, jax_state, step)
        if not match:
            return False, f"Seed {seed} step {step} after action {action}: {error}"

        # Compare observations
        for p in range(2):
            base_obs = base_state.observation_tensor(p)
            jax_obs = jax_impl.observation_tensor(jax_state, jnp.int32(p))
            match, error = compare_observations(base_obs, jax_obs, p, step)
            if not match:
                return False, f"Seed {seed} step {step} after action {action}: {error}"

    # Verify game ended
    if base_state.is_terminal() != bool(jax_state.terminal):
        return False, f"Seed {seed}: Terminal state mismatch at step {step}"

    # Compare final returns
    base_returns = base_state.returns()
    jax_returns = jax_impl.returns(jax_state)
    if abs(base_returns[0] - jax_returns[0]) > 1e-6 or abs(base_returns[1] - jax_returns[1]) > 1e-6:
        return False, f"Seed {seed}: Returns differ: base={base_returns}, jax={jax_returns}"

    if verbose:
        print(f"\n✓ Game completed successfully after {step} steps")
        print(f"  Winner: P{base_state.winner}")
        print(f"  Game points: {base_state.game_points}")
        print(f"  Returns: {base_returns}")

    return True, f"Seed {seed} passed ({step} steps)"


def test_shuffle_consistency():
    """Test that the MT19937 shuffle implementation is consistent."""
    print("\n" + "="*60)
    print("Testing MT19937 Shuffle Consistency")
    print("="*60)

    for seed in [0, 42, 12345, 999999]:
        # Base shuffle
        base_deck = list(range(20))
        base._mt_shuffle(base_deck, seed)

        # JAX shuffle
        jax_deck_arr = jnp.arange(20, dtype=jnp.int32)
        jax_deck_shuffled = jax_impl.mt_shuffle(jax_deck_arr, jnp.int32(seed))
        jax_deck = [int(x) for x in jax_deck_shuffled]

        if base_deck != jax_deck:
            print(f"✗ Seed {seed}: Shuffle mismatch")
            print(f"  Base: {base_deck}")
            print(f"  JAX:  {jax_deck}")
            return False
        else:
            print(f"✓ Seed {seed}: Shuffles match")

    return True


def run_parity_tests(num_games: int = 10, seeds: List[int] = None, verbose: bool = False):
    """Run comprehensive parity tests."""
    print("\n" + "="*60)
    print("SNAPSZER PARITY TEST SUITE")
    print("="*60)
    print(f"Comparing Base Python vs JAX implementations")
    print("="*60)

    # Test shuffle first
    print("\n[1/3] Testing RNG consistency...")
    if not test_shuffle_consistency():
        print("\n✗ SHUFFLE TESTS FAILED")
        return False

    # Test multiple games
    print(f"\n[2/3] Testing {num_games} complete games...")

    if seeds is None:
        seeds = list(range(num_games))

    passed = 0
    failed = 0

    for i, seed in enumerate(seeds, 1):
        success, message = test_single_game(seed, verbose=verbose)

        if success:
            passed += 1
            print(f"  [{i}/{num_games}] ✓ {message}")
        else:
            failed += 1
            print(f"  [{i}/{num_games}] ✗ FAILED")
            print(f"       {message}")
            if not verbose:
                # Run again with verbose to see details
                print("\n       Re-running with verbose output...")
                test_single_game(seed, verbose=True)
            break  # Stop on first failure

    # Summary
    print(f"\n[3/3] Summary:")
    print(f"  Passed: {passed}/{num_games}")
    print(f"  Failed: {failed}/{num_games}")

    if failed == 0:
        print("\n" + "="*60)
        print("✓ ALL PARITY TESTS PASSED!")
        print("="*60)
        print("\nThe JAX implementation is functionally equivalent to the base.")
        return True
    else:
        print("\n" + "="*60)
        print("✗ PARITY TESTS FAILED")
        print("="*60)
        return False


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    num_games = 10

    for arg in sys.argv[1:]:
        if arg.startswith("--games="):
            num_games = int(arg.split("=")[1])
        elif arg.startswith("-n"):
            num_games = int(sys.argv[sys.argv.index(arg) + 1])

    # Run tests
    success = run_parity_tests(num_games=num_games, verbose=verbose)

    sys.exit(0 if success else 1)
