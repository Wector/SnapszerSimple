"""
Parity tests to ensure JAX implementation matches base implementation exactly.

These tests verify that:
1. Game initialization produces identical states
2. Step function produces identical state transitions
3. Legal actions match exactly
4. Observations match exactly
5. Full games produce identical outcomes
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, Tuple

import snapszer_base as base
import snapszer_jax as jax_env


def convert_base_to_jax_cards(cards: List[int]) -> int:
    """Convert list of cards to bitmask."""
    mask = 0
    for card in cards:
        mask |= (1 << card)
    return mask


def convert_jax_to_base_cards(mask: int) -> List[int]:
    """Convert bitmask to list of cards."""
    cards = []
    for i in range(base.NUM_CARDS):
        if mask & (1 << i):
            cards.append(i)
    return cards


class TestCardUtilities:
    """Test basic card utility functions."""

    def test_card_id(self):
        """Test card ID conversion."""
        for suit in range(4):
            for rank in range(5):
                base_id = base.card_id(suit, rank)
                jax_id = jax_env.card_id(suit, rank)
                assert base_id == jax_id, f"card_id({suit}, {rank}): base={base_id}, jax={jax_id}"

    def test_cid_suit_rank(self):
        """Test card ID extraction."""
        for cid in range(base.NUM_CARDS):
            base_suit = base.cid_suit(cid)
            base_rank = base.cid_rank(cid)
            jax_suit = int(jax_env.cid_suit(jnp.int32(cid)))
            jax_rank = int(jax_env.cid_rank(jnp.int32(cid)))

            assert base_suit == jax_suit, f"cid_suit({cid}): base={base_suit}, jax={jax_suit}"
            assert base_rank == jax_rank, f"cid_rank({cid}): base={base_rank}, jax={jax_rank}"

    def test_card_points(self):
        """Test card point values."""
        for cid in range(base.NUM_CARDS):
            base_pts = base.card_points(cid)
            jax_pts = int(jax_env.get_card_points(jnp.int32(cid)))
            assert base_pts == jax_pts, f"card_points({cid}): base={base_pts}, jax={jax_pts}"

    def test_mask_operations(self):
        """Test bitmask operations."""
        # Test conversion
        cards = [0, 5, 10, 15, 19]
        base_mask = base.cards_to_mask(cards)
        jax_mask = convert_base_to_jax_cards(cards)
        assert base_mask == jax_mask

        # Test contains
        for cid in range(base.NUM_CARDS):
            base_contains = base.mask_contains(base_mask, cid)
            jax_contains = bool(jax_env.mask_contains(jnp.int32(base_mask), jnp.int32(cid)))
            assert base_contains == jax_contains, f"mask_contains({cid}): base={base_contains}, jax={jax_contains}"


class TestGameLogic:
    """Test core game logic functions."""

    def test_trick_winner(self):
        """Test trick winner determination."""
        test_cases = [
            # (lead_cid, reply_cid, trump, post_close, expected_winner)
            (0, 1, 0, True, 0),    # Same suit, A beats 10
            (1, 0, 0, True, 1),    # Same suit, A beats 10 (reversed)
            (0, 5, 0, True, 0),    # Lead is trump, reply not - leader wins
            (5, 0, 1, True, 0),    # Lead is trump (H), reply not (S) - leader wins
            (0, 5, 1, True, 1),    # Lead not trump (S), reply is trump (H) - follower wins
            (10, 15, 2, False, 0), # Different suits, leader wins
        ]

        for lead_cid, reply_cid, trump, post_close, expected in test_cases:
            base_winner = base.trick_winner(lead_cid, reply_cid, trump, post_close)
            jax_winner = int(jax_env.trick_winner(
                jnp.int32(lead_cid),
                jnp.int32(reply_cid),
                jnp.int32(trump),
                jnp.bool_(post_close)
            ))

            assert base_winner == expected, \
                f"Base trick_winner({lead_cid}, {reply_cid}, trump={trump}): expected {expected}, got {base_winner}"
            assert jax_winner == expected, \
                f"JAX trick_winner({lead_cid}, {reply_cid}, trump={trump}): expected {expected}, got {jax_winner}"
            assert base_winner == jax_winner, \
                f"Mismatch for trick_winner({lead_cid}, {reply_cid}, trump={trump}): base={base_winner}, jax={jax_winner}"

    def test_legal_reply_cards_strict(self):
        """Test legal reply cards with strict rules (always strict in this function)."""
        # Test case: must follow suit
        hand = [0, 1, 5, 10, 15]  # Has cards from suit 0, 1, 2, 3
        hand_mask = base.cards_to_mask(hand)
        lead_cid = 6  # Suit 1, rank 1 (10 of Hearts)
        trump = 0

        # Both always use strict rules internally
        base_legal = base.legal_reply_cards(hand, lead_cid, trump)
        jax_legal_mask = int(jax_env.legal_reply_cards_mask(
            jnp.int32(hand_mask),
            jnp.int32(lead_cid),
            jnp.int32(trump)
        ))
        jax_legal = convert_jax_to_base_cards(jax_legal_mask)

        assert sorted(base_legal) == sorted(jax_legal), \
            f"Legal reply mismatch: base={sorted(base_legal)}, jax={sorted(jax_legal)}"

    def test_legal_reply_must_follow_suit(self):
        """Test that must follow suit when have same suit cards."""
        hand = [0, 1, 5, 6, 10]  # 0,1 are suit 0; 5,6 are suit 1; 10 is suit 2
        hand_mask = base.cards_to_mask(hand)
        lead_cid = 7  # Suit 1, rank 2 (K of Hearts)
        trump = 0

        base_legal = base.legal_reply_cards(hand, lead_cid, trump)
        jax_legal_mask = int(jax_env.legal_reply_cards_mask(
            jnp.int32(hand_mask),
            jnp.int32(lead_cid),
            jnp.int32(trump)
        ))
        jax_legal = convert_jax_to_base_cards(jax_legal_mask)

        # Should only be able to play suit 1 cards (5, 6)
        # And must beat with 5 or 6 if possible (rank 0 or 1 beats rank 2)
        # 5 is A of Hearts (rank 0), 6 is 10 of Hearts (rank 1), both beat K (rank 2)
        assert sorted(base_legal) == sorted(jax_legal), \
            f"Legal reply mismatch: base={sorted(base_legal)}, jax={sorted(jax_legal)}"


class TestStateInitialization:
    """Test game state initialization."""

    def test_initial_state_deterministic(self):
        """Test that same seed produces identical initial states."""
        seed = 42

        # Initialize base game
        base_state = base.SnapszerState.new(seed)

        # Initialize JAX game
        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)

        # Compare trump
        assert base_state.trump == int(jax_state.trump), \
            f"Trump mismatch: base={base_state.trump}, jax={int(jax_state.trump)}"

        # Compare trump card
        assert base_state.trump_card == int(jax_state.trump_card), \
            f"Trump card mismatch: base={base_state.trump_card}, jax={int(jax_state.trump_card)}"

        # Compare deck
        base_deck = base_state.deck
        jax_deck = [int(x) for x in jax_state.deck]
        assert base_deck == jax_deck, \
            f"Deck mismatch: base={base_deck}, jax={jax_deck}"

        # Compare hands
        base_hand0 = sorted(base_state.hands[0])
        base_hand1 = sorted(base_state.hands[1])
        jax_hand0 = sorted(convert_jax_to_base_cards(int(jax_state.hand_masks[0])))
        jax_hand1 = sorted(convert_jax_to_base_cards(int(jax_state.hand_masks[1])))

        assert base_hand0 == jax_hand0, \
            f"Player 0 hand mismatch: base={base_hand0}, jax={jax_hand0}"
        assert base_hand1 == jax_hand1, \
            f"Player 1 hand mismatch: base={base_hand1}, jax={jax_hand1}"

        # Compare other state fields
        assert base_state.current_player == int(jax_state.current_player)
        assert base_state.leader == int(jax_state.leader)
        assert list(base_state.points) == [int(x) for x in jax_state.points]
        assert base_state.closed == bool(jax_state.closed)
        assert base_state.terminal == bool(jax_state.terminal)

    def test_multiple_seeds(self):
        """Test that different seeds produce different games."""
        seeds = [0, 42, 123, 999]
        base_decks = []
        jax_decks = []

        for seed in seeds:
            base_state = base.SnapszerState.new(seed)
            base_decks.append(base_state.deck)

            rng = random.PRNGKey(seed)
            jax_state = jax_env.init_game(rng)
            jax_decks.append([int(x) for x in jax_state.deck])

        # Different seeds should produce different decks
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                assert base_decks[i] != base_decks[j], \
                    f"Base decks identical for seeds {seeds[i]} and {seeds[j]}"
                assert jax_decks[i] != jax_decks[j], \
                    f"JAX decks identical for seeds {seeds[i]} and {seeds[j]}"

        # But base and JAX should match for same seed
        for i in range(len(seeds)):
            assert base_decks[i] == jax_decks[i], \
                f"Deck mismatch for seed {seeds[i]}"


class TestLegalActions:
    """Test legal action computation."""

    def test_initial_legal_actions(self):
        """Test legal actions at start of game."""
        seed = 42
        base_state = base.SnapszerState.new(seed)

        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)

        # Get legal actions
        base_legal = base_state.legal_actions()
        jax_legal_mask = int(jax_env.get_legal_actions(jax_state))
        jax_legal = [i for i in range(jax_env.TOTAL_ACTIONS) if jax_legal_mask & (1 << i)]

        assert sorted(base_legal) == sorted(jax_legal), \
            f"Initial legal actions mismatch: base={sorted(base_legal)}, jax={sorted(jax_legal)}"

    def test_legal_actions_after_lead(self):
        """Test legal actions after a card has been led."""
        seed = 42
        base_state = base.SnapszerState.new(seed)
        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)

        # Play first legal action
        base_legal = base_state.legal_actions()
        action = base_legal[0]

        base_state.apply_action(action)
        jax_state = jax_env.step(jax_state, jnp.int32(action))

        # Get legal actions for follower
        base_legal = base_state.legal_actions()
        jax_legal_mask = int(jax_env.get_legal_actions(jax_state))
        jax_legal = [i for i in range(jax_env.TOTAL_ACTIONS) if jax_legal_mask & (1 << i)]

        assert sorted(base_legal) == sorted(jax_legal), \
            f"Legal actions after lead mismatch: base={sorted(base_legal)}, jax={sorted(jax_legal)}"


class TestObservations:
    """Test observation tensor generation."""

    def test_initial_observation(self):
        """Test observation at start of game."""
        seed = 42
        base_state = base.SnapszerState.new(seed)
        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)

        for player in [0, 1]:
            base_obs = base_state.observation_tensor(player)
            jax_obs = np.array(jax_env.get_observation(jax_state, jnp.int32(player)))

            assert len(base_obs) == len(jax_obs) == jax_env.OBSERVATION_SIZE, \
                f"Observation size mismatch: base={len(base_obs)}, jax={len(jax_obs)}"

            assert np.allclose(base_obs, jax_obs, atol=1e-6), \
                f"Observation mismatch for player {player}:\nBase: {base_obs}\nJAX:  {jax_obs}"

    def test_observation_after_actions(self):
        """Test observation after several actions."""
        seed = 42
        base_state = base.SnapszerState.new(seed)
        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)

        # Play a few actions
        for _ in range(5):
            if base_state.terminal:
                break

            base_legal = base_state.legal_actions()
            action = base_legal[0]

            base_state.apply_action(action)
            jax_state = jax_env.step(jax_state, jnp.int32(action))

        # Compare observations
        for player in [0, 1]:
            base_obs = base_state.observation_tensor(player)
            jax_obs = np.array(jax_env.get_observation(jax_state, jnp.int32(player)))

            assert np.allclose(base_obs, jax_obs, atol=1e-6), \
                f"Observation mismatch after actions for player {player}"


class TestGamePlay:
    """Test full game playouts."""

    def test_single_game_deterministic(self):
        """Test that same actions produce identical game outcomes."""
        seed = 42
        base_state = base.SnapszerState.new(seed)
        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)

        step_count = 0
        max_steps = 100

        while not base_state.terminal and step_count < max_steps:
            # Verify states match before action
            assert base_state.current_player == int(jax_state.current_player), \
                f"Step {step_count}: Current player mismatch"
            assert base_state.terminal == bool(jax_state.terminal), \
                f"Step {step_count}: Terminal mismatch"

            # Get legal actions (should match)
            base_legal = base_state.legal_actions()
            jax_legal_mask = int(jax_env.get_legal_actions(jax_state))
            jax_legal = [i for i in range(jax_env.TOTAL_ACTIONS) if jax_legal_mask & (1 << i)]

            assert sorted(base_legal) == sorted(jax_legal), \
                f"Step {step_count}: Legal actions mismatch"

            # Take first legal action
            action = base_legal[0]
            base_state.apply_action(action)
            jax_state = jax_env.step(jax_state, jnp.int32(action))

            # Verify points match
            assert list(base_state.points) == [int(x) for x in jax_state.points], \
                f"Step {step_count}: Points mismatch after action {action}"

            step_count += 1

        # Verify game ended the same way
        assert base_state.terminal == bool(jax_state.terminal), \
            "Terminal state mismatch at end"

        if base_state.terminal:
            assert base_state.winner == int(jax_state.winner), \
                f"Winner mismatch: base={base_state.winner}, jax={int(jax_state.winner)}"
            assert list(base_state.game_points) == [int(x) for x in jax_state.game_points], \
                f"Game points mismatch: base={base_state.game_points}, jax={[int(x) for x in jax_state.game_points]}"

    def test_multiple_games(self):
        """Test multiple games with different seeds."""
        seeds = [0, 42, 123, 999, 2024]

        for seed in seeds:
            base_state = base.SnapszerState.new(seed)
            rng = random.PRNGKey(seed)
            jax_state = jax_env.init_game(rng)

            step_count = 0
            max_steps = 100

            while not base_state.terminal and step_count < max_steps:
                base_legal = base_state.legal_actions()
                jax_legal_mask = int(jax_env.get_legal_actions(jax_state))
                jax_legal = [i for i in range(jax_env.TOTAL_ACTIONS) if jax_legal_mask & (1 << i)]

                assert sorted(base_legal) == sorted(jax_legal), \
                    f"Seed {seed}, step {step_count}: Legal actions mismatch"

                action = base_legal[0]
                base_state.apply_action(action)
                jax_state = jax_env.step(jax_state, jnp.int32(action))

                step_count += 1

            # Verify end state
            assert base_state.terminal == bool(jax_state.terminal), \
                f"Seed {seed}: Terminal mismatch"

            if base_state.terminal:
                assert base_state.winner == int(jax_state.winner), \
                    f"Seed {seed}: Winner mismatch"
                assert list(base_state.game_points) == [int(x) for x in jax_state.game_points], \
                    f"Seed {seed}: Game points mismatch"

    def test_random_playout(self):
        """Test random game playout matches."""
        seed = 42

        # Use deterministic random playouts
        np.random.seed(seed)
        base_state = base.SnapszerState.new(seed)

        rng = random.PRNGKey(seed)
        jax_state = jax_env.init_game(rng)
        action_rng = random.PRNGKey(seed + 1)

        step_count = 0
        max_steps = 100

        while not base_state.terminal and step_count < max_steps:
            # Get legal actions
            base_legal = base_state.legal_actions()
            jax_legal_mask = int(jax_env.get_legal_actions(jax_state))
            jax_legal = [i for i in range(jax_env.TOTAL_ACTIONS) if jax_legal_mask & (1 << i)]

            assert sorted(base_legal) == sorted(jax_legal), \
                f"Step {step_count}: Legal actions mismatch"

            # Choose random action (same for both)
            action_idx = np.random.randint(len(base_legal))
            action = base_legal[action_idx]

            base_state.apply_action(action)
            jax_state = jax_env.step(jax_state, jnp.int32(action))

            step_count += 1

        # Verify same outcome
        assert base_state.terminal == bool(jax_state.terminal)
        if base_state.terminal:
            assert base_state.winner == int(jax_state.winner)


class TestSpecialActions:
    """Test special actions (exchange, close)."""

    def test_exchange_trump_jack(self):
        """Test exchange trump jack action."""
        # We need to set up a state where exchange is legal
        # This is tricky with random shuffling, so we'll test the logic
        # by checking if exchange legality matches

        seed = 42
        for test_seed in range(10):
            base_state = base.SnapszerState.new(seed + test_seed)
            rng = random.PRNGKey(seed + test_seed)
            jax_state = jax_env.init_game(rng)

            # Check if exchange is legal
            base_legal = base_state.legal_actions()
            jax_legal_mask = int(jax_env.get_legal_actions(jax_state))

            base_can_exchange = base.EXCHANGE_TRUMP_ACTION in base_legal
            jax_can_exchange = bool(jax_legal_mask & (1 << jax_env.EXCHANGE_TRUMP_ACTION))

            assert base_can_exchange == jax_can_exchange, \
                f"Seed {seed + test_seed}: Exchange legality mismatch"

            if base_can_exchange:
                # Apply exchange
                base_state.apply_action(base.EXCHANGE_TRUMP_ACTION)
                jax_state = jax_env.step(jax_state, jnp.int32(jax_env.EXCHANGE_TRUMP_ACTION))

                # Verify trump card changed
                assert base_state.trump_card == int(jax_state.trump_card), \
                    f"Seed {seed + test_seed}: Trump card mismatch after exchange"

    def test_close_talon(self):
        """Test close talon action."""
        seed = 42
        for test_seed in range(10):
            base_state = base.SnapszerState.new(seed + test_seed)
            rng = random.PRNGKey(seed + test_seed)
            jax_state = jax_env.init_game(rng)

            # Check if close is legal
            base_legal = base_state.legal_actions()
            jax_legal_mask = int(jax_env.get_legal_actions(jax_state))

            base_can_close = base.CLOSE_TALON_ACTION in base_legal
            jax_can_close = bool(jax_legal_mask & (1 << jax_env.CLOSE_TALON_ACTION))

            assert base_can_close == jax_can_close, \
                f"Seed {seed + test_seed}: Close legality mismatch"

            if base_can_close:
                # Apply close
                base_state.apply_action(base.CLOSE_TALON_ACTION)
                jax_state = jax_env.step(jax_state, jnp.int32(jax_env.CLOSE_TALON_ACTION))

                # Verify closed flag
                assert base_state.closed == bool(jax_state.closed), \
                    f"Seed {seed + test_seed}: Closed flag mismatch"
                assert base_state.closed_by == int(jax_state.closed_by), \
                    f"Seed {seed + test_seed}: Closed by mismatch"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
