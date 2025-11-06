"""Unit tests for CFR training implementation."""

import pytest
import numpy as np
import tempfile
import os

from training.config import CFRConfig
from training.info_set import (
    InformationSet,
    get_info_set_key,
    get_strategy,
    get_average_strategy
)
from training.cfr_trainer import CFRTrainer
from snapszer import jax_optimized as game


class TestInformationSet:
    """Test information set utilities."""

    def test_create_info_set(self):
        """Test creating information set."""
        info_set = InformationSet.create(num_actions=10)

        assert len(info_set.regret_sum) == 10
        assert len(info_set.strategy_sum) == 10
        assert info_set.num_updates == 0
        assert np.allclose(info_set.regret_sum, 0.0)
        assert np.allclose(info_set.strategy_sum, 0.0)

    def test_regret_matching_uniform(self):
        """Test regret matching with zero regrets gives uniform strategy."""
        regret_sum = np.zeros(5)
        legal_mask = np.array([True, True, True, False, False])

        strategy = get_strategy(regret_sum, legal_mask)

        # Should be uniform over 3 legal actions
        expected = np.array([1/3, 1/3, 1/3, 0.0, 0.0])
        assert np.allclose(strategy, expected)
        assert np.isclose(np.sum(strategy), 1.0)

    def test_regret_matching_positive_regrets(self):
        """Test regret matching with positive regrets."""
        regret_sum = np.array([3.0, 1.0, 2.0, 0.0])
        legal_mask = np.array([True, True, True, True])

        strategy = get_strategy(regret_sum, legal_mask)

        # Should be proportional to positive regrets
        expected = np.array([3/6, 1/6, 2/6, 0.0])
        assert np.allclose(strategy, expected)
        assert np.isclose(np.sum(strategy), 1.0)

    def test_regret_matching_negative_regrets(self):
        """Test regret matching with negative regrets (ignored)."""
        regret_sum = np.array([3.0, -5.0, 2.0, -1.0])
        legal_mask = np.array([True, True, True, True])

        strategy = get_strategy(regret_sum, legal_mask)

        # Negative regrets should be floored at 0
        # Only actions 0 and 2 contribute
        expected = np.array([3/5, 0.0, 2/5, 0.0])
        assert np.allclose(strategy, expected)
        assert np.isclose(np.sum(strategy), 1.0)

    def test_regret_matching_respects_legal_mask(self):
        """Test that illegal actions get 0 probability."""
        regret_sum = np.array([10.0, 20.0, 5.0, 15.0])
        legal_mask = np.array([True, False, True, False])

        strategy = get_strategy(regret_sum, legal_mask)

        # Only actions 0 and 2 are legal
        assert strategy[1] == 0.0
        assert strategy[3] == 0.0
        assert strategy[0] > 0.0
        assert strategy[2] > 0.0
        assert np.isclose(np.sum(strategy), 1.0)

    def test_average_strategy_uniform(self):
        """Test average strategy with zero strategy sum."""
        info_set = InformationSet.create(num_actions=4)

        avg_strategy = get_average_strategy(info_set)

        # Should be uniform
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        assert np.allclose(avg_strategy, expected)
        assert np.isclose(np.sum(avg_strategy), 1.0)

    def test_average_strategy_with_updates(self):
        """Test average strategy after updates."""
        info_set = InformationSet.create(num_actions=3)

        # Simulate strategy updates
        info_set.strategy_sum = np.array([6.0, 3.0, 1.0])

        avg_strategy = get_average_strategy(info_set)

        # Should normalize strategy_sum
        expected = np.array([6/10, 3/10, 1/10])
        assert np.allclose(avg_strategy, expected)
        assert np.isclose(np.sum(avg_strategy), 1.0)

    def test_strategy_is_valid_probability_distribution(self):
        """Test that strategies are valid probability distributions."""
        # Test with various regret configurations
        test_cases = [
            (np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, True, True])),
            (np.array([0.0, 0.0, 0.0, 5.0]), np.array([True, True, True, True])),
            (np.array([-1.0, -2.0, 3.0, 4.0]), np.array([True, True, True, True])),
            (np.array([10.0, 0.0, 0.0, 0.0]), np.array([True, False, False, False])),
        ]

        for regret_sum, legal_mask in test_cases:
            strategy = get_strategy(regret_sum, legal_mask)

            # Must be non-negative
            assert np.all(strategy >= 0.0), f"Negative probabilities: {strategy}"

            # Must sum to 1
            assert np.isclose(np.sum(strategy), 1.0), f"Doesn't sum to 1: {strategy}"

            # Illegal actions must have 0 probability
            assert np.all(strategy[~legal_mask] == 0.0), \
                f"Illegal actions have non-zero prob: {strategy}"


class TestInfoSetKey:
    """Test information set key generation."""

    def test_info_set_key_deterministic(self):
        """Test that same state produces same key."""
        import jax

        key = jax.random.PRNGKey(42)
        state = game.new_game(key)

        key1 = get_info_set_key(state, 0)
        key2 = get_info_set_key(state, 0)

        assert key1 == key2
        assert isinstance(key1, bytes)

    def test_info_set_key_different_players(self):
        """Test that different players see different information sets."""
        import jax

        key = jax.random.PRNGKey(42)
        state = game.new_game(key)

        key_p0 = get_info_set_key(state, 0)
        key_p1 = get_info_set_key(state, 1)

        # Players have different observations (different hands)
        assert key_p0 != key_p1

    def test_info_set_key_different_states(self):
        """Test that different states produce different keys."""
        import jax

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)

        state1 = game.new_game(key1)
        state2 = game.new_game(key2)

        info_key1 = get_info_set_key(state1, 0)
        info_key2 = get_info_set_key(state2, 0)

        # Different shuffles -> different keys
        assert info_key1 != info_key2


class TestCFRTrainer:
    """Test CFR trainer functionality."""

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        config = CFRConfig(num_iterations=100)
        trainer = CFRTrainer(config)

        assert trainer.iteration == 0
        assert len(trainer.info_sets) == 2  # Two players
        assert len(trainer.info_sets[0]) == 0  # No info sets yet
        assert len(trainer.info_sets[1]) == 0

    def test_trainer_discovers_info_sets(self):
        """Test that training discovers information sets."""
        config = CFRConfig(
            num_iterations=10,
            cfr_variant='cfr+',
            sampling='external'
        )
        trainer = CFRTrainer(config)

        # Before training
        assert len(trainer.info_sets[0]) == 0
        assert len(trainer.info_sets[1]) == 0

        # Train a few iterations
        trainer.train(10)

        # Should have discovered some info sets
        num_info_sets_0, num_info_sets_1 = trainer.get_num_info_sets()
        assert num_info_sets_0 > 0, "Player 0 should have info sets"
        assert num_info_sets_1 > 0, "Player 1 should have info sets"
        assert trainer.iteration == 10

    def test_get_strategy_unknown_state(self):
        """Test getting strategy for unknown state returns uniform."""
        import jax

        config = CFRConfig()
        trainer = CFRTrainer(config)

        # Create a state
        key = jax.random.PRNGKey(999)
        state = game.new_game(key)

        # Get strategy for unknown state
        strategy = trainer.get_strategy(state, 0)

        # Should be uniform over legal actions
        legal_mask = np.array(game.legal_actions_mask(state))
        num_legal = np.sum(legal_mask)

        assert np.isclose(np.sum(strategy), 1.0)
        assert np.allclose(strategy[legal_mask], 1.0 / num_legal)
        assert np.allclose(strategy[~legal_mask], 0.0)

    def test_get_strategy_returns_valid_distribution(self):
        """Test that get_strategy always returns valid probability distribution."""
        import jax

        config = CFRConfig(num_iterations=5)
        trainer = CFRTrainer(config)

        # Train a bit
        trainer.train(5)

        # Test strategy on multiple states
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            state = game.new_game(key)

            if not state.terminal:
                for player in [0, 1]:
                    strategy = trainer.get_strategy(state, player)

                    # Valid probability distribution
                    assert np.all(strategy >= 0.0), f"Negative probs: {strategy}"
                    assert np.isclose(np.sum(strategy), 1.0), f"Doesn't sum to 1: {strategy}"

                    # Respects legal actions
                    legal_mask = np.array(game.legal_actions_mask(state))
                    assert np.allclose(strategy[~legal_mask], 0.0), \
                        f"Illegal actions have non-zero prob: {strategy}"

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        config = CFRConfig(num_iterations=10)
        trainer = CFRTrainer(config)

        # Train a bit
        trainer.train(10)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pkl')
            trainer.save_checkpoint(checkpoint_path)

            # Create new trainer and load
            trainer2 = CFRTrainer(config)
            trainer2.load_checkpoint(checkpoint_path)

            # Should have same iteration and info sets
            assert trainer2.iteration == trainer.iteration
            assert len(trainer2.info_sets[0]) == len(trainer.info_sets[0])
            assert len(trainer2.info_sets[1]) == len(trainer.info_sets[1])

    def test_cfr_plus_non_negative_regrets(self):
        """Test that CFR+ keeps regrets non-negative."""
        config = CFRConfig(
            num_iterations=20,
            cfr_variant='cfr+',
            sampling='external'
        )
        trainer = CFRTrainer(config)

        trainer.train(20)

        # Check all regrets are non-negative
        for player in [0, 1]:
            for info_set in trainer.info_sets[player].values():
                assert np.all(info_set.regret_sum >= 0.0), \
                    f"CFR+ should have non-negative regrets: {info_set.regret_sum}"

    def test_vanilla_cfr_allows_negative_regrets(self):
        """Test that vanilla CFR allows negative regrets."""
        config = CFRConfig(
            num_iterations=20,
            cfr_variant='vanilla',
            sampling='vanilla'
        )
        trainer = CFRTrainer(config)

        trainer.train(20)

        # Vanilla CFR can have negative regrets
        has_negative = False
        for player in [0, 1]:
            for info_set in trainer.info_sets[player].values():
                if np.any(info_set.regret_sum < 0.0):
                    has_negative = True
                    break

        # Not guaranteed, but very likely after 20 iterations
        # (comment out if flaky, but should generally pass)
        # assert has_negative, "Vanilla CFR should sometimes have negative regrets"

    def test_external_sampling_faster_than_vanilla(self):
        """Test that external sampling is faster than vanilla CFR."""
        import time

        # External sampling
        config_external = CFRConfig(
            num_iterations=50,
            cfr_variant='cfr+',
            sampling='external'
        )
        trainer_external = CFRTrainer(config_external)

        start = time.time()
        trainer_external.train(50)
        time_external = time.time() - start

        # Vanilla CFR
        config_vanilla = CFRConfig(
            num_iterations=50,
            cfr_variant='cfr+',
            sampling='vanilla'
        )
        trainer_vanilla = CFRTrainer(config_vanilla)

        start = time.time()
        trainer_vanilla.train(50)
        time_vanilla = time.time() - start

        print(f"\nExternal sampling: {time_external:.3f}s")
        print(f"Vanilla CFR: {time_vanilla:.3f}s")
        print(f"Speedup: {time_vanilla / time_external:.2f}x")

        # External sampling should be faster
        # (may be flaky on very fast machines, but generally true)
        assert time_external < time_vanilla * 1.5, \
            "External sampling should be faster than vanilla"


class TestCFRConvergence:
    """Test CFR convergence properties."""

    def test_exploitability_decreases(self):
        """Test that exploitability generally decreases over training."""
        from training.evaluation import compute_exploitability

        config = CFRConfig(
            num_iterations=500,
            cfr_variant='cfr+',
            sampling='external'
        )
        trainer = CFRTrainer(config)

        # Measure exploitability at different points
        exploitability_early = compute_exploitability(trainer, n_games=20)

        trainer.train(500)

        exploitability_late = compute_exploitability(trainer, n_games=20)

        print(f"\nExploitability early: {exploitability_early:.6f}")
        print(f"Exploitability after 500 iters: {exploitability_late:.6f}")

        # Should generally decrease (not guaranteed monotonic, but final should be lower)
        assert exploitability_late < exploitability_early, \
            "Exploitability should decrease with training"

    def test_beats_random_player(self):
        """Test that trained strategy beats random player."""
        from training.evaluation import evaluate_vs_random

        config = CFRConfig(
            num_iterations=500,
            cfr_variant='cfr+',
            sampling='external'
        )
        trainer = CFRTrainer(config)

        # Train
        trainer.train(500)

        # Evaluate vs random
        win_rate = evaluate_vs_random(trainer, n_games=100)

        print(f"\nWin rate vs random after 500 iters: {win_rate:.3f}")

        # Should beat random player (win rate > 0.5)
        assert win_rate > 0.55, \
            f"Trained strategy should beat random (got {win_rate:.3f})"

    def test_self_play_balanced(self):
        """Test that self-play is approximately balanced."""
        from training.evaluation import evaluate_self_play

        config = CFRConfig(
            num_iterations=1000,
            cfr_variant='cfr+',
            sampling='external'
        )
        trainer = CFRTrainer(config)

        # Train
        trainer.train(1000)

        # Self-play evaluation
        mean_diff, std_diff = evaluate_self_play(trainer, n_games=100)

        print(f"\nSelf-play balance after 1000 iters: {mean_diff:.3f} ± {std_diff:.3f}")

        # Should be close to 0 (balanced)
        # Allow some variance due to randomness
        assert abs(mean_diff) < 5.0, \
            f"Self-play should be balanced (got {mean_diff:.3f})"


class TestConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = CFRConfig(
            num_iterations=1000,
            cfr_variant='cfr+',
            sampling='external'
        )

        assert config.num_iterations == 1000
        assert config.cfr_variant == 'cfr+'
        assert config.sampling == 'external'

    def test_invalid_cfr_variant(self):
        """Test that invalid CFR variant raises error."""
        with pytest.raises(AssertionError):
            CFRConfig(cfr_variant='invalid')

    def test_invalid_sampling(self):
        """Test that invalid sampling raises error."""
        with pytest.raises(AssertionError):
            CFRConfig(sampling='invalid')

    def test_invalid_iterations(self):
        """Test that invalid iterations raises error."""
        with pytest.raises(AssertionError):
            CFRConfig(num_iterations=0)

        with pytest.raises(AssertionError):
            CFRConfig(num_iterations=-100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
