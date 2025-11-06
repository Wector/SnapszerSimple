"""Neural CFR with ACTUAL learning (not just self-play)."""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Tuple
from collections import defaultdict

from snapszer import jax_optimized as game
from training.policy_network import PolicyNetwork, create_policy_network


class NeuralCFRLearningConfig:
    """Configuration for learning neural CFR."""

    def __init__(
        self,
        # Training
        num_iterations: int = 1000,

        # CFR sampling
        cfr_games_per_iter: int = 1000,  # Games to sample for CFR

        # Network training
        train_steps_per_iter: int = 100,
        batch_size: int = 512,

        # Network
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        learning_rate: float = 1e-3,

        # Reservoir buffer (stores diverse states)
        reservoir_size: int = 100_000,

        # Evaluation
        eval_freq: int = 10,
        checkpoint_freq: int = 50,

        # Paths
        checkpoint_dir: str = 'checkpoints/neural_learning/',
        log_dir: str = 'logs/neural_learning/'
    ):
        self.num_iterations = num_iterations
        self.cfr_games_per_iter = cfr_games_per_iter
        self.train_steps_per_iter = train_steps_per_iter
        self.batch_size = batch_size

        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate

        self.reservoir_size = reservoir_size

        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir


class ReservoirBuffer:
    """Reservoir sampling buffer for diverse state storage."""

    def __init__(self, capacity: int, obs_size: int, num_actions: int):
        self.capacity = capacity
        self.observations = np.zeros((capacity, obs_size), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.cfr_strategies = np.zeros((capacity, num_actions), dtype=np.float32)

        self.size = 0
        self.total_seen = 0

    def add(self, obs: np.ndarray, mask: np.ndarray, strategy: np.ndarray):
        """Add sample using reservoir sampling."""
        if self.size < self.capacity:
            # Fill buffer
            self.observations[self.size] = obs
            self.legal_masks[self.size] = mask
            self.cfr_strategies[self.size] = strategy
            self.size += 1
        else:
            # Reservoir sampling: random replacement
            idx = np.random.randint(0, self.total_seen + 1)
            if idx < self.capacity:
                self.observations[idx] = obs
                self.legal_masks[idx] = mask
                self.cfr_strategies[idx] = strategy

        self.total_seen += 1

    def sample(self, batch_size: int):
        """Sample random batch."""
        if self.size == 0:
            return None

        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        return (
            jnp.array(self.observations[indices]),
            jnp.array(self.legal_masks[indices]),
            jnp.array(self.cfr_strategies[indices])
        )


class NeuralCFRLearner:
    """
    Neural CFR with proper learning.

    Each iteration:
    1. Play games with current network policy
    2. Compute CFR regrets and strategies
    3. Store (state, CFR_strategy) pairs
    4. Train network to predict CFR strategies
    """

    def __init__(self, config: NeuralCFRLearningConfig, rng_key: jax.random.PRNGKey):
        self.config = config
        self.rng_key = rng_key

        # Initialize network
        self.rng_key, net_key = jax.random.split(self.rng_key)
        self.network, self.params = create_policy_network(
            net_key,
            hidden_sizes=config.hidden_sizes
        )

        # Optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Reservoir buffer
        self.buffer = ReservoirBuffer(
            capacity=config.reservoir_size,
            obs_size=game.OBSERVATION_SIZE,
            num_actions=game.TOTAL_ACTIONS
        )

        # Regret tracking (temporary, cleared each iteration)
        self.regret_sum = defaultdict(lambda: np.zeros(game.TOTAL_ACTIONS, dtype=np.float64))

        self.iteration = 0

        # Metrics for visualization
        self.metrics = {
            'iteration': [],
            'eval_iterations': [],
            'win_rate_vs_random': [],
            'avg_loss': [],
            'buffer_size': []
        }

    def train(self, num_iterations: int):
        """Train for multiple iterations."""
        for i in range(num_iterations):
            print(f"\nIteration {self.iteration + 1}/{self.iteration + num_iterations}")

            # Phase 1: Generate CFR data
            print("  Generating CFR strategies...")
            self._generate_cfr_data()

            # Phase 2: Train network
            print(f"  Training network on {self.buffer.size:,} states...")
            avg_loss = self._train_network()

            # Track metrics
            self.metrics['iteration'].append(self.iteration)
            self.metrics['avg_loss'].append(avg_loss)
            self.metrics['buffer_size'].append(self.buffer.size)

            # Clear regrets for next iteration
            self.regret_sum.clear()

            self.iteration += 1

            print(f"  Avg loss: {avg_loss:.4f} | Buffer: {self.buffer.size:,}")

    def _generate_cfr_data(self):
        """Generate games and compute CFR strategies."""
        num_games = self.config.cfr_games_per_iter

        for _ in range(num_games):
            # Play one game
            self.rng_key, game_key = jax.random.split(self.rng_key)
            state = game.new_game(game_key)

            # Run CFR traversal
            for player in [0, 1]:
                self._cfr_traversal(state, player)

    def _cfr_traversal(self, state, player: int):
        """CFR traversal to compute regrets and strategies."""
        if state.terminal:
            returns = game.returns(state)
            return float(returns[player])

        current_player = int(state.current_player)

        # Get observation and legal actions
        obs = np.array(game.observation_tensor(state, current_player), dtype=np.float32)
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)

        # Get info set key
        info_key = obs.tobytes()

        # Get current regrets
        regrets = self.regret_sum[info_key]

        # Compute CFR strategy via regret matching
        positive_regrets = np.maximum(0.0, regrets)
        masked_regrets = np.where(legal_mask, positive_regrets, 0.0)
        total = np.sum(masked_regrets)

        if total > 0:
            cfr_strategy = masked_regrets / total
        else:
            # Uniform over legal actions
            num_legal = np.sum(legal_mask)
            cfr_strategy = np.where(legal_mask, 1.0 / num_legal, 0.0) if num_legal > 0 else np.zeros(game.TOTAL_ACTIONS)

        # Store this CFR strategy in buffer
        self.buffer.add(obs, legal_mask, cfr_strategy)

        # Sample action according to CFR strategy
        action = np.random.choice(game.TOTAL_ACTIONS, p=cfr_strategy)
        next_state = game.apply_action(state, action)

        # Recurse
        value = self._cfr_traversal(next_state, player)

        # Update regrets (only for current player)
        if current_player == player:
            # Simplified regret update (outcome sampling style)
            for a in range(game.TOTAL_ACTIONS):
                if legal_mask[a]:
                    if a == action:
                        regret = 0.0  # Baseline
                    else:
                        # Estimate: assume other actions would do better
                        regret = -value  # Simplified

                    # CFR+ : floor at 0
                    regrets[a] = max(0.0, regrets[a] + regret)

        return value

    def _train_network(self) -> float:
        """Train network on buffered CFR strategies."""
        if self.buffer.size < self.config.batch_size:
            return 0.0

        total_loss = 0.0
        num_steps = self.config.train_steps_per_iter

        for _ in range(num_steps):
            # Sample batch
            batch = self.buffer.sample(self.config.batch_size)
            if batch is None:
                continue

            obs, masks, target_strats = batch

            # Compute loss and update
            loss, grads = jax.value_and_grad(
                lambda p: self.network.compute_loss(p, obs, masks, target_strats)
            )(self.params)

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            total_loss += float(loss)

        return total_loss / num_steps if num_steps > 0 else 0.0

    def get_strategy(self, state, player: int) -> np.ndarray:
        """Get current network policy."""
        obs = game.observation_tensor(state, player)
        obs = jnp.array([obs], dtype=jnp.float32)

        legal_mask = game.legal_actions_mask(state)
        legal_mask = jnp.array([legal_mask], dtype=jnp.bool_)

        strategy = self.network.apply(self.params, obs, legal_mask)
        return np.array(strategy[0])

    def save_checkpoint(self, filepath: str):
        """Save training state."""
        import pickle
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'params': self.params,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'metrics': self.metrics,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
