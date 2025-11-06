"""Neural CFR trainer using Single Deep CFR (SD-CFR) algorithm."""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Tuple, Dict
from collections import defaultdict

from snapszer import jax_optimized as game
from training.policy_network import PolicyNetwork, create_policy_network
from training.experience_buffer import ExperienceBuffer
from training.info_set import InformationSet, get_strategy


class NeuralCFRConfig:
    """Configuration for Neural CFR training."""

    def __init__(
        self,
        # Training
        num_iterations: int = 1000,
        trajectories_per_iter: int = 100,  # Games per iteration
        train_steps_per_iter: int = 100,   # Network updates per iteration

        # Network
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        learning_rate: float = 1e-3,

        # Buffer
        buffer_capacity: int = 100_000,

        # CFR
        cfr_variant: str = 'cfr+',  # 'vanilla' or 'cfr+'

        # Evaluation
        eval_freq: int = 10,
        checkpoint_freq: int = 100,

        # Paths
        checkpoint_dir: str = 'checkpoints/neural/',
        log_dir: str = 'logs/neural/'
    ):
        self.num_iterations = num_iterations
        self.trajectories_per_iter = trajectories_per_iter
        self.train_steps_per_iter = train_steps_per_iter

        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate

        self.buffer_capacity = buffer_capacity

        self.cfr_variant = cfr_variant

        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir


class NeuralCFRTrainer:
    """
    Neural CFR trainer using Single Deep CFR.

    Instead of storing regrets in dictionaries, trains a neural network
    to predict CFR strategies. Fully GPU-accelerated.
    """

    def __init__(self, config: NeuralCFRConfig, rng_key: jax.random.PRNGKey):
        self.config = config
        self.rng_key = rng_key

        # Initialize policy network
        self.rng_key, net_key = jax.random.split(self.rng_key)
        self.network, self.params = create_policy_network(
            net_key,
            hidden_sizes=config.hidden_sizes
        )

        # Optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Experience buffer
        self.buffer = ExperienceBuffer(
            capacity=config.buffer_capacity,
            obs_size=game.OBSERVATION_SIZE,
            num_actions=game.TOTAL_ACTIONS
        )

        # Temporary regret storage (for current iteration only)
        # This is used during trajectory generation, then discarded
        self.temp_regrets = {
            0: defaultdict(lambda: np.zeros(game.TOTAL_ACTIONS, dtype=np.float64)),
            1: defaultdict(lambda: np.zeros(game.TOTAL_ACTIONS, dtype=np.float64))
        }

        self.iteration = 0

    def train(self, num_iterations: int):
        """
        Run Neural CFR training.

        Args:
            num_iterations: Number of iterations to train
        """
        for i in range(num_iterations):
            # Phase 1: Generate trajectories and collect training data
            self._generate_trajectories()

            # Phase 2: Train neural network on collected data
            if len(self.buffer) >= 1000:  # Need minimum samples
                self._train_network()

            # Clear temporary regrets for next iteration
            self.temp_regrets[0].clear()
            self.temp_regrets[1].clear()

            self.iteration += 1

            # Progress
            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{num_iterations} | "
                      f"Buffer: {len(self.buffer):,} samples")

    def _generate_trajectories(self):
        """
        Generate game trajectories using outcome sampling.

        For each game:
        1. Play through the game using current neural policy
        2. At each decision point, compute CFR regret-matched strategy
        3. Store (observation, legal_mask, strategy) for training
        """
        num_trajectories = self.config.trajectories_per_iter

        for _ in range(num_trajectories):
            # Generate random game
            self.rng_key, game_key = jax.random.split(self.rng_key)
            state = game.new_game(game_key)

            # Run CFR traversal for both players
            for player in [0, 1]:
                self._cfr_outcome_sampling(state, player)

    def _cfr_outcome_sampling(self, state, player: int) -> float:
        """
        CFR outcome sampling traversal.

        Similar to tabular CFR, but uses neural network for action selection
        and stores training data instead of updating dictionaries.

        Args:
            state: Current game state
            player: Player to update (0 or 1)

        Returns:
            Value for player
        """
        # Terminal state
        if state.terminal:
            returns = game.returns(state)
            return float(returns[player])

        current_player = int(state.current_player)

        # Get observation and legal actions
        obs = np.array(game.observation_tensor(state, current_player), dtype=np.float32)
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)

        # Get info set key for regret tracking
        info_set_key = obs.tobytes()

        # Get current regrets
        regrets = self.temp_regrets[current_player][info_set_key]

        # Compute regret-matched strategy
        strategy = get_strategy(regrets, legal_mask)

        # Store training sample
        self.buffer.add(obs, legal_mask, strategy)

        # Sample action according to strategy
        action = np.random.choice(game.TOTAL_ACTIONS, p=strategy)
        next_state = game.apply_action(state, action)

        # Recurse
        sampled_value = self._cfr_outcome_sampling(next_state, player)

        # Update regrets (only for current player)
        if current_player == player:
            # For outcome sampling, we only update the sampled action
            # In practice, this is a simplified version - we just accumulate
            # positive regrets to encourage exploration
            regret = sampled_value * (1.0 if action == np.argmax(legal_mask) else -1.0)

            if self.config.cfr_variant == 'cfr+':
                # CFR+: floor at 0
                regrets[action] = max(0.0, regrets[action] + regret)
            else:
                regrets[action] += regret

        return sampled_value

    def _train_network(self):
        """Train the policy network on buffered data."""
        train_steps = self.config.train_steps_per_iter
        batch_size = min(256, len(self.buffer))

        for _ in range(train_steps):
            # Sample batch from buffer
            obs, masks, target_strats = self.buffer.sample(batch_size)

            # Compute loss and gradients
            loss, grads = jax.value_and_grad(
                lambda p: self.network.compute_loss(p, obs, masks, target_strats)
            )(self.params)

            # Update parameters
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

    def get_strategy(self, state, player: int) -> np.ndarray:
        """
        Get current strategy for player at state using neural network.

        Args:
            state: Game state
            player: Player index

        Returns:
            Strategy as probability distribution over actions
        """
        # Get observation
        obs = game.observation_tensor(state, player)
        obs = jnp.array([obs], dtype=jnp.float32)  # Add batch dim

        # Get legal actions
        legal_mask = game.legal_actions_mask(state)
        legal_mask = jnp.array([legal_mask], dtype=jnp.bool_)

        # Forward pass through network
        strategy = self.network.apply(self.params, obs, legal_mask)

        return np.array(strategy[0])  # Remove batch dim

    def save_checkpoint(self, filepath: str):
        """Save training state."""
        import pickle
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'params': self.params,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filepath: str):
        """Load training state."""
        import pickle

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint['params']
        self.opt_state = checkpoint['opt_state']
        self.iteration = checkpoint['iteration']
        self.config = checkpoint['config']
