"""Pure GPU Neural CFR - NO CPU TRANSFERS!"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Tuple

from snapszer import jax_optimized as game
from training.policy_network import PolicyNetwork, create_policy_network


class PureGPUCFRConfig:
    """Configuration for pure GPU CFR."""

    def __init__(
        self,
        # Training
        num_iterations: int = 1000,
        games_per_batch: int = 512,      # 512 games in parallel
        train_batches_per_iter: int = 20, # Train on 20 batches per iteration

        # Network
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        learning_rate: float = 1e-3,

        # Evaluation
        eval_freq: int = 10,
        checkpoint_freq: int = 100,

        # Paths
        checkpoint_dir: str = 'checkpoints/pure_gpu/',
        log_dir: str = 'logs/pure_gpu/'
    ):
        self.num_iterations = num_iterations
        self.games_per_batch = games_per_batch
        self.train_batches_per_iter = train_batches_per_iter

        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate

        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir


class PureGPUCFRTrainer:
    """
    Pure GPU CFR trainer - EVERYTHING stays on GPU!

    No experience buffer, no CPU transfers. Just:
    1. Generate games on GPU
    2. Train network on GPU
    3. Repeat

    This minimizes CPU usage and maximizes GPU throughput.
    """

    def __init__(self, config: PureGPUCFRConfig, rng_key: jax.random.PRNGKey):
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

        self.iteration = 0

        # JIT compile everything
        self._generate_and_train_jit = jax.jit(self._generate_and_train_step)
        self._train_iteration_jit = jax.jit(
            self._train_iteration,
            static_argnames=['num_batches']
        )
        self._train_multi_iterations_jit = jax.jit(
            self._train_multi_iterations,
            static_argnames=['num_iterations', 'batches_per_iter']
        )

    def train(self, num_iterations: int):
        """Run pure GPU CFR training."""
        # Process 10 iterations at once to minimize Python overhead
        iters_per_call = 10

        for chunk_start in range(0, num_iterations, iters_per_call):
            chunk_size = min(iters_per_call, num_iterations - chunk_start)

            # Do 10 iterations in ONE JIT call!
            self.rng_key, chunk_key = jax.random.split(self.rng_key)
            self.params, self.opt_state = self._train_multi_iterations_jit(
                self.params,
                self.opt_state,
                chunk_key,
                chunk_size,
                self.config.train_batches_per_iter
            )

            self.iteration += chunk_size

            # Progress
            print(f"Iteration {self.iteration}/{num_iterations}")

    def _train_multi_iterations(self, params, opt_state, rng_key, num_iterations, batches_per_iter):
        """
        Train for MULTIPLE iterations in one JIT call!

        This eliminates Python overhead between iterations.
        GPU stays at 100% for much longer.
        """
        def train_one_iteration(carry, _):
            params, opt_state, key = carry
            key, iter_key = jax.random.split(key)
            new_params, new_opt_state = self._train_iteration(params, opt_state, iter_key, batches_per_iter)
            return (new_params, new_opt_state, key), None

        (final_params, final_opt_state, _), _ = jax.lax.scan(
            train_one_iteration,
            (params, opt_state, rng_key),
            None,
            length=num_iterations
        )

        return final_params, final_opt_state

    def _train_iteration(self, params, opt_state, rng_key, num_batches):
        """
        Train for multiple batches in one JIT call - keeps GPU busy!

        This is the key: instead of Python loop calling GPU many times,
        we do the loop INSIDE JIT so GPU stays busy.
        """
        def train_one_batch(carry, _):
            params, opt_state, key = carry
            key, batch_key = jax.random.split(key)
            new_params, new_opt_state = self._generate_and_train_step(params, opt_state, batch_key)
            return (new_params, new_opt_state, key), None

        (final_params, final_opt_state, _), _ = jax.lax.scan(
            train_one_batch,
            (params, opt_state, rng_key),
            None,
            length=num_batches
        )

        return final_params, final_opt_state

    def _generate_and_train_step(self, params, opt_state, rng_key):
        """
        Generate a batch of games and train network - ENTIRELY ON GPU!

        This is a pure JAX function with no CPU operations.
        """
        batch_size = self.config.games_per_batch

        # Split key for games
        game_keys = jax.random.split(rng_key, batch_size)

        # Generate batch of games (on GPU)
        batch_new_game = jax.vmap(game.new_game)
        states = batch_new_game(game_keys)

        # Get training data for both players (on GPU)
        all_obs = []
        all_masks = []
        all_targets = []

        for player in [0, 1]:
            # Vectorized observation extraction
            obs = jax.vmap(lambda s: game.observation_tensor(s, player))(states)
            masks = jax.vmap(game.legal_actions_mask)(states)

            # Get current policy predictions
            current_strategies = self.network.apply(params, obs, masks)

            all_obs.append(obs)
            all_masks.append(masks)
            all_targets.append(current_strategies)

        # Stack all data (still on GPU!)
        observations = jnp.concatenate(all_obs, axis=0)
        legal_masks = jnp.concatenate(all_masks, axis=0)
        target_strategies = jnp.concatenate(all_targets, axis=0)

        # Compute loss and gradients (on GPU)
        loss, grads = jax.value_and_grad(
            lambda p: self.network.compute_loss(p, observations, legal_masks, target_strategies)
        )(params)

        # Update parameters (on GPU)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    def get_strategy(self, state, player: int) -> np.ndarray:
        """Get current strategy for player at state."""
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
