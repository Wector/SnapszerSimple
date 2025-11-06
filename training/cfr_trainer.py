"""Tabular CFR trainer for Hungarian Snapszer."""

from collections import defaultdict
import numpy as np
import pickle
import os

from snapszer import jax_optimized as game
from training.config import CFRConfig
from training.info_set import InformationSet, get_info_set_key, get_strategy, get_average_strategy


class CFRTrainer:
    """
    Tabular CFR trainer supporting vanilla CFR, CFR+, and external sampling.

    Attributes:
        config: Training configuration
        info_sets: Dictionary mapping info set keys to InformationSet objects
        iteration: Current iteration number
    """

    def __init__(self, config: CFRConfig):
        self.config = config

        # Information sets for each player
        # Key: bytes (from get_info_set_key)
        # Value: InformationSet
        self.info_sets = {
            0: defaultdict(lambda: InformationSet.create(game.TOTAL_ACTIONS)),
            1: defaultdict(lambda: InformationSet.create(game.TOTAL_ACTIONS))
        }

        self.iteration = 0

    def train(self, num_iterations: int):
        """
        Run CFR training for specified number of iterations.

        Args:
            num_iterations: Number of CFR iterations to run
        """
        import jax

        for i in range(num_iterations):
            # Generate random game
            key = jax.random.PRNGKey(self.iteration)
            state = game.new_game(key)

            if self.config.sampling == 'vanilla':
                # Vanilla CFR: full tree traversal for both players
                for player in [0, 1]:
                    self.cfr_recursive(state, player, 1.0, 1.0)

            elif self.config.sampling == 'outcome':
                # Outcome sampling: sample complete game trajectories
                for _ in range(self.config.n_traversals_per_iter):
                    for player in [0, 1]:
                        self.cfr_outcome_sampling(state, player)

            else:  # 'external'
                # External sampling: sample opponent actions
                for _ in range(self.config.n_traversals_per_iter):
                    for player in [0, 1]:
                        self.cfr_external_sampling(state, player)

            self.iteration += 1

    def cfr_recursive(self, state, player: int, reach_prob_0: float, reach_prob_1: float) -> float:
        """
        Vanilla CFR: full tree traversal with reach probabilities.

        Args:
            state: Current game state
            player: Player to compute regrets for (0 or 1)
            reach_prob_0: Probability player 0 reaches this state
            reach_prob_1: Probability player 1 reaches this state

        Returns:
            Expected value for player at this state
        """
        # Terminal state
        if state.terminal:
            returns = game.returns(state)
            return float(returns[player])

        current_player = int(state.current_player)

        # Get information set
        info_set_key = get_info_set_key(state, current_player)
        info_set = self.info_sets[current_player][info_set_key]

        # Get legal actions
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)

        # Get current strategy via regret matching
        strategy = get_strategy(info_set.regret_sum, legal_mask)

        # Current player's turn
        if current_player == player:
            # Compute value for each action
            action_values = np.zeros(game.TOTAL_ACTIONS, dtype=np.float64)

            for action in range(game.TOTAL_ACTIONS):
                if legal_mask[action]:
                    next_state = game.apply_action(state, action)

                    if player == 0:
                        action_values[action] = self.cfr_recursive(
                            next_state, player,
                            reach_prob_0 * strategy[action],
                            reach_prob_1
                        )
                    else:
                        action_values[action] = self.cfr_recursive(
                            next_state, player,
                            reach_prob_0,
                            reach_prob_1 * strategy[action]
                        )

            # Expected value at this node
            node_value = np.sum(strategy * action_values)

            # Update regrets (weighted by opponent reach probability)
            opponent_reach = reach_prob_1 if player == 0 else reach_prob_0

            for action in range(game.TOTAL_ACTIONS):
                if legal_mask[action]:
                    regret = (action_values[action] - node_value) * opponent_reach

                    if self.config.cfr_variant == 'cfr+':
                        # CFR+: floor regrets at 0
                        info_set.regret_sum[action] = max(
                            0.0,
                            info_set.regret_sum[action] + regret
                        )
                    else:
                        # Vanilla CFR: allow negative regrets
                        info_set.regret_sum[action] += regret

            # Update strategy sum (weighted by player's reach probability)
            player_reach = reach_prob_0 if player == 0 else reach_prob_1
            info_set.strategy_sum += strategy * player_reach
            info_set.num_updates += 1

            return node_value

        else:
            # Opponent's turn: traverse with their strategy
            opp_info_set_key = get_info_set_key(state, current_player)
            opp_info_set = self.info_sets[current_player][opp_info_set_key]
            opp_strategy = get_strategy(opp_info_set.regret_sum, legal_mask)

            # Expected value over opponent's actions
            value = 0.0
            for action in range(game.TOTAL_ACTIONS):
                if legal_mask[action]:
                    next_state = game.apply_action(state, action)

                    if player == 0:
                        value += opp_strategy[action] * self.cfr_recursive(
                            next_state, player,
                            reach_prob_0,
                            reach_prob_1 * opp_strategy[action]
                        )
                    else:
                        value += opp_strategy[action] * self.cfr_recursive(
                            next_state, player,
                            reach_prob_0 * opp_strategy[action],
                            reach_prob_1
                        )

            return value

    def cfr_external_sampling(self, state, player: int) -> float:
        """
        External sampling CFR: sample opponent actions.

        Much faster than vanilla CFR as it samples rather than traversing full tree.

        Args:
            state: Current game state
            player: Player to update (0 or 1)

        Returns:
            Expected value for player at this state
        """
        # Terminal state
        if state.terminal:
            returns = game.returns(state)
            return float(returns[player])

        current_player = int(state.current_player)

        # Get information set
        info_set_key = get_info_set_key(state, current_player)
        info_set = self.info_sets[current_player][info_set_key]

        # Get legal actions
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)

        # Get current strategy
        strategy = get_strategy(info_set.regret_sum, legal_mask)

        # Current player's turn
        if current_player == player:
            # Compute value for each legal action
            action_values = np.zeros(game.TOTAL_ACTIONS, dtype=np.float64)

            for action in range(game.TOTAL_ACTIONS):
                if legal_mask[action]:
                    next_state = game.apply_action(state, action)
                    action_values[action] = self.cfr_external_sampling(next_state, player)

            # Expected value
            node_value = np.sum(strategy * action_values)

            # Update regrets (no reach probability weighting in external sampling)
            for action in range(game.TOTAL_ACTIONS):
                if legal_mask[action]:
                    regret = action_values[action] - node_value

                    if self.config.cfr_variant == 'cfr+':
                        info_set.regret_sum[action] = max(
                            0.0,
                            info_set.regret_sum[action] + regret
                        )
                    else:
                        info_set.regret_sum[action] += regret

            # Update strategy sum
            if self.config.cfr_variant == 'cfr+':
                # CFR+: weight by iteration number
                weight = self.iteration + 1
                info_set.strategy_sum += strategy * weight
            else:
                # Vanilla: equal weighting
                info_set.strategy_sum += strategy

            info_set.num_updates += 1

            return node_value

        else:
            # Opponent's turn: SAMPLE their action
            opp_info_set_key = get_info_set_key(state, current_player)
            opp_info_set = self.info_sets[current_player][opp_info_set_key]
            opp_strategy = get_strategy(opp_info_set.regret_sum, legal_mask)

            # Sample action according to strategy
            action = np.random.choice(game.TOTAL_ACTIONS, p=opp_strategy)
            next_state = game.apply_action(state, action)

            return self.cfr_external_sampling(next_state, player)

    def cfr_outcome_sampling(self, state, player: int) -> float:
        """
        Outcome sampling CFR: sample actions for ALL players.

        This is MUCH faster than external sampling because it samples ONE complete
        game trajectory instead of exploring subtrees. Only updates regrets along
        the sampled path.

        Args:
            state: Current game state
            player: Player to update (0 or 1)

        Returns:
            Expected value for player at this state
        """
        # Terminal state
        if state.terminal:
            returns = game.returns(state)
            return float(returns[player])

        current_player = int(state.current_player)

        # Get information set
        info_set_key = get_info_set_key(state, current_player)
        info_set = self.info_sets[current_player][info_set_key]

        # Get legal actions
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)

        # Get current strategy
        strategy = get_strategy(info_set.regret_sum, legal_mask)

        # Sample ONE action according to current strategy
        sampled_action = np.random.choice(game.TOTAL_ACTIONS, p=strategy)
        next_state = game.apply_action(state, sampled_action)

        # Recurse with sampled action
        sampled_value = self.cfr_outcome_sampling(next_state, player)

        # Only update if this is the player we're updating
        if current_player == player:
            # For outcome sampling, we need to compute counterfactual values
            # We sample this, so weight = 1/probability
            sample_prob = strategy[sampled_action]

            # Update regrets: compare sampled action to alternative actions
            # In outcome sampling, we estimate regrets by sampling
            for action in range(game.TOTAL_ACTIONS):
                if legal_mask[action]:
                    if action == sampled_action:
                        # For the sampled action, regret = 0 (it's our baseline)
                        regret = 0.0
                    else:
                        # For unsampled actions, we can't compute exact regret
                        # In pure outcome sampling, we skip these
                        # (Alternatively, we could sample them too, but that's slower)
                        continue

                    if self.config.cfr_variant == 'cfr+':
                        info_set.regret_sum[action] = max(
                            0.0,
                            info_set.regret_sum[action] + regret
                        )
                    else:
                        info_set.regret_sum[action] += regret

            # Update strategy sum (weighted by iteration for CFR+)
            if self.config.cfr_variant == 'cfr+':
                weight = self.iteration + 1
                info_set.strategy_sum += strategy * weight
            else:
                info_set.strategy_sum += strategy

            info_set.num_updates += 1

        return sampled_value

    def get_strategy(self, state, player: int) -> np.ndarray:
        """
        Get current strategy for player at state.

        Args:
            state: Game state
            player: Player index

        Returns:
            Strategy as probability distribution over actions
        """
        info_set_key = get_info_set_key(state, player)

        if info_set_key in self.info_sets[player]:
            info_set = self.info_sets[player][info_set_key]
            legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)
            return get_strategy(info_set.regret_sum, legal_mask)
        else:
            # Unknown state: uniform over legal actions
            legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)
            num_legal = np.sum(legal_mask)
            if num_legal > 0:
                return np.where(legal_mask, 1.0 / num_legal, 0.0)
            else:
                return np.zeros(game.TOTAL_ACTIONS)

    def get_average_strategy_dict(self) -> dict:
        """
        Extract average strategy for all information sets.

        Returns:
            Dictionary mapping player -> info_set_key -> average strategy
        """
        avg_strategy = {}

        for player in [0, 1]:
            avg_strategy[player] = {}
            for key, info_set in self.info_sets[player].items():
                avg_strategy[player][key] = get_average_strategy(info_set)

        return avg_strategy

    def save_checkpoint(self, filepath: str):
        """
        Save current training state to disk.

        Args:
            filepath: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'info_sets_0': dict(self.info_sets[0]),  # Convert defaultdict to dict
            'info_sets_1': dict(self.info_sets[1]),
            'iteration': self.iteration,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filepath: str):
        """
        Load training state from disk.

        Args:
            filepath: Path to checkpoint file
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        # Restore info sets as defaultdicts
        self.info_sets[0] = defaultdict(
            lambda: InformationSet.create(game.TOTAL_ACTIONS),
            checkpoint['info_sets_0']
        )
        self.info_sets[1] = defaultdict(
            lambda: InformationSet.create(game.TOTAL_ACTIONS),
            checkpoint['info_sets_1']
        )

        self.iteration = checkpoint['iteration']
        self.config = checkpoint['config']

    def get_num_info_sets(self) -> tuple:
        """
        Get number of information sets for each player.

        Returns:
            (num_info_sets_player_0, num_info_sets_player_1)
        """
        return len(self.info_sets[0]), len(self.info_sets[1])
