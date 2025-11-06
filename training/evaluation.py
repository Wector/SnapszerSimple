"""Evaluation functions for CFR training."""

import numpy as np
from snapszer import jax_optimized as game


def compute_best_response_value(trainer, player: int, n_games: int = 100) -> float:
    """
    Compute expected value of best response against opponent's strategy.

    This is used to calculate exploitability - how much an optimal opponent
    can exploit the current strategy.

    Args:
        trainer: CFRTrainer with learned strategy
        player: Player to compute best response for (0 or 1)
        n_games: Number of games to average over

    Returns:
        Expected value of best response
    """
    import jax

    total_value = 0.0

    for game_idx in range(n_games):
        # Use offset seed to avoid overlap with training
        key = jax.random.PRNGKey(game_idx + 10000)
        state = game.new_game(key)

        value = _best_response_recursive(state, player, trainer)
        total_value += value

    return total_value / n_games


def _best_response_recursive(state, player: int, trainer) -> float:
    """
    Recursively compute best response value.

    Args:
        state: Current game state
        player: Player computing best response
        trainer: CFRTrainer for opponent strategy

    Returns:
        Best response value at this state
    """
    # Terminal state
    if state.terminal:
        returns = game.returns(state)
        return float(returns[player])

    current_player = int(state.current_player)

    if current_player == player:
        # Our turn: maximize over all actions
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)
        best_value = -float('inf')

        for action in range(game.TOTAL_ACTIONS):
            if legal_mask[action]:
                next_state = game.apply_action(state, action)
                value = _best_response_recursive(next_state, player, trainer)
                best_value = max(best_value, value)

        return best_value if best_value > -float('inf') else 0.0

    else:
        # Opponent's turn: use their strategy
        strategy = trainer.get_strategy(state, current_player)
        legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)

        expected_value = 0.0
        for action in range(game.TOTAL_ACTIONS):
            if legal_mask[action]:
                next_state = game.apply_action(state, action)
                value = _best_response_recursive(next_state, player, trainer)
                expected_value += strategy[action] * value

        return expected_value


def compute_exploitability(trainer, n_games: int = 100) -> float:
    """
    Compute exploitability of current strategy.

    Exploitability = BR_0 + BR_1
    where BR_i is the expected value of best response for player i.

    In a zero-sum game at Nash equilibrium, exploitability = 0.
    Lower exploitability means closer to Nash equilibrium.

    Args:
        trainer: CFRTrainer with learned strategy
        n_games: Number of games to average over

    Returns:
        Exploitability value (lower is better, 0 = Nash)
    """
    br_0 = compute_best_response_value(trainer, 0, n_games)
    br_1 = compute_best_response_value(trainer, 1, n_games)

    # In zero-sum game, exploitability is sum of absolute best responses
    # (since Nash value should be 0)
    return abs(br_0) + abs(br_1)


def evaluate_vs_random(trainer, n_games: int = 100) -> float:
    """
    Test learned strategy against random baseline.

    Args:
        trainer: CFRTrainer with learned strategy
        n_games: Number of games to play

    Returns:
        Win rate for player 0 (fraction of wins)
    """
    import jax

    wins = 0
    draws = 0

    for game_idx in range(n_games):
        # Use offset seed
        key = jax.random.PRNGKey(game_idx + 20000)
        state = game.new_game(key)

        while not state.terminal:
            if state.current_player == 0:
                # Player 0: use learned strategy
                strategy = trainer.get_strategy(state, 0)
                action = np.random.choice(game.TOTAL_ACTIONS, p=strategy)
            else:
                # Player 1: random
                legal_mask = np.array(game.legal_actions_mask(state), dtype=bool)
                legal_actions = [a for a in range(game.TOTAL_ACTIONS) if legal_mask[a]]
                action = np.random.choice(legal_actions)

            state = game.apply_action(state, action)

        if state.winner == 0:
            wins += 1
        elif state.winner == -1:
            draws += 1

    return wins / n_games


def evaluate_self_play(trainer, n_games: int = 100) -> tuple:
    """
    Evaluate self-play performance.

    At Nash equilibrium, expected game points difference should be ~0
    (since both players play optimally).

    Args:
        trainer: CFRTrainer with learned strategy
        n_games: Number of games to play

    Returns:
        (mean_game_points_diff, std_game_points_diff)
    """
    import jax

    game_points_diffs = []

    for game_idx in range(n_games):
        # Use offset seed
        key = jax.random.PRNGKey(game_idx + 30000)
        state = game.new_game(key)

        while not state.terminal:
            player = int(state.current_player)
            strategy = trainer.get_strategy(state, player)
            action = np.random.choice(game.TOTAL_ACTIONS, p=strategy)
            state = game.apply_action(state, action)

        # Record game points difference (Player 0 - Player 1)
        diff = int(state.game_points[0]) - int(state.game_points[1])
        game_points_diffs.append(diff)

    mean_diff = np.mean(game_points_diffs)
    std_diff = np.std(game_points_diffs)

    return mean_diff, std_diff


def evaluate_strategy_entropy(trainer) -> float:
    """
    Compute average strategy entropy across all information sets.

    High entropy indicates mixed strategies (common in Nash equilibrium).
    Low entropy indicates deterministic strategies.

    Args:
        trainer: CFRTrainer with learned strategy

    Returns:
        Average entropy across all info sets
    """
    from scipy.stats import entropy

    entropies = []

    for player in [0, 1]:
        for info_set in trainer.info_sets[player].values():
            # Get average strategy
            from training.info_set import get_average_strategy
            avg_strategy = get_average_strategy(info_set)

            # Compute entropy (filter out zeros for numerical stability)
            probs = avg_strategy[avg_strategy > 0]
            if len(probs) > 0:
                ent = entropy(probs)
                entropies.append(ent)

    return np.mean(entropies) if entropies else 0.0
