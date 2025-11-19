import gymnasium as gym
import numpy as np
from gymnasium import spaces
from snapszer_base import SnapszerState, NUM_CARDS, TOTAL_ACTIONS, OBSERVATION_SIZE

class SnapszerGymEnv(gym.Env):
    """
    Gymnasium environment for Snapszer.
    This environment represents a SINGLE player's perspective.
    The opponent's moves must be provided externally or simulated.
    
    However, for self-play training, it's often easier if the environment 
    handles the full game loop and switches control between players.
    
    In this implementation, step() will advance the game until it is the 
    current_player's turn again OR the game ends.
    This means we need a policy for the opponent.
    """
    
    def __init__(self, opponent_policy=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        
        self.state = None
        self.opponent_policy = opponent_policy # Function taking (observation, legal_actions) -> action
        self.player_id = 0 # The agent training in this env is always player 0 from its own perspective? 
                           # Actually, let's say this env controls `self.controlled_player`.
        self.controlled_player = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = SnapszerState.new(seed=seed)
        self.controlled_player = 0 # We can randomize this if we want to train as P1 too
        
        # If it's not our turn, opponent plays
        self._process_opponent_turns()
        
        return self._get_obs(), {"legal_actions": self.state.legal_actions()}

    def step(self, action):
        # Apply our action
        try:
            self.state.apply_action(action)
        except ValueError:
            # Illegal action penalty? Or just crash? 
            # For PPO with masking, this shouldn't happen. 
            # But if it does, let's end game with penalty.
            return self._get_obs(), -10.0, True, False, {"legal_actions": []}

        # Process opponent turns until it's our turn again or game over
        self._process_opponent_turns()
        
        terminated = self.state.is_terminal()
        truncated = False
        reward = 0.0
        
        if terminated:
            # Calculate reward
            # returns() gives (p0_score, p1_score)
            # We want reward for controlled_player
            scores = self.state.returns()
            reward = scores[self.controlled_player]
        
        info = {"legal_actions": self.state.legal_actions() if not terminated else []}
        
        return self._get_obs(), reward, terminated, truncated, info

    def _process_opponent_turns(self):
        while not self.state.is_terminal() and self.state.current_player != self.controlled_player:
            # Opponent's turn
            obs = self.state.observation_tensor(self.state.current_player)
            legal = self.state.legal_actions()
            
            if self.opponent_policy:
                # Opponent policy needs to handle the observation and return a valid action
                opp_action = self.opponent_policy(obs, legal)
            else:
                # Random fallback
                import random
                opp_action = random.choice(legal)
            
            self.state.apply_action(opp_action)

    def _get_obs(self):
        return np.array(self.state.observation_tensor(self.controlled_player), dtype=np.float32)

    def render(self):
        print(self.state.to_string(self.controlled_player))
