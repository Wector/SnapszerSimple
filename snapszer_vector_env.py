import numpy as np
import torch
from snapszer_base import SnapszerState, TOTAL_ACTIONS, OBSERVATION_SIZE

class SnapszerBatchedEnv:
    """
    A custom vector environment for Snapszer that handles batched opponent inference.
    Instead of simulating games sequentially, it advances all games in parallel.
    When opponents need to move, it collects all their observations and queries the GPU once.
    """
    def __init__(self, num_envs, device, opponent_policy_fn):
        self.num_envs = num_envs
        self.device = device
        self.opponent_policy_fn = opponent_policy_fn # Function: (obs_tensor, legal_actions_mask) -> action_tensor
        
        self.states = [SnapszerState.new() for _ in range(num_envs)]
        self.controlled_player = 0
        
        # Pre-allocate buffers
        self.obs_buf = np.zeros((num_envs, OBSERVATION_SIZE), dtype=np.float32)
        self.rewards_buf = np.zeros(num_envs, dtype=np.float32)
        self.dones_buf = np.zeros(num_envs, dtype=bool)
        
        # We need to track legal actions for the agent
        self.legal_actions_mask = torch.zeros((num_envs, TOTAL_ACTIONS), dtype=torch.bool, device=device)

    def reset(self):
        for i in range(self.num_envs):
            self.states[i] = SnapszerState.new()
        
        # Advance all games until it's the controlled player's turn
        self._advance_to_turn()
        
        return self._get_obs(), self._get_infos()

    def step(self, actions):
        # actions: numpy array or tensor of shape (num_envs,)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
            
        self.rewards_buf.fill(0.0)
        self.dones_buf.fill(False)
        
        # Apply agent actions
        for i in range(self.num_envs):
            if self.states[i].is_terminal():
                # Auto-reset if it was terminal from previous step (shouldn't happen if handled correctly)
                self.states[i] = SnapszerState.new()
                # We might need to advance this new state, but let's handle it in the loop
            
            try:
                self.states[i].apply_action(actions[i])
            except ValueError:
                # Illegal action?
                # In PPO with masking, this shouldn't happen.
                # If it does, we could penalize or just reset.
                # Let's reset for stability.
                self.states[i] = SnapszerState.new()
                self.rewards_buf[i] = -1.0 # Penalty?
        
        # Advance games
        self._advance_to_turn()
        
        # Check for terminals and calculate rewards
        for i in range(self.num_envs):
            if self.states[i].is_terminal():
                self.dones_buf[i] = True
                # Calculate reward
                scores = self.states[i].returns()
                self.rewards_buf[i] = scores[self.controlled_player]
                
                # Reset immediately (Auto-reset)
                # Note: In standard Gym, we return the terminal obs and info contains the reset obs.
                # Here, for simplicity in PPO loop, we return the RESET obs directly.
                # But we must be careful about the reward (it belongs to the terminal state).
                self.states[i] = SnapszerState.new()
                # We need to advance the NEW state to player's turn
                # This might require another round of opponent moves?
                # For simplicity, let's assume the new state starts at P0 or P1.
                # If P1 starts, we need to simulate.
                # To avoid infinite recursion, we can do a quick advance for just this env.
                self._advance_single_env(i)

        return self._get_obs(), self.rewards_buf.copy(), self.dones_buf.copy(), self.dones_buf.copy(), self._get_infos()

    def _advance_to_turn(self):
        """
        Advances all environments until they are either:
        1. Waiting for the controlled_player's move.
        2. Terminal.
        
        This handles opponent moves in BATCHES.
        """
        while True:
            # Identify which envs need an opponent move
            opponent_indices = []
            opponent_obs = []
            opponent_legal_masks = []
            
            for i in range(self.num_envs):
                s = self.states[i]
                if not s.is_terminal() and s.current_player != self.controlled_player:
                    opponent_indices.append(i)
                    opponent_obs.append(s.observation_tensor(s.current_player))
                    
                    # Create mask for this state
                    mask = torch.zeros(TOTAL_ACTIONS, dtype=torch.bool)
                    for act in s.legal_actions():
                        mask[act] = True
                    opponent_legal_masks.append(mask)
            
            if not opponent_indices:
                break # No opponents need to move
            
            # Batch inference
            obs_tensor = torch.tensor(opponent_obs, dtype=torch.float32, device=self.device)
            mask_tensor = torch.stack(opponent_legal_masks).to(self.device)
            
            with torch.no_grad():
                actions = self.opponent_policy_fn(obs_tensor, mask_tensor)
                actions_np = actions.cpu().numpy()
            
            # Apply actions
            for idx, act in zip(opponent_indices, actions_np):
                self.states[idx].apply_action(act)

    def _advance_single_env(self, idx):
        """
        Advances a single environment (used during reset).
        We could batch this too, but resets are sparse.
        """
        s = self.states[idx]
        while not s.is_terminal() and s.current_player != self.controlled_player:
            # Simple random policy or query policy individually?
            # Let's query policy individually to be consistent, or just random for speed?
            # Better to be consistent.
            obs = torch.tensor([s.observation_tensor(s.current_player)], dtype=torch.float32, device=self.device)
            mask = torch.zeros((1, TOTAL_ACTIONS), dtype=torch.bool, device=self.device)
            for act in s.legal_actions():
                mask[0, act] = True
            
            with torch.no_grad():
                action = self.opponent_policy_fn(obs, mask)[0].item()
            s.apply_action(action)

    def _get_obs(self):
        for i in range(self.num_envs):
            self.obs_buf[i] = self.states[i].observation_tensor(self.controlled_player)
        return self.obs_buf.copy() # Return copy to be safe

    def _get_infos(self):
        # We need to construct the legal actions mask for the agent
        self.legal_actions_mask.fill_(False)
        infos = {"legal_actions": []} # Legacy support if needed
        
        for i in range(self.num_envs):
            acts = self.states[i].legal_actions() if not self.states[i].is_terminal() else []
            infos["legal_actions"].append(acts)
            for a in acts:
                self.legal_actions_mask[i, a] = True
                
        infos["legal_actions_mask"] = self.legal_actions_mask # Tensor on device
        return infos
