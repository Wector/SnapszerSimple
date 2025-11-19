import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from snapszer_base import SnapszerState, TOTAL_ACTIONS, OBSERVATION_SIZE, card_str
import argparse
import os

# --- Re-define Agent Architecture (Must match training) ---
# Ideally this should be imported, but for standalone script we can duplicate or import if refactored.
# We will assume the structure from train_ppo_selfplay.py
# We need to know hidden size. We'll add it as arg.

class Agent(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, TOTAL_ACTIONS),
        )

    def get_action_and_value(self, x, action=None, legal_actions_mask=None):
        logits = self.actor(x)
        if legal_actions_mask is not None:
            logits = logits.masked_fill(legal_actions_mask == 0, -1e9)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def safe_card_str(cid):
    if cid is None: return "None"
    return card_str(cid)

def print_state(state, human_player_id):
    print("\n" + "="*40)
    print(f"Score: You {state.points[human_player_id]} - {state.points[1-human_player_id]} Agent")
    gp_str = f"You {state.game_points[human_player_id]} - {state.game_points[1-human_player_id]} Agent" if state.game_points else "0 - 0"
    print(f"Game Points: {gp_str}")
    print(f"Trump: {safe_card_str(state.trump_card)}")
    print(f"Stock: {state.stock_remaining()} cards")
    print(f"State: {'CLOSED' if state.closed else 'OPEN'}")
    
    print(f"\nTrick: {safe_card_str(state.trick_cards[0])} (Leader: {'You' if state.leader == human_player_id else 'Agent'})")
    
    print("\nYour Hand:")
    hand = state.hands[human_player_id]
    for i, cid in enumerate(hand):
        print(f"  {i}: {safe_card_str(cid)}")

def get_human_action(state, human_player_id):
    legal = state.legal_actions()
    print("\nLegal Actions:")
    # Map actions to readable strings
    # 0-19: Play Card (if in hand)
    # 20: Exchange Trump
    # 21: Close Talon
    # 22-25: Announce Marriage (implicit in play usually, but here explicit actions?)
    # In snapszer_base, actions are just card indices for playing, or special ints.
    
    # Let's look at snapszer_base.py action definitions if possible.
    # Assuming:
    # 0-19: Play card with ID 0-19
    # 20: Exchange Trump
    # 21: Close Talon
    # 22-41: Play card + Marriage? No, usually marriage is automatic or specific action.
    # Let's assume standard mapping from base.
    
    # We need to map back action ID to description
    action_map = {}
    for act in legal:
        if act < 20:
            desc = f"Play {safe_card_str(act)}"
        elif act == 20:
            desc = "Exchange Trump Jack"
        elif act == 21:
            desc = "Close Talon"
        else:
            desc = f"Action {act}"
        action_map[act] = desc
        print(f"  [{act}] {desc}")
        
    while True:
        try:
            choice = int(input("Choose action: "))
            if choice in legal:
                return choice
            print("Invalid action.")
        except ValueError:
            print("Please enter a number.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    agent = Agent(hidden_size=args.hidden_size).to(args.device)
    agent.load_state_dict(torch.load(args.model, map_location=args.device))
    agent.eval()
    
    human_player = 0
    state = SnapszerState.new()
    
    while True:
        if state.is_terminal():
            print("\n" + "="*40)
            print("GAME OVER")
            scores = state.returns()
            print(f"Result: You {scores[human_player]} - {scores[1-human_player]} Agent")
            break

        if state.current_player == human_player:
            print_state(state, human_player)
            action = get_human_action(state, human_player)
            state.apply_action(action)
        else:
            # Agent turn
            obs = torch.tensor(state.observation_tensor(state.current_player), dtype=torch.float32).to(args.device).unsqueeze(0)
            legal = state.legal_actions()
            mask = torch.zeros((1, TOTAL_ACTIONS)).to(args.device)
            for act in legal:
                mask[0, act] = 1.0
            
            with torch.no_grad():
                action_tensor, _, _, _ = agent.get_action_and_value(obs, legal_actions_mask=mask)
                action = action_tensor.item()
            
            print(f"\nAgent chose action: {action}") # Improve description later
            state.apply_action(action)

if __name__ == "__main__":
    main()
