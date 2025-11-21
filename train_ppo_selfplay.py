import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions.categorical import Categorical
from snapszer_gym import SnapszerGymEnv
from snapszer_base import TOTAL_ACTIONS, OBSERVATION_SIZE
import time
import os

import argparse

# --- STRICT GPU CHECK ---
if not torch.cuda.is_available():
    raise RuntimeError("CRITICAL ERROR: GPU not detected! This script requires a CUDA-capable GPU (like your RTX 4090) to run. Aborting.")

print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda")
# ------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden layer size")
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Total timesteps")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (short duration)")
    parser.add_argument("--eval-freq", type=int, default=10, help="Evaluate every N updates")
    parser.add_argument("--exp-name", type=str, default="snapszer_ppo", help="Experiment name for saving models")
    parser.add_argument("--save-freq", type=int, default=50, help="Save model every N updates")
    parser.add_argument("--load-model", type=str, default="", help="Path to model checkpoint to load")
    args = parser.parse_args()
    return args

args = parse_args()

# Hyperparameters
LEARNING_RATE = 2.5e-4
NUM_STEPS = 128
NUM_ENVS = args.num_envs
TOTAL_TIMESTEPS = args.total_timesteps if not args.benchmark else NUM_ENVS * NUM_STEPS * 5 # 5 updates for benchmark
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = (NUM_ENVS * NUM_STEPS) // 4 # Auto-scale minibatch
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, TOTAL_ACTIONS),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, legal_actions_mask=None):
        logits = self.actor(x)
        
        # Action Masking: Set illegal actions to -inf
        if legal_actions_mask is not None:
            logits = logits.masked_fill(legal_actions_mask == 0, -1e9)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

from snapszer_vector_env import SnapszerBatchedEnv
from snapszer_gym import SnapszerGymEnv

def opponent_policy_fn(obs_tensor, mask_tensor):
    # Batched inference function
    # obs_tensor: (batch, obs_size)
    # mask_tensor: (batch, actions)
    global latest_agent
    
    # If no agent yet, random actions
    if latest_agent is None:
        # We need to return random valid actions
        # This is a bit tricky on GPU efficiently without an agent
        # Let's just use a simple CPU fallback or a dummy agent?
        # Actually, let's just wait for first update or initialize agent before loop.
        # But for now, let's assume latest_agent is initialized at start of train()
        pass

    with torch.no_grad():
        logits = latest_agent.actor(obs_tensor)
        logits = logits.masked_fill(~mask_tensor, -1e9)
        probs = Categorical(logits=logits)
        action = probs.sample()
        
    return action

import json
from snapszer_base import cid_suit, cid_rank, trick_winner, SUITS, RANK_STRENGTH

class MetricsLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, metrics):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

def evaluate_detailed(agent, update_step, num_games=100):
    """Evaluates the agent against a random opponent and collects detailed metrics."""
    wins = 0
    draws = 0
    losses = 0
    
    # Detailed stats
    game_points_dist = {3: 0, 2: 0, 1: 0, 0: 0} # 0 for loss
    marriage_opportunities = 0
    marriage_declarations = 0
    
    # New metrics
    trump_exchanges = 0
    talon_closures = 0
    
    # Talon specific tracking
    games_with_talon_close = 0
    wins_with_talon_close = 0
    
    total_point_diff = 0
    
    # Create a separate env for evaluation
    def random_opp(obs, legal):
        import random
        return random.choice(legal)
        
    env = SnapszerGymEnv(opponent_policy=random_opp)
    
    for _ in range(num_games):
        obs, info = env.reset()
        done = False
        talon_closed_in_this_game = False
        
        while not done:
            # Agent's turn
            # Check for marriage opportunity (Agent is leader, has K+Q)
            if env.state.can_declare_marriage(env.controlled_player):
                marriage_opportunities += 1
            
            is_responder = (env.state.leader != env.controlled_player)
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            legal = info["legal_actions"]
            mask = torch.zeros((1, TOTAL_ACTIONS)).to(device)
            for act in legal:
                mask[0, act] = 1.0
                
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor, legal_actions_mask=mask)
                
            act_val = action.item()
            
            # Track special actions
            from snapszer_base import EXCHANGE_TRUMP_ACTION, CLOSE_TALON_ACTION
            if act_val == EXCHANGE_TRUMP_ACTION:
                trump_exchanges += 1
            elif act_val == CLOSE_TALON_ACTION:
                talon_closures += 1
                talon_closed_in_this_game = True
            
            points_before = env.state.points[env.controlled_player]
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(act_val)
            done = terminated or truncated
            
            # Post-action checks
            points_after = env.state.points[env.controlled_player]
            
            # 1. Marriage Declaration Check
            if not is_responder:
                diff = points_after - points_before
                if diff == 20 or diff == 40:
                    marriage_declarations += 1
                            
            if done:
                # Calculate point difference
                p_agent = env.state.points[env.controlled_player]
                p_opp = env.state.points[1-env.controlled_player]
                total_point_diff += (p_agent - p_opp)

                if reward > 0:
                    wins += 1
                    gp = int(reward)
                    if gp in game_points_dist:
                        game_points_dist[gp] += 1
                    
                    if talon_closed_in_this_game:
                        wins_with_talon_close += 1
                        
                elif reward < 0:
                    losses += 1
                    game_points_dist[0] += 1
                else:
                    draws += 1
                
                if talon_closed_in_this_game:
                    games_with_talon_close += 1
                    
    win_rate = (wins / (wins + draws + losses)) * 100 if (wins + draws + losses) > 0 else 0
    
    metrics = {
        "update": update_step,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "gp_3": game_points_dist[3],
        "gp_2": game_points_dist[2],
        "gp_1": game_points_dist[1],
        "marriage_decl_rate": (marriage_declarations / marriage_opportunities) * 100 if marriage_opportunities > 0 else 0,
        "trump_exchange_rate": (trump_exchanges / num_games) * 100,
        "talon_close_rate": (talon_closures / num_games) * 100,
        "talon_close_count": talon_closures,
        "talon_close_win_rate": (wins_with_talon_close / games_with_talon_close) * 100 if games_with_talon_close > 0 else 0,
        "avg_point_diff": total_point_diff / num_games,
        "timestamp": time.time()
    }
    
    return metrics

def train():
    global latest_agent
    
    # Setup directories
    model_dir = os.path.join("models", args.exp_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize Agent
    agent = Agent().to(device)
    
    # Load checkpoint if requested
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        agent.load_state_dict(torch.load(args.load_model))
        
    latest_agent = agent # Initialize immediately so opponent has a policy
    
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Initialize Batched Env
    # Note: SnapszerBatchedEnv takes (num_envs, device, policy_fn)
    envs = SnapszerBatchedEnv(NUM_ENVS, device, opponent_policy_fn)
    
    # Storage
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, OBSERVATION_SIZE)).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    
    # Starting state
    # SnapszerBatchedEnv returns (obs_np, infos)
    next_obs_np, infos = envs.reset()
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    start_time = time.time()
    
    # Initialize Metrics Logger
    metrics_logger = MetricsLogger(os.path.join(model_dir, "metrics.jsonl"))

    print(f"Starting training loop on {device}...")
    print(f"{'Update':<8} | {'SPS':<8} | {'Win Rate (vs Rnd)':<18} | {'Loss':<8}")
    print("-" * 50)
    
    for update in range(1, TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS) + 1):
        latest_agent = agent 
        
        frac = 1.0 - (update - 1.0) / (TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS))
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get legal actions mask directly from infos (it's a tensor now!)
            current_masks = infos["legal_actions_mask"]
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, legal_actions_mask=current_masks)
                
                # DEBUG: FORCE TALON CLOSE with 30% probability if available
                # This acts as a targeted exploration boost for this specific action
                force_indices = (current_masks[:, 21] == 1).nonzero(as_tuple=True)[0]
                if len(force_indices) > 0:
                    # Create a mask for 20% probability
                    probs = torch.rand(len(force_indices)).to(device)
                    to_force = force_indices[probs < 0.2]
                    
                    # PROTECT EXCHANGE: If agent chose EXCHANGE (20), do NOT force close
                    # We want to encourage exchange, not punish it
                    if len(to_force) > 0:
                        # Filter out indices where agent chose 20
                        current_actions = action[to_force]
                        to_force = to_force[current_actions != 20]
                        
                        if len(to_force) > 0:
                            action[to_force] = 21
                    
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Execute
            # envs.step takes tensor or numpy
            next_obs_np, reward, terminated, truncated, infos = envs.step(action)
            
            # reward is numpy array
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # done is numpy array
            done = np.logical_or(terminated, truncated)
            next_obs, next_done = torch.Tensor(next_obs_np).to(device), torch.Tensor(done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1, OBSERVATION_SIZE))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(NUM_STEPS * NUM_ENVS)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS * NUM_ENVS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if True: 
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_COEF,
                    CLIP_COEF,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        sps = int((NUM_STEPS * NUM_ENVS) / (time.time() - start_time))
        
        # Evaluation
        win_rate_str = "-"
        if update % args.eval_freq == 0:
            metrics = evaluate_detailed(agent, update, num_games=100)
            metrics_logger.log(metrics)
            win_rate_str = f"{metrics['win_rate']:.1f}%"
            
        # Checkpointing
        if update % args.save_freq == 0:
            save_path = os.path.join(model_dir, f"agent_{update}.pt")
            latest_path = os.path.join(model_dir, "latest.pt")
            torch.save(agent.state_dict(), save_path)
            torch.save(agent.state_dict(), latest_path)
            # print(f"Model saved to {save_path}") # Optional: keep output clean
            
        print(f"{update:<8} | {sps:<8} | {win_rate_str:<18} | {loss.item():.4f}")
        start_time = time.time()

    # Final save
    final_path = os.path.join(model_dir, "final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # envs.close() # BatchedEnv might not have close()
    print("Training complete.")

if __name__ == "__main__":
    train()
