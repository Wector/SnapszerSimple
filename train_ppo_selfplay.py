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
ENT_COEF = 0.01
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

def evaluate_vs_random(agent, num_games=100):
    """Evaluates the agent against a random opponent."""
    wins = 0
    draws = 0
    losses = 0
    
    # Create a separate env for evaluation (single threaded is fine for 100 games)
    # We need an opponent policy that is purely random
    def random_opp(obs, legal):
        import random
        return random.choice(legal)
        
    env = SnapszerGymEnv(opponent_policy=random_opp)
    
    for _ in range(num_games):
        obs, info = env.reset()
        done = False
        while not done:
            # Agent's turn
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            legal = info["legal_actions"]
            mask = torch.zeros((1, TOTAL_ACTIONS)).to(device)
            for act in legal:
                mask[0, act] = 1.0
                
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor, legal_actions_mask=mask)
                
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                    
    return wins, draws, losses

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
            w, d, l = evaluate_vs_random(agent, num_games=100)
            win_rate = (w / (w + d + l)) * 100
            win_rate_str = f"{win_rate:.1f}%"
            
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
