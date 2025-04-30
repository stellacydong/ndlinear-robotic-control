# train.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from models.baseline_policy import BaselinePolicy
from models.ndlinear_policy import NdLinearPolicy


def select_action(policy, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = policy(state)
    return action.squeeze(0).detach().numpy()


def run_episode(env, policy, gamma=0.99):
    states, actions, rewards = [], [], []
    state = env.reset()[0]
    done = False

    while not done:
        action = select_action(policy, state)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(clipped_action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        done = terminated or truncated

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
    return states, actions, returns


def train(policy_class, env_name="Pendulum-v1", episodes=500, lr=1e-3, rank=32):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if policy_class.__name__ == "NdLinearPolicy":
        policy = policy_class(obs_dim, action_dim, rank=rank)
    else:
        policy = policy_class(obs_dim, action_dim)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    print(f"Training {policy_class.__name__} on {env_name}...")

    for ep in range(episodes):
        states, actions, returns = run_episode(env, policy)

        policy.train()
        optimizer.zero_grad()

        loss = 0
        for state, action, G in zip(states, actions, returns):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.float32)

            predicted_action = policy(state).squeeze(0)
            loss += ((predicted_action - action)**2).mean() * G  # MSE scaled by return

        loss.backward()
        optimizer.step()

        if ep % 25 == 0:
            print(f"Episode {ep}, Loss: {loss.item():.4f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "ndlinear"], default="baseline")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    if args.model == "baseline":
        train(BaselinePolicy, episodes=args.episodes, lr=args.lr)
    else:
        train(NdLinearPolicy, episodes=args.episodes, lr=args.lr, rank=args.rank)

