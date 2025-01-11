#!/usr/bin/env python3
"""
Reset Deep Ensemble (RDE) for Atari (Freeway-v4)
------------------------------------------------
- N=2 ensemble DQN agents
- Round-robin reset
- Softmax-based action composition from the "oldest" agent's Q-values
- TensorBoard logging
- Model saving/loading for evaluation

Usage:
1) Train (default):
   python rde_freeway.py

2) Evaluate:
   python rde_freeway.py --evaluate
   (loads the ensemble from "models/rde_ensemble_Freeway-v4_seed42_rr4.pth"
    and runs e.g. 5 episodes at epsilon=0.05)
"""

import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import argparse

import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
ENV_ID             = "Freeway-v4"
SEED               = 42

# Ensemble
N_ENSEMBLE         = 2         # RDE typically uses 2 ensemble agents
RESET_FREQUENCY    = 40_000    # round-robin reset every 40k steps for Freeway
RESET_DEPTH        = "last2"   # partial reset ("full", "last2", or "last1")
SOFTMAX_BETA       = 50        # temperature for action composition

# DQN basics
LR                 = 1e-4
GAMMA              = 0.99
BATCH_SIZE         = 32
REPLAY_BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE    = 10_000
TOTAL_TIMESTEPS    = 100_000

TRAIN_FREQUENCY    = 1         # 1 training call per env step
UPDATES_PER_STEP   = 1         # replay ratio => 4
TARGET_UPDATE_FREQ = 1         # update target net after each training step
MAX_GRAD_NORM      = 10

# Epsilon schedule
EPS_START          = 1.0
EPS_END            = 0.01
EPS_DECAY_STEPS    = 10_000    # from 1 -> 0.01 over 10k steps

# Logging
LOG_INTERVAL       = 1_000

# Model/log paths
MODEL_PATH         = f"models/rde_ensemble_{ENV_ID}_seed{SEED}_rr{UPDATES_PER_STEP}.pth"
LOG_DIR            = f"runs/rde_ensemble_{ENV_ID}_seed{SEED}_rr{UPDATES_PER_STEP}"


# QNetwork
class QNetworkAtari(nn.Module):
    """
    3-layer conv net + 1 FC for Atari. 
    We allow partial or full re-init via reset_parameters(...).
    """
    def __init__(self, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        feats = self.features(x)
        feats = feats.reshape(feats.size(0), -1)
        out = self.fc(feats)
        return out

    def reset_parameters(self, reset_depth="full"):
        """
        Re-init some or all layers:
         'full' => re-init conv + fc
         'last2' => re-init last 2 layers in self.fc
         'last1' => re-init only final linear layer
        """
        def _init_layer(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if reset_depth == "full":
            self.apply(_init_layer)

        elif reset_depth == "last2":
            if len(self.fc) == 3:  # [Linear -> ReLU -> Linear]
                _init_layer(self.fc[-1])  # final linear
                _init_layer(self.fc[-3])  # linear before ReLU
            else:
                raise ValueError("Unexpected architecture for partial reset (last2).")

        elif reset_depth == "last1":
            if len(self.fc) == 3:
                _init_layer(self.fc[-1])
            else:
                raise ValueError("Unexpected architecture for partial reset (last1).")

        else:
            raise ValueError("Unknown reset depth option.")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        obs      = np.stack(obs)
        acts     = np.array(acts, dtype=np.int64)
        rews     = np.array(rews, dtype=np.float32)
        next_obs = np.stack(next_obs)
        dones    = np.array(dones, dtype=np.float32)
        return obs, acts, rews, next_obs, dones

    def __len__(self):
        return len(self.buffer)


# RDE Agent (Ensemble, round-robin reset, adaptive composition)
class RDEAgent:
    def __init__(self,
                 n_actions,
                 n_ensemble=N_ENSEMBLE,
                 reset_depth=RESET_DEPTH,
                 softmax_beta=SOFTMAX_BETA,
                 writer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.n_ensemble   = n_ensemble
        self.gamma        = GAMMA
        self.reset_depth  = reset_depth
        self.softmax_beta = softmax_beta
        self.writer       = writer

        self.q_networks     = []
        self.target_networks= []
        self.optimizers     = []

        for _ in range(n_ensemble):
            qnet = QNetworkAtari(n_actions).to(self.device)
            tnet = QNetworkAtari(n_actions).to(self.device)
            tnet.load_state_dict(qnet.state_dict())

            optimizer = optim.Adam(qnet.parameters(), lr=LR)
            self.q_networks.append(qnet)
            self.target_networks.append(tnet)
            self.optimizers.append(optimizer)

        self.n_actions = n_actions

        self.global_step     = 0  # env steps
        self.episodes        = 0
        self.train_steps     = 0  # training steps
        self.last_reset_idx  = 0  # which agent to reset next
        self.oldest_agent_idx= 1 % self.n_ensemble  # "oldest" after first reset

    def select_evaluation_action(self, obs_np, epsilon=0.05):
        """
        Epsilon-greedy selection that returns the best action
        across all ensemble networks based on the highest Q-value.
        """
        # Epsilon-greedy: take random action with probability epsilon
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        # Preprocess observation: from (84,84,4) channels-last to channels-first
        obs_ch_first = np.transpose(obs_np, (2, 0, 1))
        obs_t = torch.from_numpy(obs_ch_first).unsqueeze(0).float().to(self.device)

        best_val = -float('inf')
        best_act = None

        # Evaluate each network's Q-values and select the highest scoring action
        with torch.no_grad():
            for qnet in self.q_networks:
                qvals = qnet(obs_t)
                val, act = qvals.max(dim=1)  # maximum Q-value and corresponding action
                if val.item() > best_val:
                    best_val = val.item()
                    best_act = act.item()

        return best_act

    def select_action(self, obs_np, epsilon=0.05):
        """
        Epsilon-greedy on top of the ensemble composition.
        Each agent picks argmax Q_i(s). Then we do a softmax
        weighting from the 'oldest' agent's Q-values on those actions.
        """
        if random.random() < epsilon:
            return random.randint(0, self.n_actions-1)

        # channels-last => channels-first => torch
        obs_ch_first = np.transpose(obs_np, (2,0,1))
        obs_t = torch.from_numpy(obs_ch_first).unsqueeze(0).float().to(self.device)

        # gather each agent's argmax
        candidate_actions = []
        with torch.no_grad():
            for qnet in self.q_networks:
                qvals = qnet(obs_t)
                act_i = qvals.argmax(dim=1).item()
                candidate_actions.append(act_i)

        # Use Q-values from 'oldest' agent
        oldest = self.oldest_agent_idx
        with torch.no_grad():
            qvals_oldest = self.q_networks[oldest](obs_t).squeeze(0)

        # for each agent's chosen action, get Q_oldest(s, a_i)
        r_values = []
        for act in candidate_actions:
            r_values.append(qvals_oldest[act].item())

        # softmax
        max_r = max(abs(r) for r in r_values) if r_values else 1.0
        if max_r == 0:
            max_r = 1.0
        scaled_r = [(val / max_r)*self.softmax_beta for val in r_values]
        exp_r = np.exp(scaled_r)
        sum_exp = np.sum(exp_r)
        if sum_exp < 1e-9:
            probs = np.ones(self.n_ensemble) / self.n_ensemble
        else:
            probs = exp_r / sum_exp

        chosen_idx = np.random.choice(self.n_ensemble, p=probs)
        return candidate_actions[chosen_idx]

    def step_env(self, vec_env, replay_buffer, epsilon=0.05, episode_count = 0):
        obs_np = vec_env.current_obs  # shape [n_envs,84,84,4]
        n_envs = obs_np.shape[0]

        actions = []
        for i in range(n_envs):
            a = self.select_action(obs_np[i], epsilon)
            actions.append(a)

        next_obs, rewards, dones, infos = vec_env.step(actions)
        self.global_step += n_envs

        # Store
        for i in range(n_envs):
            replay_buffer.add(
                np.transpose(obs_np[i], (2,0,1)),
                actions[i],
                rewards[i],
                np.transpose(next_obs[i], (2,0,1)),
                bool(dones[i])
            )

        # Round-robin reset if needed
        if (self.global_step>0) and (self.global_step % RESET_FREQUENCY == 0):
            self.writer.add_scalar("ResetAgentIdx", self.last_reset_idx, episode_count)
            print(f"[INFO] Round-robin reset: agent {self.last_reset_idx} at step={self.global_step}")
            self.reset_agent(self.last_reset_idx)
            self.oldest_agent_idx = (self.last_reset_idx + 1) % self.n_ensemble
            self.last_reset_idx   = (self.last_reset_idx + 1) % self.n_ensemble

        # return average reward, done-any
        return float(np.mean(rewards)), any(dones)

    def train_on_batch(self, replay_buffer):
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            return

        obs, acts, rews, next_obs, dones = replay_buffer.sample(BATCH_SIZE)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        acts_t     = torch.LongTensor(acts).unsqueeze(1).to(self.device)
        rews_t     = torch.FloatTensor(rews).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        for i in range(self.n_ensemble):
            with torch.no_grad():
                q_next = self.target_networks[i](next_obs_t)
                max_q_next = q_next.max(dim=1, keepdim=True)[0]
                target = rews_t + self.gamma*(1 - dones_t)*max_q_next

            current_q = self.q_networks[i](obs_t).gather(1, acts_t)
            loss = nn.SmoothL1Loss()(current_q, target)

            self.optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), MAX_GRAD_NORM)
            self.optimizers[i].step()

        self.train_steps += 1

    def update_target_nets(self):
        for i in range(self.n_ensemble):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())

    def reset_agent(self, idx):
        """Reset Q-network parameters for agent idx, then copy to target net."""
        print(f"Resetting agent {idx} with depth={self.reset_depth}")
        self.q_networks[idx].reset_parameters(self.reset_depth)
        self.target_networks[idx].load_state_dict(self.q_networks[idx].state_dict())

    def save_ensemble(self, path=MODEL_PATH):
        """Save each Q-network's state_dict to a single file as a list."""
        ensemble_states = []
        for qnet in self.q_networks:
            ensemble_states.append(qnet.state_dict())
        torch.save(ensemble_states, path)
        print(f"Ensemble saved to {path}")

    def load_ensemble(self, path=MODEL_PATH):
        """Load from the saved list of Q-network state_dicts."""
        ensemble_states = torch.load(path, map_location=self.device, weights_only= True)
        if len(ensemble_states) != self.n_ensemble:
            raise ValueError("Ensemble checkpoint mismatch: # of states != # of agents.")
        for i in range(self.n_ensemble):
            self.q_networks[i].load_state_dict(ensemble_states[i])
            self.target_networks[i].load_state_dict(ensemble_states[i])
        print(f"Ensemble loaded from {path}")

# Build single VecEnv
def make_vec_env(env_id=ENV_ID, seed=SEED):
    venv = make_atari_env(env_id, n_envs=1, seed=seed)
    venv = VecFrameStack(venv, n_stack=4)
    venv.current_obs = None
    return venv

# Main Training (RDE)
def run_training():
    writer = SummaryWriter(log_dir=LOG_DIR)
    env = make_vec_env(ENV_ID, seed=SEED)
    n_actions = env.action_space.n

    agent = RDEAgent(
        n_actions=n_actions,
        n_ensemble=N_ENSEMBLE,
        reset_depth=RESET_DEPTH,
        softmax_beta=SOFTMAX_BETA,
        writer=writer
    )

    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    obs = env.reset()
    env.current_obs = obs

    episode_reward = 0.0
    episode_count  = 0
    rewards_history= []

    print(f"Starting RDE on {ENV_ID}, total={TOTAL_TIMESTEPS}, resets every {RESET_FREQUENCY}, depth={RESET_DEPTH}...")

    while agent.global_step < TOTAL_TIMESTEPS:
        # epsilon
        fraction = min(1.0, agent.global_step / EPS_DECAY_STEPS)
        epsilon  = EPS_START + fraction*(EPS_END - EPS_START)
        epsilon  = max(EPS_END, epsilon)

        r, done = agent.step_env(env, replay_buffer, epsilon=epsilon, episode_count = episode_count)
        episode_reward += r

        if done:
            episode_count += 1
            rewards_history.append(episode_reward)
            writer.add_scalar("EpisodeReward", episode_reward, episode_count)

            obs = env.reset()
            env.current_obs = obs
            episode_reward = 0.0
        else:
            env.current_obs = env.current_obs  # no change needed

        # Train
        if (agent.global_step % TRAIN_FREQUENCY) == 0:
            for _ in range(UPDATES_PER_STEP):
                agent.train_on_batch(replay_buffer)
                if agent.train_steps>0 and (agent.train_steps % TARGET_UPDATE_FREQ)==0:
                    agent.update_target_nets()

        # Log epsilon
        writer.add_scalar("Epsilon", epsilon, agent.global_step)

        # Logging
        if (agent.global_step % LOG_INTERVAL)==0 and agent.global_step>0:
            last_10 = (np.mean(rewards_history[-10:])
                       if len(rewards_history)>=10
                       else np.mean(rewards_history)
                          if len(rewards_history)>0 else 0)
            print(f"Step={agent.global_step} | Ep={episode_count} | "
                  f"AvgRew(last10)={last_10:.2f} | Eps={epsilon:.3f}")

    env.close()
    writer.close()

    agent.save_ensemble(MODEL_PATH)
    final_10 = np.mean(rewards_history[-10:]) if len(rewards_history)>=10 else 0
    print(f"Training done. Final 10-ep avg reward: {final_10:.2f}")

# Evaluation
def evaluate_model(model_path=MODEL_PATH, episodes=5, eval_epsilon=0.05):
    random_seed = random.randint(0, 2**32 - 1)
    print(f"Using random seed: {random_seed}")
    env = make_vec_env(ENV_ID, seed=random_seed)  # different seed if desired
    n_actions = env.action_space.n

    agent = RDEAgent(
        n_actions=n_actions,
        n_ensemble=N_ENSEMBLE,
        reset_depth=RESET_DEPTH,
        softmax_beta=SOFTMAX_BETA,
        writer=None
    )
    agent.load_ensemble(model_path)

    episode_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        env.current_obs = obs

        done = False
        ep_rew = 0.0
        while not done:
            obs_np = env.current_obs
            action = agent.select_evaluation_action(obs_np[0], epsilon=eval_epsilon)
            # action = agent.select_action(obs_np[0], epsilon=eval_epsilon)
            next_obs, reward, done_bool, info = env.step([action])
            ep_rew += float(reward[0])
            env.current_obs = next_obs

            done = bool(done_bool[0])

        episode_rewards.append(ep_rew)

    env.close()
    avg_rew = np.mean(episode_rewards)
    print(f"Evaluation over {episodes} episodes at eps={eval_epsilon}: average reward={avg_rew:.2f}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved RDE ensemble.")
    parser.add_argument("--episodes", type=int, default=5, help="# of evaluation episodes.")
    parser.add_argument("--eval_epsilon", type=float, default=0.05, help="Epsilon during evaluation.")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(model_path=MODEL_PATH, episodes=args.episodes, eval_epsilon=args.eval_epsilon)
    else:
        run_training()
