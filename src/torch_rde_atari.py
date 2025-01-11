#!/usr/bin/env python3
"""
A Reset Deep Ensemble (RDE) script for the Atari-100k benchmark,
using the hyperparameters from the paper's Section 4 / Appendix B.

Specifically for AlienNoFrameskip-v4:
- 100k environment steps total
- Reset interval: 8e4 (80,000)
- Reset depth: "last1"
- Replay buffer size: 1e5
- Min replay size: 1e4
- Batch size: 32
- Target net update period: 1 (i.e., every training update)
- Max gradient norm: 10
- Softmax β = 50
- Possibly do 4 updates per environment step (replay ratio = 4).
"""

import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque

import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

#   Hyperparameters (referenced from the paper)
ENV_ID              = "AlienNoFrameskip-v4"
N_ENSEMBLE          = 2           # # of ensemble agents
N_ENVS              = 1           # Single env to can be used to keep it truly at N time steps
LR                  = 1e-4        # Learning rate
GAMMA               = 0.99        # Discount factor
BATCH_SIZE          = 32          # Batch size
REPLAY_BUFFER_SIZE  = 100_000     # Replay buffer size (100 000)
MIN_REPLAY_SIZE     = 10_000      # Mininum replay buffer size
TOTAL_TIMESTEPS     = 100_000     # Total time steps
TRAIN_FREQUENCY     = 1           # train every environment step
UPDATES_PER_STEP    = 4           # replay ratio (1,2,4) => pick 4
TARGET_UPDATE_FREQ  = 1           # update target net every training step

RESET_FREQUENCY     = 80_000      # 8e4 for Alien
RESET_DEPTH         = "last1"     # "last1" as per the table for Alien
SOFTMAX_BETA        = 50          # beta for action selection

EPS_START           = 1.0
EPS_END             = 0.01
EPS_DECAY_FRAC      = 0.10        # decay epsilon over 10% of total steps => 10k
LOG_INTERVAL        = 10_000

MAX_GRAD_NORM       = 10          # paper indicates max grad norm = 10

# Register the Atari environments (ALE) as in the code snippet
gym.register_envs(ale_py)

#   CNN Q-Network for Atari (3 conv layers + 1 FC)
class QNetworkAtari(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # From paper: 3 conv layers [32,64,64], kernels [8x8,4x4,3x3], strides [4,2,1].
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # 7x7 is typical after [8,4,3] filters/strides
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.n_actions = n_actions
        self.reset_parameters("full")

    def forward(self, x):
        # x shape: [batch, 4, 84, 84]
        feats = self.features(x)
        feats = feats.contiguous().view(feats.size(0), -1)
        out = self.fc(feats)
        return out

    def reset_parameters(self, reset_depth="full"):
        """Reset some or all of the network parameters."""
        def _init_module(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if reset_depth == "full":
            self.apply(_init_module)
        elif reset_depth == "last2":
            # Re-init the last 2 layers in self.fc
            if len(self.fc) == 3:
                _init_module(self.fc[-1])
                _init_module(self.fc[-3])
            else:
                raise ValueError("Unexpected architecture for partial reset.")
        elif reset_depth == "last1":
            # Re-init ONLY the final linear layer in self.fc
            if len(self.fc) == 3:
                _init_module(self.fc[-1])
            else:
                raise ValueError("Unexpected architecture for partial reset.")
        else:
            raise ValueError("Unknown reset depth option.")

#   Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)

        obs       = np.stack(obs)        # shape: [B, 4,84,84]
        acts      = np.array(acts, dtype=np.int64)
        rews      = np.array(rews, dtype=np.float32)
        next_obs  = np.stack(next_obs)
        dones     = np.array(dones, dtype=np.float32)
        return obs, acts, rews, next_obs, dones

    def __len__(self):
        return len(self.buffer)

#   Ensemble DQN Agent
class EnsembleDQNAgent:
    def __init__(self, n_actions,
                 n_ensemble=N_ENSEMBLE, lr=LR, gamma=GAMMA,
                 reset_freq=RESET_FREQUENCY, reset_depth=RESET_DEPTH,
                 softmax_beta=SOFTMAX_BETA):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.n_ensemble = n_ensemble
        self.gamma = gamma
        self.reset_freq = reset_freq
        self.reset_depth = reset_depth
        self.softmax_beta = softmax_beta

        self.q_networks = []
        self.target_networks = []
        self.optimizers = []

        for _ in range(n_ensemble):
            qnet = QNetworkAtari(n_actions).to(self.device)
            tnet = QNetworkAtari(n_actions).to(self.device)
            tnet.load_state_dict(qnet.state_dict())

            optimizer = optim.Adam(qnet.parameters(), lr=lr)
            self.q_networks.append(qnet)
            self.target_networks.append(tnet)
            self.optimizers.append(optimizer)

        self.global_step = 0

        # Round-robin
        self.last_reset_idx = 0
        # The 'oldest' agent is next after the just-reset agent
        self.oldest_agent_idx = (self.last_reset_idx + 1) % self.n_ensemble

        self.n_actions = n_actions

    def select_action(self, obs_np, epsilon=0.05):
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        # obs_np shape: (84,84,4) => transpose to (4,84,84)
        obs_ch_first = np.transpose(obs_np, (2,0,1))
        obs_t = torch.from_numpy(obs_ch_first).unsqueeze(0).float().to(self.device)

        # Each agent picks argmax
        candidate_actions = []
        with torch.no_grad():
            for qnet in self.q_networks:
                qvals = qnet(obs_t)
                a = qvals.argmax(dim=1).item()
                candidate_actions.append(a)

        # Use oldest agent's Q-values to compute softmax distribution
        oldest = self.oldest_agent_idx
        with torch.no_grad():
            qvals_oldest = self.q_networks[oldest](obs_t).squeeze(0)

        # For each agent's chosen action, get Q_oldest(s, a_i)
        r_values = []
        for act in candidate_actions:
            r_values.append(qvals_oldest[act].item())

        # Scale for stable softmax
        max_r = max(abs(v) for v in r_values) if r_values else 1.0
        if max_r == 0:
            max_r = 1.0
        scaled_r = [(val / max_r) * self.softmax_beta for val in r_values]
        exp_r = np.exp(scaled_r)
        sum_exp = np.sum(exp_r)
        if sum_exp <= 1e-9:
            probs = np.ones(self.n_ensemble) / self.n_ensemble
        else:
            probs = exp_r / sum_exp

        chosen_agent_idx = np.random.choice(self.n_ensemble, p=probs)
        return candidate_actions[chosen_agent_idx]

    def reset_agent(self, idx):
        """Reset the parameters for agent idx."""
        self.q_networks[idx].reset_parameters(self.reset_depth)
        self.target_networks[idx].load_state_dict(
            self.q_networks[idx].state_dict()
        )

    def step_env(self, vec_env, replay_buffer, epsilon=0.05):
        """
        Interact with the single-env (n_envs=1) or multi-env. We store transitions.
        """
        obs_np = vec_env.current_obs  # shape: (N_ENVS, 84,84,4)
        n_envs = obs_np.shape[0]

        # Get actions from the ensemble
        actions = []
        for i in range(n_envs):
            single_obs = obs_np[i]  # (84,84,4)
            a = self.select_action(single_obs, epsilon=epsilon)
            actions.append(a)

        next_obs, rewards, dones, infos = vec_env.step(actions)
        self.global_step += n_envs

        # Store transitions
        for i in range(n_envs):
            single_obs    = obs_np[i]
            single_next   = next_obs[i]
            single_reward = rewards[i]
            done_bool     = bool(dones[i])

            # Transpose to channels-first before storing
            obs_ch_first     = np.transpose(single_obs,  (2,0,1))
            next_ch_first    = np.transpose(single_next, (2,0,1))

            replay_buffer.add(obs_ch_first, actions[i],
                              single_reward, next_ch_first,
                              done_bool)

        # Round-robin reset
        if (self.global_step % self.reset_freq) == 0:
            print(f"[INFO] Resetting agent {self.last_reset_idx} at step {self.global_step}")
            self.reset_agent(self.last_reset_idx)
            self.oldest_agent_idx = (self.last_reset_idx + 1) % self.n_ensemble
            self.last_reset_idx = (self.last_reset_idx + 1) % self.n_ensemble

        # Return aggregated reward, done if any env ended
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
                target = rews_t + self.gamma * (1 - dones_t) * max_q_next

            current_q = self.q_networks[i](obs_t).gather(1, acts_t)
            loss = nn.SmoothL1Loss()(current_q, target)

            self.optimizers[i].zero_grad()
            loss.backward()

            # Clip gradients per the paper (max grad norm=10)
            nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), MAX_GRAD_NORM)

            self.optimizers[i].step()

    def update_target_nets(self):
        """Here, we copy the online Q-net parameters into each target net."""
        for i in range(self.n_ensemble):
            self.target_networks[i].load_state_dict(
                self.q_networks[i].state_dict()
            )

# Build single VecEnv
def make_vec_env(env_id=ENV_ID, n_envs=N_ENVS, seed=0):
    # Gray-scaling, 84x84, 4 frame stacks, reward clipping, etc. are typical
    # for "make_atari_env". We assume that or we can provide wrapper_kwargs
    venv = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    # Frame stacking => shape: (n_envs, 84,84,4)
    venv = VecFrameStack(venv, n_stack=4)
    venv.current_obs = None
    return venv

# Main Training Loop
def run_training():
    env = make_vec_env(ENV_ID, N_ENVS)
    n_actions = env.action_space.n

    agent = EnsembleDQNAgent(n_actions=n_actions)
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    obs = env.reset()  # shape: (N_ENVS, 84,84,4)
    env.current_obs = obs

    episode_reward = 0.0
    episode_count = 0
    rewards_history = []

    train_steps = 0
    # We'll decay epsilon over the first 10% (EPS_DECAY_FRAC=0.1) of total timesteps => 10k steps
    eps_decay_steps = int(EPS_DECAY_FRAC * TOTAL_TIMESTEPS)

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps on {ENV_ID} with RDE...")

    while agent.global_step < TOTAL_TIMESTEPS:
        fraction = min(1.0, agent.global_step / eps_decay_steps)  # in [0,1]
        epsilon = EPS_START + fraction * (EPS_END - EPS_START)
        epsilon = max(EPS_END, epsilon)  # clamp

        r, done = agent.step_env(env, replay_buffer, epsilon=epsilon)
        episode_reward += r

        if done:
            episode_count += 1
            rewards_history.append(episode_reward)
            obs = env.reset()
            env.current_obs = obs
            episode_reward = 0.0

        # train multiple times => replay ratio=4
        if (agent.global_step % TRAIN_FREQUENCY) == 0:
            for _ in range(UPDATES_PER_STEP):
                agent.train_on_batch(replay_buffer)
                train_steps += 1

        # Because the table says "target update period=1",
        # we do it every training step
        if train_steps > 0 and (train_steps % TARGET_UPDATE_FREQ) == 0:
            agent.update_target_nets()

        # Logging
        if (agent.global_step % LOG_INTERVAL) == 0 and agent.global_step > 0:
            last_10 = np.mean(rewards_history[-10:]) if len(rewards_history)>=10 else np.mean(rewards_history)
            print(f"Step={agent.global_step} | Episodes={episode_count} | "
                  f"AvgRew(last10)={last_10:.2f} | Sum Reward={sum(rewards_history)} | Eps={epsilon:.3f}")

    env.close()
    final_10 = np.mean(rewards_history[-10:]) if len(rewards_history)>=10 else 0.0
    print("Training done. Final 10-episode average reward:", final_10)

if __name__ == "__main__":
    start_time = time.time()
    run_training()
    elapsed = time.time() - start_time
    print(f"Finished. Elapsed time: {elapsed:.2f}s")
