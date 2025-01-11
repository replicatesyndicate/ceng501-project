#!/usr/bin/env python3
"""
rde_minigrid.py

Demonstration of Reset Deep Ensemble (RDE) on MiniGrid tasks, following the
hyperparameters in the paper's Section 4.1 / Appendix B for MiniGrid:

 - DQN with a 5-layer MLP
 - RMSProp with lr = 0.0001
 - Replay buffer size = 5e5
 - Batch size = 256
 - Epsilon decays 0.9 -> 0.05 over 1e5 steps
 - Target network update every 1e3 training steps 
 - Reset interval in gradient steps: 
    - 2e5 for (GoToDoor, LavaCrossing, LavaGap),
    - 1e5 for (FourRooms, SimpleCrossing)
 - Beta (softmax coefficient) = 50

We choose by default FourRooms and a reset interval of 1e5 gradient steps.
Change ENV_ID and RESET_INTERVAL_GRAD_STEPS to match your exact task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

# Make sure you install gym-minigrid:
import gymnasium as gym
import minigrid

# Hyperparameters (per the paper's table for MiniGrid)
ENV_ID                   = "MiniGrid-Empty-Random-5x5-v0"  # or "MiniGrid-GoToDoor-8x8-v0", etc. 
#ENV_ID                   = "MiniGrid-FourRooms-v0"  # or "MiniGrid-GoToDoor-8x8-v0", etc. 
N_ENSEMBLE              = 2         # Number of ensemble agents
LR                      = 1e-4       # Learning rate for RMSProp
GAMMA                   = 0.99
REPLAY_BUFFER_SIZE      = 500_000    # 5e5
BATCH_SIZE              = 256
EPS_START               = 0.9
EPS_END                 = 0.05
EPS_DECAY_STEPS         = 100_000    # decay from 0.9->0.05 over 1e5 steps
TARGET_UPDATE_FREQ      = 1000       # gradient steps (not environment steps)
BETA_SOFTMAX            = 50         # coefficient for action selection
MAX_GRAD_STEPS          = 500_000    # total gradient steps? or environment steps?

# The paper says "The maxmimum number of steps = 100" is the max steps per episode
# We'll treat that as env._max_episode_steps = 100 if you want. We'll do so below if needed.

# Reset interval in gradient steps:
# For FourRooms, we do 1e5 (the paper's table).
RESET_INTERVAL_GRAD_STEPS = 100_000  # 1e5 for FourRooms, SimpleCrossingS9N1
# If you are using GoToDoor, LavaCrossing, or LavaGap, set it to 2e5.

# We'll allow "max total environment steps" = some number to run. 
# The paper didn't specify an exact total, but we can e.g. do 1e6 environment steps or so.
# We'll define a local constant for environment steps:
TOTAL_ENV_STEPS = 1_000_000

# 5-layer MLP for DQN (MiniGrid)
# The paper states "For Minigrid, DQN employed 5 layers MLPs."
# We'll define a simple 5-layer feedforward net w/ ReLU.
class QNetworkMiniGrid(nn.Module):
    def _init_(self, obs_dim, n_actions):
        super()._init_()
        # We'll define a 5-layer MLP (including output).
        # For example: input -> 256 -> 256 -> 256 -> 256 -> output(n_actions)
        hidden_size = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        # If we want partial resets, we can define "reset_parameters" similarly.

    def forward(self, x):
        # x shape: [batch, obs_dim]
        return self.net(x)

# Replay Buffer
class ReplayBuffer:
    def _init_(self, capacity=500_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (np.array(obs, dtype=np.float32),
                np.array(acts, dtype=np.int64),
                np.array(rews, dtype=np.float32),
                np.array(next_obs, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def _len_(self):
        return len(self.buffer)

# Ensemble DQN Agent
class EnsembleDQNAgent:
    """
    Similar to the RDE logic in the other scripts. 
    We maintain N Q-networks, each has a target net, we do round-robin resets.
    We do a softmax-based ensemble action selection using the 'oldest' agent's Q-values.
    """
    def _init_(self, obs_dim, n_actions, n_ensemble=N_ENSEMBLE,
                 lr=LR, gamma=GAMMA, beta_softmax=BETA_SOFTMAX):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.n_ensemble = n_ensemble
        self.gamma = gamma
        self.beta_softmax = beta_softmax

        # Build N Q networks + target networks
        self.q_networks = []
        self.target_networks = []
        self.optimizers = []

        for _ in range(n_ensemble):
            qnet = QNetworkMiniGrid(obs_dim, n_actions).to(self.device)
            tnet = QNetworkMiniGrid(obs_dim, n_actions).to(self.device)
            tnet.load_state_dict(qnet.state_dict())

            # RMSProp with lr=1e-4, as stated
            optimizer = optim.RMSprop(qnet.parameters(), lr=lr)
            self.q_networks.append(qnet)
            self.target_networks.append(tnet)
            self.optimizers.append(optimizer)

        self.global_env_step = 0     # count environment steps
        self.global_grad_step = 0    # count gradient steps

        self.oldest_agent_idx = 1 % n_ensemble
        self.last_reset_idx = 0      # who to reset next

        self.obs_dim = obs_dim
        self.n_actions = n_actions

    def select_action(self, obs_np, epsilon=0.05):
        # Epsilon-greedy on top of the ensemble logic
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        # Convert obs_np -> torch
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)  # shape [1, obs_dim]

        # Each agent picks argmax
        candidate_actions = []
        with torch.no_grad():
            for qnet in self.q_networks:
                qvals = qnet(obs_t)          # shape [1, n_actions]
                a = qvals.argmax(dim=1).item()
                candidate_actions.append(a)

        # Now get the 'oldest' agent's Q-values
        oldest = self.oldest_agent_idx
        with torch.no_grad():
            qvals_oldest = self.q_networks[oldest](obs_t).squeeze(0)

        # For each agent-chosen action, get Q_oldest(s, a_i)
        r_values = []
        for act_i in candidate_actions:
            r_values.append(qvals_oldest[act_i].item())

        # Softmax
        max_r = max(abs(v) for v in r_values) if len(r_values) > 0 else 1.0
        if max_r == 0:
            max_r = 1.0
        scaled_r = [(val / max_r)*self.beta_softmax for val in r_values]
        exp_r = np.exp(scaled_r)
        sum_exp = np.sum(exp_r)
        if sum_exp <= 1e-9:
            probs = np.ones(self.n_ensemble)/self.n_ensemble
        else:
            probs = exp_r / sum_exp

        chosen_agent = np.random.choice(self.n_ensemble, p=probs)
        return candidate_actions[chosen_agent]

    def reset_agent(self, idx):
        """
        Reset entire DNN. The paper says "higher reset depth is beneficial for tasks
        requiring extensive exploration in Minigrid." 
        We'll just re-init the entire net for demonstration.
        """
        # Re-init the weights of q_networks[idx]
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        self.q_networks[idx].apply(_init_weights)

        # Also copy over to target net
        self.target_networks[idx].load_state_dict(self.q_networks[idx].state_dict())

    def step_env(self, env, replay_buffer, epsilon=0.05):
        # We'll assume a single environment (discrete).
        # For multi-env, you'd do a vectorized approach similarly.

        obs_np = env.current_obs  # we store obs in run_training
        action = self.select_action(obs_np, epsilon=epsilon)
        next_obs, reward, done, truncated, info = env.step(action)
        done_bool = done or truncated

        if isinstance(next_obs, dict):
            if 'image' in next_obs:
                next_obs = next_obs['image'].flatten()
            else:
                raise ValueError("Unknown MiniGrid obs structure in step_env!")

        replay_buffer.add(obs_np, action, reward, next_obs, done_bool)

        env.current_obs = next_obs
        self.global_env_step += 1
        return reward, done_bool

    def train_on_batch(self, replay_buffer):
        # One gradient update
        if len(replay_buffer) < BATCH_SIZE:
            return

        obs, acts, rews, next_obs, dones = replay_buffer.sample(BATCH_SIZE)

        obs_t      = torch.FloatTensor(obs).to(self.device)           # [B, obs_dim]
        acts_t     = torch.LongTensor(acts).unsqueeze(1).to(self.device)  # [B,1]
        rews_t     = torch.FloatTensor(rews).unsqueeze(1).to(self.device)  # [B,1]
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)       # [B, obs_dim]
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # [B,1]

        for i in range(self.n_ensemble):
            with torch.no_grad():
                next_q = self.target_networks[i](next_obs_t)
                max_next_q = next_q.max(dim=1, keepdim=True)[0]
                target = rews_t + self.gamma*(1-dones_t)*max_next_q

            current_q = self.q_networks[i](obs_t).gather(1, acts_t)
            loss = nn.MSELoss()(current_q, target)

            self.optimizers[i].zero_grad()
            loss.backward()
            # max grad norm = 10
            nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), 10)
            self.optimizers[i].step()

        self.global_grad_step += 1

        # Check if we need to reset an agent (round-robin)
        # The paper says the reset frequency is w.r.t gradient steps, e.g. 1e5
        if (self.global_grad_step % RESET_INTERVAL_GRAD_STEPS) == 0:
            # Reset the agent at index = self.last_reset_idx
            print(f"[INFO] Reset agent {self.last_reset_idx} at grad_step={self.global_grad_step}")
            self.reset_agent(self.last_reset_idx)
            self.oldest_agent_idx = (self.last_reset_idx + 1) % self.n_ensemble
            self.last_reset_idx = (self.last_reset_idx + 1) % self.n_ensemble

    def update_target_networks(self):
        """
        Update target nets for each ensemble member.
        Called every 1e3 gradient steps, per the paper.
        """
        for i in range(self.n_ensemble):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())

# Main training function
def run_training():
    # Create environment
#    env = gym.make(ENV_ID, render_mode="human")
    env = gym.make(ENV_ID, render_mode="human")

    # The paper mentions "the maximum number of steps = 100" in an episode,
    # so let's forcibly set that if the env supports _max_episode_steps
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = 10

    obs0, _ = env.reset()
    env.current_obs = obs0  # store current obs in env for convenience

    # Observations in MiniGrid might be (height,width,channels)? Or a dictionary?
    # Typically "MiniGrid-FourRooms-v0" returns a 2D "obs['image']" or something.
    # If it's a dict-based obs, you'd parse it. We'll assume 1D flatten for MLP.
    # We'll check the shape from env.reset().
    # For instance, if obs0 is a dict, we do obs0 = obs0['image'].flatten() or something.
    # We'll do a small check:
    if isinstance(obs0, dict):
        # Possibly 'image' shape?
        if 'image' in obs0:
            obs0 = obs0['image'].flatten()
            env.current_obs = obs0
        else:
            raise ValueError("Unknown MiniGrid obs structure.")
    else:
        # if it's already a numpy array, we flatten if needed
        obs0 = np.array(obs0).flatten()
        env.current_obs = obs0

    obs_dim = env.current_obs.shape[0]
    # Action space
    n_actions = env.action_space.n

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    # Create the RDE agent
    agent = EnsembleDQNAgent(obs_dim, n_actions)

    episode_reward = 0.0
    episode_count = 0
    rewards_history = []

    # We'll run up to e.g. TOTAL_ENV_STEPS or some number
    # The paper doesn't specify exactly how many env steps in total for minigrid,
    # but let's say 1 million steps for a demonstration. (Or maybe 200k if you like.)
    max_env_steps = TOTAL_ENV_STEPS

    train_steps_since_update = 0

    # For epsilon decay
    def get_epsilon(step):
        fraction = min(1.0, step / EPS_DECAY_STEPS)
        return max(EPS_END, EPS_START + fraction*(EPS_END - EPS_START))

    for s in range(int(max_env_steps)):
        if s % 5000 == 0:
            env.render()  # Show window every 100 steps

        epsilon = get_epsilon(agent.global_env_step)
        r, done = agent.step_env(env, replay_buffer, epsilon=epsilon)
        episode_reward += r

        # If done, reset
        if done:
            episode_count += 1
            rewards_history.append(episode_reward)
            obsN, _ = env.reset()
            if isinstance(obsN, dict):
                obsN = obsN['image'].flatten()
            else:
                obsN = np.array(obsN).flatten()
            env.current_obs = obsN
            episode_reward = 0.0

        # We'll train every single environment step:
        agent.train_on_batch(replay_buffer)

        # Periodically update target networks every 1e3 gradient steps
        if agent.global_grad_step > 0 and (agent.global_grad_step % TARGET_UPDATE_FREQ) == 0:
            agent.update_target_networks()

        # Logging occasionally
        if agent.global_env_step % 10000 == 0 and agent.global_env_step > 0:
            last_10_mean = np.mean(rewards_history[-10:]) if len(rewards_history)>=10 else np.mean(rewards_history)
            print(f"EnvStep={agent.global_env_step} | Episodes={episode_count} | "
                  f"AvgRew(last10)={last_10_mean:.2f} | Eps={epsilon:.3f}")

        if agent.global_env_step >= max_env_steps:
            break

    env.close()
    final_10 = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else 0.0
    print("Training done. Final 10-ep average reward:", final_10)

if _name_ == "_main_":
    start_t = time.time()
    run_training()
    elapsed = time.time() - start_t
    print(f"Finished. Elapsed time: {elapsed:.2f}s")