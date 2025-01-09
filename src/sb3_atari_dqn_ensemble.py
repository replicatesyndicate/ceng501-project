import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update, get_linear_fn
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.spaces import Box

gym.register_envs(ale_py)

# TensorBoard setup
writer = SummaryWriter("./logs")

# Hyperparameters
N_ENSEMBLE = 2  # Number of ensemble agents
RESET_FREQUENCY = 40000  # Reset frequency in timesteps
BETA = 50  # Action selection coefficient
REPLAY_BUFFER_SIZE = 100000  # Replay buffer size
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_STARTS = 2000  # Timesteps before training starts
TAU = 0.005  # Polyak update coefficient
TOTAL_TIMESTEPS = int(1e5)  # Total training timesteps
TRAIN_FREQ = 1  # Frequency of training (steps)
GRADIENT_STEPS = 1  # Gradient steps per update
TARGET_UPDATE_INTERVAL = 1  # Update target networks every step
n_stack = 4  # Number of stacked frames

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create and preprocess Atari environment
env = make_atari_env("AlienNoFrameskip-v4", n_envs=4, seed=42)
env = VecFrameStack(env, n_stack=n_stack,channels_order='last') 

eval_env = make_atari_env("AlienNoFrameskip-v4", n_envs=4, seed=84)
eval_env = VecFrameStack(eval_env, n_stack=n_stack, channels_order='last')  # Use channel-first order

# Fix observation space for CnnPolicy
channel_first_shape = (n_stack, env.observation_space.shape[0], env.observation_space.shape[1])
env.observation_space = Box(low=0, high=255, shape=channel_first_shape, dtype=np.uint8)
eval_env.observation_space = Box(low=0, high=255, shape=channel_first_shape, dtype=np.uint8)

# Debug observation space       
print(f"Corrected Observation Space: {env.observation_space.shape}")

# Replay buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, env.observation_space, env.action_space, device=device, n_envs=4)

# Ensemble setup
ensemble_agents = [
    CnnPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=get_linear_fn(1e-4, 1e-5, 1.0),
        net_arch=[256, 256],
    ).to(device)
    for _ in range(N_ENSEMBLE)
]
target_networks = [
    CnnPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=get_linear_fn(1e-4, 1e-5, 1.0),
        net_arch=[256, 256],
    ).to(device)
    for _ in range(N_ENSEMBLE)
]
optimizers = [optim.Adam(agent.parameters(), lr=1e-4) for agent in ensemble_agents]

# Helper functions
def reset_agent(agent):
    for layer in agent.q_net.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def adaptive_action_selection(q_values, beta):
    q_values_normalized = q_values / (q_values.max(dim=-1, keepdim=True)[0] + 1e-8)
    scaled_q_values = beta * torch.max(q_values_normalized, torch.zeros_like(q_values_normalized))
    summed_q_values = scaled_q_values.sum(dim=0)
    action_distributions = torch.softmax(summed_q_values, dim=-1).detach().cpu().numpy()
    return np.array([np.random.choice(len(distribution), p=distribution) for distribution in action_distributions])

# Training loop
state = env.reset()
current_agent_index = 0
step_count = 0

for step in range(TOTAL_TIMESTEPS):
    state_tensor = torch.tensor(state, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    q_values = torch.stack([agent.q_net(state_tensor) for agent in ensemble_agents])

    action = adaptive_action_selection(q_values, BETA)
    next_state, reward, done, info = env.step(action)

    replay_buffer.add(state, next_state, action, reward, done, info)
    state = next_state

    if replay_buffer.size() > BATCH_SIZE and step > LEARNING_STARTS:
        batch = replay_buffer.sample(BATCH_SIZE)
        observations = batch.observations.permute(0, 3, 1, 2).to(device)
        next_observations = batch.next_observations.permute(0, 3, 1, 2).to(device)

        for i, agent in enumerate(ensemble_agents):
            q_values = agent.q_net(observations).gather(1, batch.actions.to(device))
            with torch.no_grad():
                target_q_values = target_networks[i].q_net(next_observations).max(1, keepdim=True)[0]
                target = batch.rewards.to(device) + GAMMA * (1 - batch.dones.to(device)) * target_q_values

            loss = F.smooth_l1_loss(q_values, target)
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

    if step % TARGET_UPDATE_INTERVAL == 0:
        for i in range(N_ENSEMBLE):
            polyak_update(ensemble_agents[i].q_net.parameters(), target_networks[i].q_net.parameters(), TAU)

    if step % RESET_FREQUENCY == 0:
        reset_agent(ensemble_agents[current_agent_index])
        current_agent_index = (current_agent_index + 1) % N_ENSEMBLE

    if np.any(done):
        state = env.reset()

    if step % 1000 == 0:
        print(f"Step: {step}, Average Reward: {np.mean(reward)}")

writer.close()

# Evaluation loop
def evaluate(agents, env, n_eval_episodes=10):
    total_rewards = []
    for _ in range(n_eval_episodes):
        state = env.reset()
        episode_reward = np.zeros(env.num_envs)
        done = np.zeros(env.num_envs, dtype=bool)

        while not np.all(done):
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            q_values = torch.stack([agent.q_net(state_tensor) for agent in agents])
            action = torch.argmax(q_values.mean(dim=0), dim=1).cpu().numpy()

            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(np.mean(episode_reward))

    avg_reward = np.mean(total_rewards)
    print(f"Evaluation Results: Mean Reward = {avg_reward}")
    return avg_reward

# Perform evaluation
evaluate(ensemble_agents, eval_env)

# Close environments
env.close()
eval_env.close()