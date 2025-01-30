#!/usr/bin/env python3

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from .version7_RL import CustomDonkeyEnv, DonkeyCarConfig
from stable_baselines3 import SAC

# Define the environment creation function
def make_env():
    config = DonkeyCarConfig()
    env = CustomDonkeyEnv(level=config.env_list[1], conf=config.env_config)
    env = Monitor(env)
    return env

# Recreate the vectorized environment
env = DummyVecEnv([make_env])

# Load normalization statistics
env = VecNormalize.load("vecnormalize.pkl", env)

# Initialize the SAC model
model = SAC.load("sac_donkeycar", env=env)

# Optionally, disable normalization for evaluation
env.training = False
env.norm_reward = False

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
