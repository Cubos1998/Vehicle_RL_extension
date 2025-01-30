# Script for testong the Policies:

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Standard Libraries
import argparse
import random
import uuid
import time

# Third-Party Libraries
import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter       

# Local Modules
import gym_donkeycar
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from version7_RL import DonkeyCarConfig, CustomDonkeyEnv

import time
from stable_baselines3.common.vec_env import VecEnv

def evaluate_policy_with_delay(model, env: VecEnv, n_eval_episodes: int = 10, delay: float = 1.0):
    """
    Evaluate the policy with a custom delay between decisions.

    :param model: The trained model to evaluate.
    :param env: The environment to evaluate on (should be VecEnv or compatible).
    :param n_eval_episodes: Number of episodes to evaluate.
    :param delay: Delay in seconds between each decision.
    :return: Mean reward per episode and standard deviation of rewards.
    """
    episode_rewards = []
    now = None
    time_t = 0

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Predict the action using the model
            action, _states = model.predict(obs, deterministic=True)
            # Take a step in the environment
            obs, reward, done, info = env.step(action)
            # Accumulate the reward
            episode_reward += reward
            # Render the environment (optional)
            env.render()
            # Wait for the specified delay
            time.sleep(delay)
            #if now is not None:
            #    time_t = now - time.time()
            #now = time.time()
            #print(time_t)
        episode_rewards.append(episode_reward)

    mean_reward = sum(episode_rewards) / n_eval_episodes
    std_reward = (sum([(x - mean_reward)**2 for x in episode_rewards]) / n_eval_episodes) ** 0.5
    return mean_reward, std_reward


def main():
    """
    Main function to initialize the environment, model, callbacks, and testing.
    """
    # Initialize configuration
    config = DonkeyCarConfig()
    
    # Set random seed for reproducibility
    seed = config.env_config["conf"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the environment
    def make_env():
        """
        Creates and returns a wrapped DonkeyCar environment.
        """
        env = CustomDonkeyEnv(level=config.env_list[1], conf=config.env_config)
        env = Monitor(env)  # Wrap with Monitor to track episode statistics
        return env
    
    env = DummyVecEnv([make_env])  # Vectorized environment
    
    # Check if a checkpoint exists
    #env = VecNormalize.load("nodes/vecnormalize_53.pkl", env)

    # Load the trained model
    #model_path = "./final_models/sac_donkeycar_200000_steps.zip"
    #model_path2 = "./final_models/Model_try_1_normalized.zip"
    model_path = "./final_models/try2.zip"

    ################

    #check2 = torch.load(model_path2, map_location=torch.device('cpu'))
    #check = torch.load(model_path, map_location=torch.device('cpu'))


    #print("Does nor work:   ", check2.keys())
    #print("Works:  ", check.keys())
    

    ################


    model = SAC.load(model_path)
    print(f"Successfully loaded model from checkpoint: {model_path}")

    # Evaluate the policy
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    #print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Test the model in the environment     

    mean_reward, std_reward = evaluate_policy_with_delay(model, env, n_eval_episodes=10, delay=0.09)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Close the environment
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
