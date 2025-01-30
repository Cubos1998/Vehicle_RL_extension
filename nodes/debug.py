from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from version7_RL import DonkeyCarConfig, CustomDonkeyEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
import torch


# Initialize configuration
#config = DonkeyCarConfig()

# Set random seed for reproducibility
#seed = config.env_config["conf"]["random_seed"]
#np.random.seed(seed)
#random.seed(seed)
#torch.manual_seed(seed)

# Create the environment
#def make_env():
#    """
#    Creates and returns a wrapped DonkeyCar environment.
#    """
#    env = CustomDonkeyEnv(level=config.env_list[1], conf=config.env_config)
#    env = Monitor(env)  # Wrap with Monitor to track episode statistics
#    return env

#env = DummyVecEnv([make_env])  # Vectorized environment

#root = "Model_try_1_normalized"
root = "try2"

pth_file_path = f'./final_models/{root}/pytorch_variables.pth'

variab = torch.load(pth_file_path, map_location='cpu') 
print("keys:  ",variab.keys())
model_path = f"./final_models/{root}.zip"
SAC.load(model_path)
print("success")