#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Implementation for Line Following in DonkeyCar Environment

This script sets up and trains an SAC agent to perform line-following tasks using the
DonkeyCar simulation environment. It includes custom image preprocessing, a tailored
CNN feature extractor, and callbacks for monitoring training progress and saving checkpoints.

Author: Cristian Cubides-Herrera
Date: 2024-11-23
"""

# ================================
# 1. Imports
# ================================

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

# ================================
# 2. Configuration
# ================================

class DonkeyCarConfig:
    """
    Configuration parameters for the DonkeyCar environment.
    """
    env_list = [
        "warehouse",
        "generated_road",
        "sparkfun_avc",
        "generated_track",
        "roboracingleague_1",
    ]
    
    env_config = {
        "exe_path": "/home/cubos98/Desktop/MA/sim/sim_vehicle.x86_64",
        "host": "127.0.0.1",
        "port": 9091,
        "conf": {
            "car_name": "SAC",
            "race": False,
            "racer_name": "SAC_agent",
            "country": "Germany",
            "bio": "Learning to drive with SAC",
            "guid": str(uuid.uuid4()),
            "random_seed": random.randint(0, 10000),
            "max_cte": 0.5,
            "frame_skip": 1,
            "cam_resolution": (240, 320, 4),
            "log_level": 20,
            # "steer_limit": 1.0,
            # "throttle_min": 0.0,
            # "throttle_max": 1.0,
        }
    }
    
    max_cte = env_config["conf"]["max_cte"]

# ================================
# 3. Preprocessing Functions
# ================================

def process_image(observation: np.ndarray) -> np.ndarray:

    # Convert RGBA to RGB if necessary
    if observation.shape[2] == 4:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

    # Convert RGB to YUV color space
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2YUV)

    # Resize to (80, 60)
    observation = cv2.resize(observation, (80, 60), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    #observation = observation / 255.0

    # Transpose to channel-first format
    #observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
    # Convert the image to HSV color space for better color segmentation
    hsv = cv2.cvtColor(observation , cv2.COLOR_BGR2HSV)

    # Define the yellow color range for masking
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where yellow is detected
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Calculate moments of the binary image to find the center of the yellow line
    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        # Calculate the center of the yellow line
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Calculate the steering angle based on the position of the yellow line
        # Center of the image frame
        height, width, _ = observation .shape
        center_x = width // 2

        # Error is the difference between the line position and the center of the image
        error_x = cx - center_x + 1

        # Calculate the steering angle (proportional control). Steereng gain is the only learnable parameter.
        print("error_x: ", error_x)


        return error_x, False
    else:
        # If no yellow line is detected, return a zero steering angle
        return error_x, False


# ================================
# 4. Custom Environment
# ================================

class CustomDonkeyEnv(DonkeyEnv):
    """
    Custom DonkeyCar environment with image preprocessing and tailored reward function for line tracking.
    """
    def __init__(self, level: str, conf: dict):
        """
        Initializes the custom environment.

        Args:
            level (str): The environment level name.
            conf (dict): Configuration dictionary for the environment.
        """
        super(CustomDonkeyEnv, self).__init__(level=level, conf=conf)
        
        # Define the observation space based on preprocessed image dimensions
        self.observation_space = gym.spaces.Box(
            low=np.array([-15.0]),  
            high=np.array([15.0]),
            #shape=(3, 60, 80), 
            dtype=np.float32
        )
        
        # Define the action space (steering angle)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]), 
            high=np.array([1.0]), 
            dtype=np.float32
        )
        
        # Initialize maximum cross-track error and target speed
        self.max_cte = conf["max_cte"]
        self.max_steering_angle = 1.0  # Assuming steering angle ranges between -1 and 1
        self.target_speed = 0.5  # Define a target speed for the agent
    
    def step(self, action: np.ndarray) -> tuple:
        """
        Executes a step in the environment with the given action.

        Args:
            action (np.ndarray): The steering action taken by the agent.

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The preprocessed observation.
                - reward (float): The computed reward.
                - done (bool): Flag indicating if the episode has ended.
                - info (dict): Additional information from the environment.
        """
        # Execute the action with a constant throttle of 0.5
        observation, original_reward, done, info = super().step([action[0], 0.5])
        
        # Preprocess the image observation
        obs, _ = process_image(observation)
        
        # Extract relevant information for reward calculation
        cte = info.get('cte', 0.0)  # Cross-track error
        print("cte: ", cte)
        speed = info.get('speed', 0.0)  # Current speed
        collision = info.get('hit', False)  # Collision flag
        
        # Compute the custom reward
        reward = self.compute_reward(cte, action[0], speed, collision, done)
        
        return obs, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment and returns the initial observation.

        Returns:
            np.ndarray: The preprocessed initial observation.
        """
        observation = super().reset()
        obs, _ = process_image(observation)
        return obs
    
    def compute_reward(
        self, 
        cte: float, 
        steering_angle: float, 
        speed: float, 
        collision: bool, 
        done: bool
    ) -> float:
        """
        Computes the reward based on cross-track error, steering, speed, and collision status.

        Args:
            cte (float): Cross-track error.
            steering_angle (float): Current steering angle.
            speed (float): Current speed of the agent.
            collision (bool): Collision status.
            done (bool): Flag indicating if the episode has ended.

        Returns:
            float: The computed reward.
        """
        # Normalize cross-track error
        cte_normalized = cte / self.max_cte
        cte_reward = 1.0 - abs(cte_normalized)
        cte_reward = np.clip(cte_reward, 0.0, 1.0)
        
        # Steering penalty to encourage smooth driving
        steering_penalty = abs(steering_angle) / self.max_steering_angle
        steering_penalty = np.clip(steering_penalty, 0.0, 1.0)
        
        # Speed reward to encourage maintaining target speed
        speed_error = abs(speed - self.target_speed) / self.target_speed
        speed_reward = 1.0 - speed_error
        speed_reward = np.clip(speed_reward, 0.0, 1.0)
        
        # Collision penalty
        collision_penalty = -100.0 if (collision or done) else 0.0
        
        # Aggregate rewards with appropriate weighting
        total_reward = (
            1.0 * cte_reward - 
            0.0 * steering_penalty + 
            0.0 * speed_reward + 
            0.0 * collision_penalty
        )
        
        return total_reward

# ================================
# 5. Custom Feature Extractor
# ================================

class NvidiaCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor inspired by NVIDIA's architecture for processing image inputs.
    """
    def __init__(self, observation_space: gym.spaces.Box):
        """
        Initializes the feature extractor.

        Args:
            observation_space (gym.spaces.Box): The observation space of the environment.
        """
        # Calculate the number of features after the CNN layers
        super(NvidiaCNNExtractor, self).__init__(observation_space, features_dim=100)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # Output: (24, 28, 38)
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # Output: (36, 12, 17)
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # Output: (48, 4, 7)
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # Output: (64, 2, 5)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 5, 100),  # Adjust based on the output dimensions
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            observations (torch.Tensor): Batch of preprocessed observations.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self.cnn(observations)

# ================================
# 6. Custom Callbacks
# ================================

class AverageRewardCallback(BaseCallback):
    """
    Custom callback for logging and printing the average reward per episode.
    """
    def __init__(self, verbose: int = 0):
        """
        Initializes the callback.

        Args:
            verbose (int): Verbosity level. 0 = no output, 1 = info messages.
        """
        super(AverageRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """
        Called at every environment step.

        Returns:
            bool: Whether training should continue.
        """
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info.keys():
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                avg_reward = np.mean(self.episode_rewards)
                
                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"Reward = {episode_reward:.2f}, "
                          f"Average Reward = {avg_reward:.2f}, "
                          f"Length = {episode_length}")
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        """
        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards)
            print(f"Training completed. Final average reward: {final_avg_reward:.2f}")

class SACLossLogger(BaseCallback):
    """
    Custom callback for logging SAC loss components to TensorBoard.
    """
    def __init__(self, log_dir: str = "./sac_losses_tensorboard/", verbose: int = 0):
        """
        Initializes the callback.

        Args:
            log_dir (str): Directory to save TensorBoard logs.
            verbose (int): Verbosity level.
        """
        super(SACLossLogger, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0
    
    def _on_step(self) -> bool:
        """
        Called at every environment step.

        Returns:
            bool: Whether training should continue.
        """
        if hasattr(self.model, 'policy'):
            policy = self.model.policy
            # Access loss attributes if they exist
            critic_loss = getattr(policy, 'critic_loss', None)
            actor_loss = getattr(policy, 'actor_loss', None)
            entropy_loss = getattr(policy, 'alpha_loss', None)
            
            if critic_loss is not None:
                self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.step_count)
            if actor_loss is not None:
                self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.step_count)
            if entropy_loss is not None:
                self.writer.add_scalar('Loss/Entropy', entropy_loss.item(), self.step_count)
            
            self.step_count += 1
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        """
        self.writer.close()
        if self.verbose > 0:
            print("Training completed. Losses have been logged to TensorBoard.")

# ================================
# 7. Model Initialization and Training
# ================================

def main():
    """
    Main function to initialize the environment, model, callbacks, and start training.
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
    
    # Apply normalization to observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Define policy keyword arguments with custom feature extractor
    #policy_kwargs = dict(
    #    features_extractor_class=NvidiaCNNExtractor,
    #)
    
    # Initialize the SAC model
    model = SAC(
        policy='MlpPolicy',  # Use 'CnnPolicy' for image-based observations
        env=env,
        learning_rate=3e-4,
        buffer_size=60000,
        learning_starts=1,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        use_sde_at_warmup=False,
        #policy_kwargs=policy_kwargs,
        verbose=1,  # Set verbosity to see SB3's logging
        seed=seed,
        device="auto",
        tensorboard_log="./sac_donkeycar_tensorboard/",
    )
    
    # Initialize callbacks
    average_reward_callback = AverageRewardCallback(verbose=1)
    sac_loss_logger = SACLossLogger(log_dir="./sac_losses_tensorboard/", verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,                              # Save every 5000 steps
        save_path='./sac_donkeycar_checkpoints/',    # Directory to save checkpoints
        name_prefix='sac_donkeycar',
        verbose=2                                    # Verbosity level
    )
    
    # Combine callbacks into a CallbackList
    callbacks = CallbackList([average_reward_callback, sac_loss_logger, checkpoint_callback])
    
    # Start training
    total_timesteps = 20000  # Replace with desired number of timesteps
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    # Save the final model
    model.save("sac_donkeycar_final")
    print("Model saved as 'sac_donkeycar_final'")
    
    # Save the VecNormalize statistics for future use
    env.save("vecnormalize.pkl")
    print("VecNormalize statistics saved as 'vecnormalize.pkl'")
    
    # Close the environment
    env.close()
    print("Environment closed.")

# ================================
# 8. Entry Point
# ================================

if __name__ == "__main__":
    main()
