#!usr/bin/env python3
"""
Soft Actor-Critic (SAC) Implementation for Line Following in DonkeyCar Environment
Using a CycleGAN to transform observations before feeding them to the agent.

Author: Cristian Cubides-Herrera
Date: 2024-11-23
Modified for performance optimization
"""

# =========================================
# 1. Imports
# =========================================
import argparse
import random
import uuid
import time
import os
import sys

# Optionally comment this out if it's not needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# If using donkeycar:
# from gym_donkeycar.envs.donkey_env import DonkeyEnv
# If you have local donkeycar environment code, import from there:
from gym_donkeycar.envs.donkey_env import DonkeyEnv

# =========================================
# 2. Configuration Classes
# =========================================

class ResidualBlock(nn.Module):
    """A simple residual block for the CycleGAN generator."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    CycleGAN Generator:
    Uses downsampling, residual blocks, and upsampling.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6):
        super(Generator, self).__init__()
        # Initial convolution block
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling (2 times)
        in_channels = ngf
        out_channels = ngf * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling (2 times)
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels //= 2

        # Output layer
        model += [
            nn.Conv2d(in_channels, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

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
        "sandbox_track",
    ]

    env_config = {
        "exe_path": "/home/students/Desktop/Cristian_MA/sdsim_2/sim_NT.x86_64",
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        #"max_cte": 2.5,
        "max_cte": 5.0,
        "frame_skip": 1,
        "cam_resolution": (120, 160, 3),
        "steer_limit": 1.0,
        "throttle_min": 0.0,
        "throttle_max": 1.0,
        "conf": {
            "car_name": "SAC",
            "race": False,
            "racer_name": "SAC_agent",
            "country": "Germany",
            "bio": "Learning to drive with SAC",
            "guid": str(uuid.uuid4()),
            "random_seed": random.randint(0, 10000),
            "max_cte": 2.5,
            "frame_skip": 1,
            "cam_resolution": (240, 320, 4),
        }
    }
    max_cte = env_config["max_cte"]

# =========================================
# 3. Custom Environment
# =========================================

class CustomDonkeyEnv(DonkeyEnv):
    """
    Custom DonkeyCar environment with image preprocessing via CycleGAN
    and a tailored reward function for line tracking.
    """
    def __init__(self, level: str, conf: dict, throttle: int, verbose=0):
        """
        Args:
            level (str): environment level name
            conf (dict): config dictionary
            verbose (int): if > 0, prints additional logs
        """
        super(CustomDonkeyEnv, self).__init__(level=level, conf=conf)
        self.verbose = verbose

    

        self.throttle = throttle
        self.distance_traveled = 0.0
        self.prev_pos = None  # will store (x, y, z)

        # Define the observation space for (3, 60, 80) in [0,1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, 60, 80),
            dtype=np.float32
        )

        # Define the action space: only steering ([-0.9, 0.9])
        self.action_space = gym.spaces.Box(
            low=np.array([-0.9]),
            high=np.array([0.9]),
            dtype=np.float32
        )

        # Environment parameters
        self.max_cte = conf["max_cte"]
        self.max_steering_angle = 0.5  # for normalizing steering penalty
        self.target_speed = 0.8 # No need now

        # --------------------------------------------------
        #  Load CycleGAN model ONCE (instead of every step)
        # --------------------------------------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Adjust this path to your own environment if needed:
        dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(dir_path, "CycleGAN", "CarF_netG_AB_epoch_9.pth")

        self.netG_AB = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6).to(self.device)
        self.netG_AB.load_state_dict(torch.load(model_path, map_location=self.device))
        self.netG_AB.eval()

        # Transforms for the CycleGAN input
        self.cyclegan_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def step(self, action: np.ndarray):
        # We'll run with throttle=0.8, for instance
        observation, original_reward, done, info = super().step([action[0], self.throttle])

        # Preprocess the image (including CycleGAN transformation)
        obs = self._preprocess_image(observation)

        x, y, z = info.get("pos")

        # Current position
        current_pos = np.array([x, y, z])

        if self.prev_pos is not None:
            step_dist = np.linalg.norm(current_pos - self.prev_pos)
            self.distance_traveled += step_dist

        # Update prev_pos
        self.prev_pos = current_pos

        if done:
            info["distance_traveled"] = self.distance_traveled
            # Reset for next episode
            self.distance_traveled = 0.0
            self.prev_pos = None

        # Calculate custom reward
        cte = info.get('cte', 0.0)
        speed = info.get('speed', 0.0)
        collision = info.get('hit', False)
        reward = self.compute_reward(cte, action[0], speed, collision, done)

        return obs, reward, done, info

    def reset(self):
        observation = super().reset()
        obs = self._preprocess_image(observation)

        # Reset distance each time we start a new episode
        self.distance_traveled = 0.0
        self.prev_pos = None
        return obs

    def _preprocess_image(self, observation: np.ndarray) -> np.ndarray:
        """
        Runs CycleGAN on the raw observation, then resizes to (80, 60), normalizes to [0,1],
        and returns a (C, H, W) float32 array.
        """
        # Convert RGBA to RGB if needed
        if observation.shape[2] == 4:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

        # Convert to PIL, apply CycleGAN transformation
        input_img = Image.fromarray(observation, "RGB")
        input_tensor = self.cyclegan_transform(input_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_car = self.netG_AB(input_tensor)  # shape: (1,3,H,W), in [-1,1]

        # Convert from [-1,1] to [0,1]
        fake_car = (fake_car * 0.5) + 0.5
        fake_car = fake_car.squeeze(0).cpu().clamp_(0, 1)
        out_pil = transforms.ToPILImage()(fake_car)

        # Resize to (80,60)
        out_pil = out_pil.resize((80, 60), Image.Resampling.LANCZOS)
        obs_np = np.asarray(out_pil, dtype=np.float32) / 255.0  # in [0,1]
        obs_np = np.transpose(obs_np, (2, 0, 1))  # (C, H, W)

        return obs_np

    def compute_reward(self, cte, steering_angle, speed, collision, done):
        """
        Compute custom reward for line-tracking.
        """
        # Normalized CTE
        cte_normalized = cte / self.max_cte
        cte_reward = 1.0 - abs(cte_normalized)
        cte_reward = np.clip(cte_reward, 0.0, 1.0)

        # Steering penalty
        steering_penalty = abs(steering_angle) / self.max_steering_angle
        steering_penalty = np.clip(steering_penalty, 0.0, 1.0)

        # Speed reward
        speed_error = abs(speed - self.target_speed) / self.target_speed
        speed_reward = 1.0 - speed_error
        speed_reward = np.clip(speed_reward, 0.0, 1.0)

        # Collision penalty
        collision_penalty = -100.0 if (collision or done) else 0.0

        # Weighted sum of terms (example weighting)
        total_reward = (1.0 * cte_reward
                        - 0.0 * steering_penalty
                        + 0.0 * speed_reward
                        + 0.0 * collision_penalty)

        return total_reward

# =========================================
# 4. Custom Feature Extractor
# =========================================
class NvidiaCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor inspired by NVIDIA's architecture for image inputs.
    """
    def __init__(self, observation_space: gym.spaces.Box):
        super(NvidiaCNNExtractor, self).__init__(observation_space, features_dim=100)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # (24, 28, 38)
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # (36, 12, 17)
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # (48, 4, 7)
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # (64, 2, 5)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 5, 100),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

# =========================================
# 5. Custom Callbacks
# =========================================


class AverageRewardCallback(BaseCallback):
    """
    Logs and prints average reward per episode and also logs
    total distance traveled in each episode.
    """
    def __init__(self, verbose: int = 1):
        super(AverageRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []

    def _on_step(self) -> bool:
        """
        Called at every step; we watch for the end of an episode
        by checking info['episode'] (provided by Monitor wrapper).
        """
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info.keys():
                # 'episode': {'r': total_reward, 'l': total_length}
                episode_reward = info['episode']['r']

                # The environment stored total distance as info["distance_traveled"]
                distance = info.get('distance_traveled', 0.0)

                self.episode_rewards.append(episode_reward)
                self.episode_distances.append(distance)

                avg_reward = np.mean(self.episode_rewards)
                avg_distance = np.mean(self.episode_distances)

                if self.verbose > 0:
                    print(
                        f"Episode {len(self.episode_rewards)}: "
                        f"Reward = {episode_reward:.2f}, "
                        f"Average Reward = {avg_reward:.2f}, "
                        f"Distance = {distance:.2f}, "
                        f"Average Distance = {avg_distance:.2f}"
                    )
                
                # 2) Log to TensorBoard
                # self.model.num_timesteps holds the total number of steps so far
                self.logger.record("env/episode_reward", episode_reward)
                self.logger.record("env/episode_distance", distance)
                self.logger.record("env/avg_reward", avg_reward)
                self.logger.record("env/avg_distance", avg_distance)

        return True

    def _on_training_end(self) -> None:
        """
        Called when training ends (i.e., after model.learn()).
        """
        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards)
            final_avg_distance = np.mean(self.episode_distances)
            print(
                f"Training completed. "
                f"Final average reward: {final_avg_reward:.2f}, "
                f"Final average distance: {final_avg_distance:.2f}"
            )
            

class SavePreprocessedObservationCallback(BaseCallback):
    """
    Saves a post-processed observation to disk every 'save_freq' steps.
    """
    def __init__(self, save_freq=5000, save_dir="./saved_observations", verbose=1):
        super(SavePreprocessedObservationCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        current_obs = self.locals["new_obs"]  # shape: (n_envs, C, H, W)
        if self.n_calls % self.save_freq == 0:
            obs = current_obs[0]  # just the first env
            obs = np.transpose(obs, (1, 2, 0))  # (H, W, C)
            obs = (obs * 255.0).clip(0, 255).astype(np.uint8)
            obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

            filename = os.path.join(self.save_dir, f"step_{self.n_calls}.png")
            cv2.imwrite(filename, obs_bgr)

            if self.verbose > 0:
                print(f"[SaveObsCallback] Saved post-processed observation at step={self.n_calls} to {filename}")
        return True

class SACLossLogger(BaseCallback):
    """
    Logs SAC loss components to TensorBoard, if available.
    """
    def __init__(self, log_dir: str = "./sac_losses_tensorboard/", verbose: int = 0):
        super(SACLossLogger, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0

    def _on_step(self) -> bool:
        if hasattr(self.model, 'policy'):
            policy = self.model.policy
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
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
        if self.verbose > 0:
            print("Training completed. Losses logged to TensorBoard.")

# =========================================
# 6. Main Training Code
# =========================================

def main():
    # ------------------------------------------------------------------------
    # Here we define the hyperparameters to embed in our checkpoint names:
    # ------------------------------------------------------------------------
    custom_lr = 5.3e-4
    custom_ent_coef = 0.25
    custom_tau = 0.02
    custom_gamma = 0.99
    custom_batch_size = 256
    total_timesteps = 1000000
    throttle = 0.45

    # Create a string to embed in the checkpoint/model names
    param_str = f"lr{custom_lr}_ent{custom_ent_coef}_tau{custom_tau}_gamma{custom_gamma}_bs{custom_batch_size}_throttle{throttle}"
    name_prefix = f"model_Ch_{param_str}"  # e.g. model_Ch_lr7.3e-4_ent0.25_...

    # Initialize configuration
    config = DonkeyCarConfig()

    # Set random seed for reproducibility
    seed = config.env_config["conf"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Create the environment
    def make_env():
        def _init():
            env = CustomDonkeyEnv(level=config.env_list[1], conf=config.env_config, throttle = throttle, verbose=0)
            env = Monitor(env)
            return env
        return _init

    env = DummyVecEnv([make_env()])

    # Policy kwargs to use the custom CNN
    policy_kwargs = dict(
        features_extractor_class=NvidiaCNNExtractor,
    )

    # Define callbacks
    average_reward_callback = AverageRewardCallback(verbose=1)
    sac_loss_logger = SACLossLogger(log_dir="./sac_losses_tensorboard_GAN_pc/", verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./sac_donkeycar_checkpoints_GAN_pc/',
        name_prefix=name_prefix,  # <--- Incorporate parameters into the checkpoint name!
        verbose=2
    )
    save_observation_callback = SavePreprocessedObservationCallback(
        save_freq=5000,
        save_dir="./saved_preprocessed_observations",
        verbose=1
    )
    callbacks = CallbackList([
        average_reward_callback,
        #sac_loss_logger,
        checkpoint_callback,
        # Enable or disable as needed:
        #save_observation_callback
    ])

    # Optionally, load from a checkpoint if available:
    checkpoint_path = "./sac_donkeycar_checkpoints_GAN_pc/sac_donkeycar_GAN_pc_try2_X_steps.zip"

    try:
        model = SAC.load(checkpoint_path, env=env)
        print(f"Successfully loaded model from checkpoint: {checkpoint_path}")
        # Customize parameters after loading if desired:
        model.learning_rate = custom_lr
        model.ent_coef = custom_ent_coef

    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=custom_lr,
            buffer_size=70000,
            learning_starts=1,
            batch_size=custom_batch_size,
            tau=custom_tau,
            gamma=custom_gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef=custom_ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device="auto",
            tensorboard_log="./sac_donkeycar_tensorboard_GAN_pc/",
        )

    # ----------------------------------------
    # Print out current hyperparameters
    # ----------------------------------------
    print(f"Starting training with parameters:")
    print(f"  Learning Rate : {model.learning_rate}")
    print(f"  Ent Coef      : {model.ent_coef}")
    print(f"  Batch Size    : {model.batch_size}")
    print(f"  Tau           : {model.tau}")
    print(f"  Gamma         : {model.gamma}")
    print(f"  Throttle      : {throttle}")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    # ----------------------------------------
    # Save final model with the param_str
    # ----------------------------------------
    final_model_name = f"{name_prefix}_final_{total_timesteps}"
    model.save(final_model_name)
    print(f"Model saved as '{final_model_name}'.")

    # Close environment
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
