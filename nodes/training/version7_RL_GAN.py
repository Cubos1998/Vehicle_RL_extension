#!usr/bin/ python3
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
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)

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
from torchvision import transforms
# Local Modules
#import gym_donkeycar
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from PIL import Image


# ================================
# 2. Configuration
# ================================

class ResidualBlock(nn.Module):
    """A simple residual block for the generator."""
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
        #"small_looping_course",
        "sandbox_track"
    ]

    env_config = {
        "exe_path": "/home/students/Desktop/Cristian_MA/sdsim_2/sim_NT.x86_64",
        #"exe_path": "/home/cubos98/Desktop/MA/sim/sim_vehicle.x86_64",
        #"exe_path": "/home/cubos98/Desktop/MA/DonkeySimLinux/donkey_sim.x86_64",
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "max_cte": 2.5,
        "frame_skip": 1,
        "cam_resolution": (120, 160, 3), #(240, 320, 4)
        "host": "localhost",
        "port": 9091,
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
            "random_seed": 42,
            "max_cte": 2.5,
            "frame_skip": 1,
            "cam_resolution": (240, 320, 4),
        }
    }
    
    max_cte = env_config["max_cte"]

# ================================
# 3. Preprocessing Functions
# ================================

def preprocess_image(observation: np.ndarray) -> np.ndarray:
    """
    Preprocesses the input image for the SAC agent.

    Steps:
    1. Converts RGBA images to RGB if necessary.
    2. Converts RGB to YUV color space.
    3. Resizes the image to (80, 60).
    4. Normalizes pixel values to [0, 1].
    5. Transposes the image to channel-first format for PyTorch.

    Args:
        observation (np.ndarray): The raw image observation from the environment.

    Returns:
        np.ndarray: The preprocessed image.
    """
    # Convert RGBA to RGB if necessary
    if observation.shape[2] == 4:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB) 

    #print("Observation shape after preprocessing: ", observation.shape) #(240, 320, 3)
    #time.sleep(10)

    # Convert RGB to YUV color space
    #observation = cv2.cvtColor(observation, cv2.COLOR_RGB2YUV)

    ####################################
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    ####################################

    # 1. Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dir_path + "/CycleGAN/CarF_netG_AB_epoch_9.pth"

    # 2. Instantiate the Generator, load weights
    netG_AB = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6).to(device)
    netG_AB.load_state_dict(torch.load(model, map_location=device))
    netG_AB.eval()

    # 3. Define the same transforms you used in training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])  # transforms from [0,1] -> [-1,1]
    ])

    # 4. Load and transform the input image
    input_img = Image.fromarray(observation, "RGB")
    input_tensor = transform(input_img).unsqueeze(0).to(device)  # shape: (1,3,H,W)

    #print("size: ", input_tensor.shape)

    # 5. Generate with the loaded model (inference)
    with torch.no_grad():
        fake_car = netG_AB(input_tensor)  # shape: (1,3,H,W), range ~ [-1,1]

    # 6. "Denormalize" from [-1,1] back to [0,1]
    fake_car = (fake_car * 0.5) + 0.5  # shape: (1,3,H,W), now in [0,1]

    # 7. Convert tensor to PIL image
    #    1) remove batch dimension
    #    2) clamp to [0,1] just in case
    #    3) turn into PIL
    #output = dir_path + "/CycleGAN/output.png"
    fake_car = fake_car.squeeze(0).cpu().clamp_(0,1)
    out_pil = transforms.ToPILImage()(fake_car)

    # 8. Save the image
    #os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    #out_pil.save(output)
    #print(f"Saved transformed image to {output}")

    #print("Observation shape: ", observation.shape)

    # Resize to (80, 60)

    #observation = cv2.resize(output, (80, 60), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    #observation = observation / 255.0

    # Transpose to channel-first format
    #observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
    #
    # Step 2: Resize to (80, 60)
    img = out_pil.resize((80, 60), Image.Resampling.LANCZOS)
    #print(f"Resized image size: {img.size}")

    # Step 3: Convert to NumPy array and normalize to [0, 1]
    image_array = np.asarray(img).astype(np.float32) / 255.0
    #print(f"Image array shape after normalization: {image_array.shape}, dtype: {image_array.dtype}")

    # Step 4: Transpose to channel-first format (C, H, W)
    transposed_image = np.transpose(image_array, (2, 0, 1))
    #print(f"Transposed image shape: {transposed_image.shape}")

    # Step 5: Ensure dtype is float32
    transposed_image = transposed_image.astype(np.float32)

    observation = transposed_image

    #print("Observation shape after preprocessing: ", observation.shape)

    return observation

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
            low=0.0, 
            high=1.0, 
            shape=(3, 60, 80), 
            dtype=np.float32
        )
        
        # Define the action space (steering angle)  
        """
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, 0.2]), 
            high=np.array([0.5, 0.8]), 
            dtype=np.float32
        )"""

        self.action_space = gym.spaces.Box(
            low=np.array([-0.9]), 
            high=np.array([0.9]), 
            dtype=np.float32
        )
        
        # Initialize maximum cross-track error and target speed
        self.max_cte = conf["max_cte"]
        self.max_steering_angle = 0.5  # Assuming steering angle ranges between -1 and 1
        self.target_speed = 0.8  # Define a target speed for the agent
    
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
        #observation, original_reward, done, info = super().step([action[0], action[1]])
        observation, original_reward, done, info = super().step([action[0], 0.3])
        # Preprocess the image observation
        obs = preprocess_image(observation)
        
        # Extract relevant information for reward calculation
        cte = info.get('cte', 0.0)  # Cross-track error
        speed = info.get('speed', 0.0)  # Current speed
        collision = info.get('hit', False)  # Collision flag
        
        # Compute the custom reward
        reward = self.compute_reward(cte, action[0], speed, collision, done)
        print(time.time())
        
        return obs, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment and returns the initial observation.

        Returns:
            np.ndarray: The preprocessed initial observation.
        """
        observation = super().reset()
        return preprocess_image(observation)
    
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
        print("CTE: ", cte)
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

class SavePreprocessedObservationCallback(BaseCallback):
    """
    Callback for saving the post-processed observations to a directory every 'save_freq' steps.
    Assumes your environment observations are already scaled to [0, 1] with shape (C, H, W).
    """
    def __init__(self, save_freq=1000, save_dir="./saved_observations", verbose=1):
        """
        :param save_freq: (int) How often (in timesteps) we save an image.
        :param save_dir: (str) Directory to save the images.
        :param verbose: (int) Verbosity level.
        """
        super(SavePreprocessedObservationCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method is called at every step. We check if `n_calls` (the number of calls)
        is a multiple of `save_freq`. If so, we save the observation.
        """
        # 'new_obs' in self.locals is the observation *after* the current step
        # Shape is typically (n_envs, C, H, W) when using a VecEnv
        current_obs = self.locals["new_obs"]  # A numpy array or tensor

        # Check if the callback has been called a multiple of save_freq steps
        if self.n_calls % self.save_freq == 0:
            # We only save the observation from the first environment in the vectorized env
            obs = current_obs[0]  # Shape: (C, H, W)
            
            # Convert from (C, H, W) to (H, W, C)
            obs = np.transpose(obs, (1, 2, 0))  # Now shape: (H, W, C)
            
            # Scale up to [0, 255] if needed (assuming your obs is in [0, 1])
            obs = (obs * 255.0).clip(0, 255).astype(np.uint8)
            
            # Convert from RGB to BGR for OpenCV
            obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            # Create a filename
            filename = os.path.join(self.save_dir, f"step_{self.n_calls}.png")
            # Save the image
            cv2.imwrite(filename, obs_bgr)

            if self.verbose > 0:
                print(f"[SaveObsCallback] Saved post-processed observation at step={self.n_calls} to {filename}")
        
        return True  # Return False to stop training early


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
    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Define policy keyword arguments with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=NvidiaCNNExtractor,
    )
    
    # Initialize callbacks
    average_reward_callback = AverageRewardCallback(verbose=1)
    sac_loss_logger = SACLossLogger(log_dir="./sac_losses_tensorboard_GAN_pc/", verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,                              # Save every 5000 steps
        save_path='./sac_donkeycar_checkpoints_GAN_pc/',    # Directory to save checkpoints
        name_prefix='sac_donkeycar_GAN_pc_try2',
        verbose=2                                    # Verbosity level
    )

    save_observation_callback = SavePreprocessedObservationCallback(
        save_freq=1,
        save_dir="./saved_preprocessed_observations",
        verbose=1
    )
    
    # Combine callbacks into a CallbackList
    callbacks = CallbackList([sac_loss_logger, checkpoint_callback])

    # Path to the last checkpoint (replace with your actual path)
    #checkpoint_path = "./sac_donkeycar_checkpoints_GAN_pc/sac_donkeycar_GAN_pc_200000_steps.zip"
    checkpoint_path = "./sac_donkeycar_checkpoints_GAN_pc/sac_donkeycar_GAN_pc_X_steps.zip"
    
    # Check if a checkpoint exists
    try:
        #env = VecNormalize.load("vecnormalize_53.pkl", env)
        # Load the model from the checkpoint
        model = SAC.load(checkpoint_path, env=env)
        print(f"Successfully loaded model from checkpoint: {checkpoint_path}")
        # Access the optimizer and change the learning rate
        new_learning_rate = 2e-4  # Set your desired learning rate
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        print(f"Learning rate updated to {new_learning_rate}")

    except FileNotFoundError:

        #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        # Initialize the SAC model
        model = SAC(
            policy='MlpPolicy',  # Use 'CnnPolicy' for image-based observations
            env=env,
            learning_rate=5.3e-4,
            buffer_size=60000,
            learning_starts=1,
            batch_size=256,
            tau=0.02,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef=0.25,
            target_update_interval=1,
            #target_entropy=-0.5,
            use_sde=False,
            use_sde_at_warmup=False,
            policy_kwargs=policy_kwargs,
            verbose=1,  # Set verbosity to see SB3's logging
            seed=seed,
            device="auto",
            tensorboard_log="./sac_donkeycar_tensorboard_GAN_pc/",
        )
    
    # Start training
    total_timesteps = 1000000  # Replace with desired number of timesteps
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    # Save the final model
    model.save("sac_donkeycar_new_track")
    print("Model saved as 'sac_donkeycar3'")
    
    # Save the VecNormalize statistics for future use
    #env.save("vecnormalize_new_track.pkl")
    #print("VecNormalize statistics saved as 'vecnormalize3.pkl'")
    
    # Close the environment
    env.close()
    print("Environment closed.")

# ================================
# 8. Entry Point
# ================================

if __name__ == "__main__":
    main()
