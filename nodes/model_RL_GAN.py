#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import time
import torch
import os


from PIL import Image
from sensor_msgs.msg import Image as SensorImage
from mixed_reality.msg import Control
from cv_bridge import CvBridge

import torch.nn as nn
from torchvision import transforms
from PIL import Image
from stable_baselines3 import SAC  # Import the Stable-Baselines3 SAC


# Path to the Stable-Baselines3 SAC model
MODEL_PATH = '/home/cubos98/catkin_ws/src/Vehicle/final_models/model_Ch_lr0.00053_ent0.25_tau0.02_gamma0.99_bs256_throttle0.45_50000_steps.zip'
FIXED_THROTTLE = True
STEERING = 0
THROTTLE = 1

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
    
model = None
pub_throttle_steering = None
bridge = CvBridge()

prev_time = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Adjust this path to your own environment if needed:
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(dir_path, "CycleGAN", "CarF_netG_AB_epoch_9.pth")

netG_AB = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6).to(device)
netG_AB.load_state_dict(torch.load(model_path, map_location=device))
netG_AB.eval()

# Transforms for the CycleGAN input
cyclegan_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])
])

def preprocess_image(self, observation: np.ndarray) -> np.ndarray:
    """
    Runs CycleGAN on the raw observation, then resizes to (80, 60), normalizes to [0,1],
    and returns a (C, H, W) float32 array.
    """
    global device
    global netG_AB
    global cyclegan_transform

    # Convert RGBA to RGB if needed
    if observation.shape[2] == 4:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

    # Convert to PIL, apply CycleGAN transformation
    input_img = Image.fromarray(observation, "RGB")
    input_tensor = cyclegan_transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_car = netG_AB(input_tensor)  # shape: (1,3,H,W), in [-1,1]

    # Convert from [-1,1] to [0,1]
    fake_car = (fake_car * 0.5) + 0.5
    fake_car = fake_car.squeeze(0).cpu().clamp_(0, 1)
    out_pil = transforms.ToPILImage()(fake_car)

    # Resize to (80,60)
    out_pil = out_pil.resize((80, 60), Image.Resampling.LANCZOS)
    obs_np = np.asarray(out_pil, dtype=np.float32) / 255.0  # in [0,1]
    obs_np = np.transpose(obs_np, (2, 0, 1))  # (C, H, W)

    return obs_np

def new_image(msg):

    #global prev_time
    #now = time.time()
    #if prev_time is not None and (now - prev_time) < 0.0:
    #    return
    #prev_time = now
    
    global FIXED_THROTTLE
    global model
    global pub_throttle_steering
    global bridge

    # Convert ROS image to OpenCV image
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
    #image = image[None, ...]
    image = preprocess_image(image)

    # Predict actions using the SAC model
    action, _ = model.predict(image, deterministic=True)
    print(f"Action: {action}")
    steering = action[0]

    if FIXED_THROTTLE:
        throttle = 0.15

    # Publish throttle and steering commands
    if pub_throttle_steering is None:
        pub_throttle_steering = rospy.Publisher("model/throttle_steering", Control, queue_size=10)
    msg = Control()
    msg.throttle = throttle
    msg.steering = steering
    msg.brake = False
    msg.reverse = False
    msg.stopping = False
    #msg.header.stamp = rospy.Time.now()
    
    pub_throttle_steering.publish(msg)

def model_node():
    global MODEL_PATH
    global model

    print(f"Loading model: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH, device="cpu")  # Load the Stable-Baselines3 SAC model
    print("Model loaded successfully.")

    rospy.init_node("model_node", anonymous=True)
    rate = rospy.Rate(20)  # Set the desired frequency to 20 Hz
    rospy.Subscriber("/sim/image", SensorImage, new_image)
    #rospy.Subscriber("/camera", SensorImage, new_image)

    while not rospy.is_shutdown():
        # Let ROS handle the callbacks and maintain a 3 Hz loop
        rate.sleep()

if __name__ == '__main__':
    try:
        model_node()
    except rospy.ROSInterruptException:
        pass
