#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import time

from PIL import Image
from sensor_msgs.msg import Image as SensorImage
from mixed_reality.msg import Control
from cv_bridge import CvBridge

from stable_baselines3 import SAC  # Import the Stable-Baselines3 SAC

# Path to the Stable-Baselines3 SAC model
MODEL_PATH = '/home/cubos98/catkin_ws/src/Vehicle/final_models/very_very_good_vanilla_m1.zip'

FIXED_THROTTLE = True
STEERING = 0
THROTTLE = 1

model = None
pub_throttle_steering = None
bridge = CvBridge()

prev_time = None

def preprocess_image(self, observation: np.ndarray) -> np.ndarray:
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

    # Convert RGB to YUV color space
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2YUV)

    #print("Observation shape: ", observation.shape)

    # Resize to (80, 60)
    observation = cv2.resize(observation, (80, 60), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    observation = observation / 255.0

    # Transpose to channel-first format
    observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)

    #print("Observation shape after preprocessing: ", observation.shape)

    return observation

def new_image(msg):

    global prev_time
    now = time.time()
    if prev_time is not None and (now - prev_time) < 0.00:
        return
    prev_time = now

    #if prev_time is None:
    #    prev_time = time.time()
    #elif prev_time - time.time() < 0.2:
    #    prev_time = time.time()
    #else:
    #    return
    
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
        throttle = 0.65

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
    rate = rospy.Rate(20)  # Set the desired frequency to 3 Hz
    rospy.Subscriber("/sim/image", SensorImage, new_image)

    while not rospy.is_shutdown():
        # Let ROS handle the callbacks and maintain a 3 Hz loop
        rate.sleep()

if __name__ == '__main__':
    try:
        model_node()
    except rospy.ROSInterruptException:
        pass
