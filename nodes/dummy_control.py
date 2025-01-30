import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn

class DrivingModel(nn.Module):
    def __init__(self):
        super(DrivingModel, self).__init__()
        # Define your CNN and fully connected layers here

    def forward(self, image, speed, euler_angles):
        # Define the forward pass
        return steering, throttle

class ROSNode:
    def __init__(self):
        rospy.init_node('driving_model_node')
        self.bridge = CvBridge()
        self.model = DrivingModel()
        self.load_model_weights('path_to_weights.pth')

        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        rospy.Subscriber('/vehicle/speed', Float32, self.speed_callback)
        rospy.Subscriber('/vehicle/euler_angles', Float32MultiArray, self.euler_callback)

        self.steering_pub = rospy.Publisher('/vehicle/steering_cmd', Float32, queue_size=10)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', Float32, queue_size=10)

        # Initialize variables to store the latest data
        self.latest_image = None
        self.latest_speed = None
        self.latest_euler_angles = None

    def load_model_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def speed_callback(self, msg):
        self.latest_speed = msg.data

    def euler_callback(self, msg):
        self.latest_euler_angles = msg.data

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.latest_image is not None and self.latest_speed is not None and self.latest_euler_angles is not None:
                # Preprocess inputs
                image_tensor = self.preprocess_image(self.latest_image)
                speed_tensor = torch.tensor([self.latest_speed], dtype=torch.float32)
                euler_tensor = torch.tensor(self.latest_euler_angles, dtype=torch.float32)

                # Run inference
                steering, throttle = self.model(image_tensor, speed_tensor, euler_tensor)

                # Publish commands
                self.steering_pub.publish(steering.item())
                self.throttle_pub.publish(throttle.item())

            rate.sleep()

    def preprocess_image(self, image):
        # Implement image preprocessing steps
        return image_tensor

if __name__ == '__main__':
    node = ROSNode()
    node.run()
