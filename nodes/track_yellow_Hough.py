import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_yellow_lanes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Convert to HSV for yellow detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])  
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow lanes
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the image
    yellow_lanes = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Canny Edge Detection
    edges = cv2.Canny(yellow_mask, threshold1=50, threshold2=150)

    # Hough Line Transform to detect lane lines
    lines_image = image.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue lines for lanes

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(yellow_mask, cmap="gray")
    axs[1].set_title("Yellow Lane Mask")
    axs[1].axis("off")

    axs[2].imshow(lines_image)
    axs[2].set_title("Tracked Yellow Lanes")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

# Specify the path to your road image
image_path = "/home/cubos98/catkin_ws/src/Vehicle/first_image_received.jpg"  # Replace with the path to your road image
track_yellow_lanes(image_path)
